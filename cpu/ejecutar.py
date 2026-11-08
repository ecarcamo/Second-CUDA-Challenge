"""version cpu completa: leer -> decodificar -> preprocesar -> densenet-121 -> puntaje -> cola.

mide cada etapa por separado: 1 corrida de calentamiento y la mediana de N corridas.

uso:
  python -m cpu.ejecutar                      # subconjunto completo, 1 + 5 corridas
  python -m cpu.ejecutar --limite 500 -r 1    # prueba rapida
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchxrayvision as xrv

from comun.tarea import (ETIQUETAS, EXTRAIDO, HALLAZGOS, PESOS, RAIZ, decodificar,
                         indices_hallazgos, leer_bytes, listar_imagenes)
from comun.model import load_model
from comun.thermal import cool_down
from cpu.preprocesado import preprocesar_lote

ETAPAS = ("lectura", "decodificacion", "preprocesado", "inferencia", "puntaje", "ordenamiento")


def _argumentos(argv):
    p = argparse.ArgumentParser(prog="python -m cpu.ejecutar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extraido", type=Path, default=EXTRAIDO, help="carpeta con los png (default: %(default)s)")
    p.add_argument("--etiquetas", type=Path, default=ETIQUETAS,
                   help="csv filtrado de diego; fija la lista de imagenes (default: %(default)s)")
    p.add_argument("--salida", type=Path, default=RAIZ / "resultados" / "cpu",
                   help="carpeta de resultados (default: %(default)s)")
    p.add_argument("--lote", type=int, default=64, help="imagenes por lote (default: %(default)s)")
    p.add_argument("--hilos", type=int, default=os.cpu_count(),
                   help="hilos para pytorch y para leer/decodificar/preprocesar (default: todos, %(default)s)")
    p.add_argument("-r", "--repeticiones", type=int, default=5, help="corridas medidas (default: %(default)s)")
    p.add_argument("--calentamiento", type=int, default=1, help="corridas sin medir (default: %(default)s)")
    p.add_argument("--limite", type=int, default=None, help="usar solo las primeras N imagenes (pruebas)")
    p.add_argument("--pesos", default=PESOS, help="pesos de torchxrayvision (default: %(default)s)")
    p.add_argument("--sin-pausas", action="store_true", help="no esperar a que la laptop se enfrie entre corridas")
    p.add_argument("--pesos-originales", action="store_true",
                   help="no pasar a cero los pesos desnormalizados (en x86 la inferencia en cpu es mucho mas lenta)")
    return p.parse_args(argv)


def _ram_gb() -> float | None:
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError, AttributeError):
        return None


def _cpu_nombre() -> str:
    if sys.platform == "darwin":
        return os.popen("sysctl -n machdep.cpu.brand_string").read().strip()
    try:
        for linea in Path("/proc/cpuinfo").read_text().splitlines():
            if linea.startswith("model name"):
                return linea.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def corrida(rutas, modelo, columnas, pool, lote, progreso=None):
    """una pasada completa sobre todas las imagenes. devuelve (tiempos por etapa, probs, puntajes, orden)."""
    t = dict.fromkeys(ETAPAS, 0.0)
    n = len(rutas)
    probs = np.empty((n, len(columnas)), dtype=np.float32)
    puntajes = np.empty(n, dtype=np.float32)
    inicio = time.perf_counter()

    for a in range(0, n, lote):
        parte = rutas[a:a + lote]

        t0 = time.perf_counter()
        datos = list(pool.map(leer_bytes, parte))
        t1 = time.perf_counter()
        imgs = list(pool.map(decodificar, datos))
        t2 = time.perf_counter()
        x = preprocesar_lote(imgs, pool)
        t3 = time.perf_counter()
        with torch.inference_mode():
            salida = modelo(torch.from_numpy(x))
        t4 = time.perf_counter()
        criticos = salida[:, columnas].numpy()
        probs[a:a + len(parte)] = criticos
        puntajes[a:a + len(parte)] = criticos.max(axis=1)
        t5 = time.perf_counter()

        t["lectura"] += t1 - t0
        t["decodificacion"] += t2 - t1
        t["preprocesado"] += t3 - t2
        t["inferencia"] += t4 - t3
        t["puntaje"] += t5 - t4
        if progreso:
            progreso(a + len(parte), n, time.perf_counter() - inicio)

    t0 = time.perf_counter()
    # mas urgente primero; estable para que los empates queden en orden de nombre
    orden = np.argsort(-puntajes, kind="stable")
    t["ordenamiento"] = time.perf_counter() - t0
    t["total"] = time.perf_counter() - inicio
    return t, probs, puntajes, orden


def _progreso(etiqueta):
    ultimo = [0.0]

    def mostrar(hechas, total, seg):
        if seg - ultimo[0] < 10 and hechas < total:
            return
        ultimo[0] = seg
        tasa = hechas / seg if seg else 0
        falta = (total - hechas) / tasa if tasa else 0
        print(f"  [{etiqueta}] {hechas}/{total} imagenes, {tasa:.1f} img/s, faltan ~{falta / 60:.1f} min",
              file=sys.stderr, flush=True)
    return mostrar


def escribir_cola(ruta: Path, rutas, probs, puntajes, orden):
    nombres = list(HALLAZGOS)
    with ruta.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["posicion", "imagen", "puntaje", "hallazgo_principal", *nombres])
        for pos, i in enumerate(orden, 1):
            w.writerow([pos, rutas[i].name, f"{puntajes[i]:.6f}", nombres[int(probs[i].argmax())],
                        *(f"{v:.6f}" for v in probs[i])])


def main(argv=None) -> int:
    args = _argumentos(argv)
    torch.set_num_threads(args.hilos)
    cv2.setNumThreads(1)  # cada hilo del pool decodifica una imagen; sin hilos extra de opencv

    rutas = listar_imagenes(args.extraido, args.etiquetas, args.limite)
    modelo, flushed = load_model(args.pesos, args.pesos_originales)
    columnas = indices_hallazgos(list(modelo.pathologies))
    print(f"{len(rutas)} imagenes, lote {args.lote}, {args.hilos} hilos, pesos {args.pesos}", file=sys.stderr)

    corridas, pauses = [], []
    with ThreadPoolExecutor(args.hilos) as pool:
        for k in range(args.calentamiento):
            t, *_ = corrida(rutas, modelo, columnas, pool, args.lote, _progreso(f"calentamiento {k + 1}"))
            print(f"calentamiento {k + 1}: {t['total']:.1f} s", file=sys.stderr)
        for k in range(args.repeticiones):
            if not args.sin_pausas and (k or args.calentamiento):
                pauses.append(cool_down(label=f"corrida {k + 1}"))
            t, probs, puntajes, orden = corrida(rutas, modelo, columnas, pool, args.lote,
                                               _progreso(f"corrida {k + 1}/{args.repeticiones}"))
            corridas.append(t)
            print(f"corrida {k + 1}: {t['total']:.1f} s", file=sys.stderr)

    mediana = {e: statistics.median(c[e] for c in corridas) for e in (*ETAPAS, "total")}
    n = len(rutas)
    resumen = {
        "version": "cpu",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "imagenes": n,
        "lote": args.lote,
        "hilos": args.hilos,
        "hilos_torch": torch.get_num_threads(),
        "pesos": args.pesos,
        "precision": "fp32",
        "pesos_desnormalizados_a_cero": flushed,
        "calentamiento": args.calentamiento,
        "repeticiones": args.repeticiones,
        "equipo": {"cpu": _cpu_nombre(), "nucleos_logicos": os.cpu_count(), "ram_gb": _ram_gb(),
                   "so": platform.platform(), "python": platform.python_version()},
        "librerias": {"torch": torch.__version__, "torchxrayvision": xrv.__version__,
                      "numpy": np.__version__, "opencv": cv2.__version__},
        "mediana_s": mediana,
        "imagenes_por_s": n / mediana["total"],
        "fraccion_por_etapa": {e: mediana[e] / mediana["total"] for e in ETAPAS},
        "pausas_termicas": pauses,
        "corridas_s": corridas,
    }

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "tiempos.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    escribir_cola(args.salida / "cola.csv", rutas, probs, puntajes, orden)

    print(f"\nmediana de {args.repeticiones} corridas ({n} imagenes):")
    for e in (*ETAPAS, "total"):
        print(f"  {e:<15}{mediana[e]:10.3f} s  {100 * mediana[e] / mediana['total']:5.1f} %")
    print(f"  {n / mediana['total']:.1f} imagenes/s")
    print(f"resultados en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
