from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cupy as cp
import cv2
import numpy as np
import torch
import torchxrayvision as xrv

from comun.tarea import (ETIQUETAS, EXTRAIDO, PESOS, RAIZ, TAMANO, decodificar, indices_hallazgos,
                         leer_bytes, listar_imagenes)
from comun.model import load_model
from comun.thermal import cool_down
from cpu.ejecutar import _cpu_nombre, _progreso, _ram_gb, escribir_cola
from gpu.preprocess import Preprocessor

HOST_STAGES = ("lectura", "decodificacion")
GPU_STAGES = ("copia", "preprocesado", "inferencia", "puntaje")
FINAL_STAGES = ("ordenamiento", "copia_resultados")
STAGES = HOST_STAGES + GPU_STAGES + FINAL_STAGES
DTYPES = {"uint8": (torch.uint8, np.uint8), "float32": (torch.float32, np.float32)}

USAGE = """\
version gpu completa: leer -> decodificar (cpu) -> copia a la gpu -> recorte, 224 x 224 y
normalizacion (kernel cupy) -> densenet-121 (cudnn, fp32) -> puntaje -> cola ordenada (gpu).

uso:
  python -m gpu.run                         # subconjunto completo, 1 + 5 corridas
  python -m gpu.run --limite 500 -r 1       # prueba rapida
"""


def parse_args(argv):
    p = argparse.ArgumentParser(prog="python -m gpu.run", description=USAGE,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extraido", type=Path, default=EXTRAIDO, help="carpeta con los png (default: %(default)s)")
    p.add_argument("--etiquetas", type=Path, default=ETIQUETAS,
                   help="csv filtrado de diego; fija la lista de imagenes (default: %(default)s)")
    p.add_argument("--salida", type=Path, default=RAIZ / "resultados" / "gpu",
                   help="carpeta de resultados (default: %(default)s)")
    p.add_argument("--lote", type=int, default=64, help="imagenes por lote (default: %(default)s)")
    p.add_argument("--hilos", type=int, default=os.cpu_count(),
                   help="hilos de la cpu para leer y decodificar (default: todos, %(default)s)")
    p.add_argument("-r", "--repeticiones", type=int, default=5, help="corridas medidas (default: %(default)s)")
    p.add_argument("--calentamiento", type=int, default=1, help="corridas sin medir (default: %(default)s)")
    p.add_argument("--limite", type=int, default=None, help="usar solo las primeras N imagenes (pruebas)")
    p.add_argument("--pesos", default=PESOS, help="pesos de torchxrayvision (default: %(default)s)")
    p.add_argument("--transferencia", choices=DTYPES, default="uint8",
                   help="tipo con el que se copia cada imagen a la gpu (default: %(default)s)")
    p.add_argument("--preprocesado", choices=("kernel", "torch"), default="kernel",
                   help="kernel propio en cupy o la misma operacion en pytorch (default: %(default)s)")
    p.add_argument("--memoria", choices=("fijada", "paginable"), default="fijada",
                   help="memoria del host desde la que se copia (default: %(default)s)")
    p.add_argument("--modo", choices=("etapas", "solapado"), default="etapas",
                   help="etapas: sincroniza en cada lote para medir cada etapa; solapado: decodifica el "
                        "lote siguiente mientras la gpu procesa el actual (default: %(default)s)")
    p.add_argument("--sin-pausas", action="store_true", help="no esperar a que la laptop se enfrie entre corridas")
    p.add_argument("--pesos-originales", action="store_true",
                   help="no pasar a cero los pesos desnormalizados (en x86 la inferencia en cpu es mucho mas lenta)")
    return p.parse_args(argv)


def gpu_info() -> dict:
    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {"gpu": props.name, "vram_gb": round(props.total_memory / 2**30, 2), "driver": driver,
            "cuda_torch": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_runtime_cupy": cp.cuda.runtime.runtimeGetVersion()}


def strict_fp32() -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"


class Pipeline:
    def __init__(self, args, paths, model, columns, pool):
        self.args, self.paths, self.model, self.columns, self.pool = args, paths, model, columns, pool
        self.torch_dtype, self.np_dtype = DTYPES[args.transferencia]
        self.height, self.width = decodificar(leer_bytes(paths[0])).shape
        self.preprocessors = {(self.height, self.width): Preprocessor(self.height, self.width)}
        slots = 2 if args.modo == "solapado" else 1
        pinned = args.memoria == "fijada"
        shape = (args.lote, self.height, self.width)
        self.host = [torch.empty(shape, dtype=self.torch_dtype, pin_memory=pinned) for _ in range(slots)]
        self.host_np = [h.numpy() for h in self.host]
        self.device = [torch.empty(shape, dtype=self.torch_dtype, device="cuda") for _ in range(slots)]
        self.inputs = torch.empty((args.lote, 1, TAMANO, TAMANO), dtype=torch.float32, device="cuda")
        self.copy_done = [None] * slots
        self.copied_bytes = 0

    def preprocessor(self, shape):
        if shape not in self.preprocessors:
            self.preprocessors[shape] = Preprocessor(*shape)
        return self.preprocessors[shape]

    def produce(self, start, slot):
        part = self.paths[start:start + self.args.lote]
        if self.copy_done[slot] is not None:
            self.copy_done[slot].synchronize()
        target = self.host_np[slot]
        odd = {}

        def fill(k):
            img = decodificar(raw[k])
            if img.shape == (self.height, self.width):
                target[k] = img
            else:
                odd[k] = img.astype(self.np_dtype)

        t0 = time.perf_counter()
        raw = list(self.pool.map(leer_bytes, part))
        t1 = time.perf_counter()
        list(self.pool.map(fill, range(len(part))))
        t2 = time.perf_counter()
        return len(part), odd, t1 - t0, t2 - t1

    def process(self, m, odd, slot, start, probs, scores):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
        events[0].record()
        device = self.device[slot][:m]
        device.copy_(self.host[slot][:m], non_blocking=True)
        odd_device = {k: torch.from_numpy(img).cuda(non_blocking=False) for k, img in odd.items()}
        events[1].record()
        x = self.inputs[:m]
        self.preprocessors[(self.height, self.width)](device, x, self.args.preprocesado)
        for k, img in odd_device.items():
            self.preprocessor(tuple(img.shape))(img[None], x[k:k + 1], self.args.preprocesado)
        events[2].record()
        with torch.inference_mode():
            out = self.model(x)
        events[3].record()
        critical = out[:, self.columns]
        probs[start:start + m] = critical
        scores[start:start + m] = critical.amax(dim=1)
        events[4].record()
        self.copy_done[slot] = events[1]
        self.copied_bytes += device.nelement() * device.element_size()
        self.copied_bytes += sum(t.nelement() * t.element_size() for t in odd_device.values())
        return events

    def run_once(self, progress=None):
        n, lote = len(self.paths), self.args.lote
        overlap = self.args.modo == "solapado"
        times = dict.fromkeys(STAGES, 0.0)
        probs = torch.empty((n, len(self.columns)), dtype=torch.float32, device="cuda")
        scores = torch.empty(n, dtype=torch.float32, device="cuda")
        batches = []
        self.copied_bytes = 0
        self.copy_done = [None] * len(self.host)
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with ThreadPoolExecutor(1) as producer:
            future = producer.submit(self.produce, 0, 0) if overlap else None
            for i, start in enumerate(range(0, n, lote)):
                slot = i % len(self.host)
                if overlap:
                    m, odd, t_read, t_decode = future.result()
                    if start + lote < n:
                        future = producer.submit(self.produce, start + lote, (i + 1) % len(self.host))
                else:
                    m, odd, t_read, t_decode = self.produce(start, slot)
                times["lectura"] += t_read
                times["decodificacion"] += t_decode
                batches.append(self.process(m, odd, slot, start, probs, scores))
                if not overlap:
                    torch.cuda.synchronize()
                if progress:
                    progress(start + m, n, time.perf_counter() - start_time)

        final = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        final[0].record()
        order = torch.argsort(-scores, stable=True)
        final[1].record()
        probs_h, scores_h, order_h = probs.cpu().numpy(), scores.cpu().numpy(), order.cpu().numpy()
        final[2].record()
        torch.cuda.synchronize()
        times["total"] = time.perf_counter() - start_time

        for events in batches:
            for stage, a, b in zip(GPU_STAGES, events, events[1:]):
                times[stage] += a.elapsed_time(b) / 1000
        times["ordenamiento"] = final[0].elapsed_time(final[1]) / 1000
        times["copia_resultados"] = final[1].elapsed_time(final[2]) / 1000
        times["bytes_copiados"] = self.copied_bytes
        return times, probs_h, scores_h, order_h


def main(argv=None) -> int:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("no hay gpu con cuda disponible", file=sys.stderr)
        return 1
    torch.set_num_threads(args.hilos)
    cv2.setNumThreads(1)
    strict_fp32()

    paths = listar_imagenes(args.extraido, args.etiquetas, args.limite)
    model, flushed = load_model(args.pesos, args.pesos_originales)
    model = model.cuda()
    columns = indices_hallazgos(list(model.pathologies))
    print(f"{len(paths)} imagenes, lote {args.lote}, {args.hilos} hilos, pesos {args.pesos}, "
          f"transferencia {args.transferencia}, preprocesado {args.preprocesado}, memoria {args.memoria}, "
          f"modo {args.modo}", file=sys.stderr)

    runs, pauses = [], []
    with ThreadPoolExecutor(args.hilos) as pool:
        pipeline = Pipeline(args, paths, model, columns, pool)
        torch.cuda.reset_peak_memory_stats()
        for k in range(args.calentamiento):
            t, *_ = pipeline.run_once(_progreso(f"calentamiento {k + 1}"))
            print(f"calentamiento {k + 1}: {t['total']:.1f} s", file=sys.stderr)
        for k in range(args.repeticiones):
            if not args.sin_pausas and (k or args.calentamiento):
                pauses.append(cool_down(min_s=30, label=f"corrida {k + 1}"))
            t, probs, scores, order = pipeline.run_once(_progreso(f"corrida {k + 1}/{args.repeticiones}"))
            runs.append(t)
            print(f"corrida {k + 1}: {t['total']:.1f} s", file=sys.stderr)

    median = {e: statistics.median(r[e] for r in runs) for e in (*STAGES, "total")}
    n = len(paths)
    copied = runs[-1]["bytes_copiados"]
    summary = {
        "version": "gpu",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "imagenes": n,
        "lote": args.lote,
        "hilos": args.hilos,
        "pesos": args.pesos,
        "precision": "fp32 (tf32 apagado)",
        "pesos_desnormalizados_a_cero": flushed,
        "transferencia": args.transferencia,
        "preprocesado": args.preprocesado,
        "memoria_host": args.memoria,
        "modo": args.modo,
        "calentamiento": args.calentamiento,
        "repeticiones": args.repeticiones,
        "equipo": {"cpu": _cpu_nombre(), "nucleos_logicos": os.cpu_count(), "ram_gb": _ram_gb(),
                   "so": platform.platform(), "python": platform.python_version(), **gpu_info()},
        "librerias": {"torch": torch.__version__, "torchxrayvision": xrv.__version__, "cupy": cp.__version__,
                      "numpy": np.__version__, "opencv": cv2.__version__},
        "mediana_s": median,
        "imagenes_por_s": n / median["total"],
        "fraccion_por_etapa": {e: median[e] / median["total"] for e in STAGES},
        "bytes_copiados_a_gpu": copied,
        "ancho_de_banda_copia_gb_s": copied / median["copia"] / 1e9 if median["copia"] else None,
        "vram_pico_asignada_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "vram_pico_reservada_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "pausas_termicas": pauses,
        "corridas_s": runs,
    }

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "tiempos.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    escribir_cola(args.salida / "cola.csv", paths, probs, scores, order)

    print(f"\nmediana de {args.repeticiones} corridas ({n} imagenes):")
    for e in (*STAGES, "total"):
        print(f"  {e:<17}{median[e]:10.3f} s  {100 * median[e] / median['total']:5.1f} %")
    print(f"  {n / median['total']:.1f} imagenes/s, copia {summary['ancho_de_banda_copia_gb_s'] or 0:.1f} GB/s, "
          f"vram pico {summary['vram_pico_reservada_mib']} MiB")
    print(f"resultados en {args.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
