"""prepara el subconjunto images_001..images_005 del nih (informe 1, seccion 2)"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from datos import archivos, config, etiquetas, imagenes
from datos.errores import ErrorDatos

BASE = Path(__file__).resolve().parent

DESCRIPCION = """\
descarga (opcional), descomprime, verifica y filtra etiquetas del subconjunto.
antes de correrlo hay que llenar datos/dataset.json con los datos reales del nih.

salidas en --salida:
  reporte.json               d-e2: conteos, modos, dimensiones, ilegibles, positivos
  etiquetas_subconjunto.csv  d-e3: csv filtrado, una fila por imagen
  particion_prueba.txt       d-e4: imagenes del subconjunto en la particion de prueba
"""

EPILOGO = """\
codigos de salida:
  0  todo bien
  1  error de config, archivos o csv (no se generan salidas)
  2  corrio completo, pero hay problemas en los datos (ver 'problemas' en reporte.json)

ejemplo:
  python -m datos.preparar_subconjunto --csv RUTA_CSV --lista-prueba RUTA_LISTA
"""


def _argumentos(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m datos.preparar_subconjunto",
        description=DESCRIPCION,
        epilog=EPILOGO,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=BASE / "dataset.json",
                   help="json con los datos del nih (default: %(default)s)")
    p.add_argument("--archivos", type=Path, default=BASE / "archivos_nih",
                   help="carpeta con los comprimidos, o donde se bajan (default: %(default)s)")
    p.add_argument("--extraido", type=Path, default=BASE / "extraido",
                   help="carpeta donde se descomprimen los png (default: %(default)s)")
    p.add_argument("--salida", type=Path, default=BASE / "salidas",
                   help="carpeta de los resultados (default: %(default)s)")
    p.add_argument("--csv", type=Path, required=True, help="csv de etiquetas del nih")
    p.add_argument("--lista-prueba", type=Path, required=True,
                   help="lista oficial de la particion de prueba del nih")
    p.add_argument("--descargar", action="store_true",
                   help="baja primero lo que falte usando 'urls' del config")
    p.add_argument("--omitir-extraccion", action="store_true",
                   help="reusa lo que ya esta en --extraido en vez de descomprimir")
    return p.parse_args(argv)


def _extraer(rutas: list[Path], destino: Path) -> dict:
    # no se borra nada por el usuario: si hay algo, que decida el
    if destino.exists() and any(destino.iterdir()):
        raise ErrorDatos(f"{destino} no esta vacia; vaciala o usa --omitir-extraccion")
    total = {"pngs": 0, "otros_ignorados": 0}
    for ruta in rutas:
        pngs, otros = archivos.extraer_pngs(ruta, destino)
        total["pngs"] += pngs
        total["otros_ignorados"] += otros
    return total


def _problemas(img: imagenes.ResultadoImagenes, sin_fila: list[str], extraccion: dict | None) -> list[str]:
    problemas = []
    if img.total == 0:
        problemas.append("no hay png en la carpeta de extraccion")
    if extraccion and extraccion["pngs"] != img.total:
        problemas.append(f"se extrajeron {extraccion['pngs']} png pero hay {img.total} en la carpeta")
    repetidos = sorted(n for n, c in Counter(img.nombres).items() if c > 1)
    if repetidos:
        problemas.append(f"{len(repetidos)} nombres de imagen repetidos entre carpetas")
    if img.ilegibles:
        problemas.append(f"{len(img.ilegibles)} imagenes ilegibles")
    if sin_fila:
        problemas.append(f"{len(sin_fila)} imagenes sin fila en el csv")
    return problemas


def preparar(args: argparse.Namespace) -> dict:
    cfg = config.cargar(args.config)
    if args.descargar:
        if not cfg.urls:
            raise ErrorDatos("--descargar necesita 'urls' en el config")
        archivos.descargar(cfg.urls, args.archivos)

    rutas = archivos.validar_presentes(args.archivos, cfg.archivos)
    for ruta in rutas:
        if ruta.name in cfg.checksums:
            archivos.verificar_checksum(ruta, cfg.checksum_algoritmo, cfg.checksums[ruta.name])

    extraccion = None if args.omitir_extraccion else _extraer(rutas, args.extraido)
    img = imagenes.verificar(args.extraido)

    columnas, filas = etiquetas.leer_csv(args.csv, cfg)
    filtradas, sin_fila = etiquetas.filtrar(filas, img.nombres)
    prueba = sorted(etiquetas.leer_lista_prueba(args.lista_prueba) & set(img.nombres))

    reporte = {
        "archivos": {
            "tamano_comprimido_bytes": sum(r.stat().st_size for r in rutas),
            "por_archivo": {r.name: r.stat().st_size for r in rutas},
            "extraccion": extraccion,
        },
        "imagenes": {"total": img.total, **{k: v for k, v in asdict(img).items() if k != "nombres"}},
        "etiquetas": {
            "filas_subconjunto": len(filtradas),
            "imagenes_sin_fila": sin_fila,
            "positivos": etiquetas.contar_positivos(filtradas, cfg),
        },
        "particion_prueba": {
            "total": len(prueba),
            "positivos": etiquetas.contar_positivos((filas[n] for n in prueba if n in filas), cfg),
        },
        "problemas": _problemas(img, sin_fila, extraccion),
    }

    args.salida.mkdir(parents=True, exist_ok=True)
    (args.salida / "reporte.json").write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    etiquetas.escribir_csv(args.salida / "etiquetas_subconjunto.csv", columnas, filtradas)
    (args.salida / "particion_prueba.txt").write_text(
        "".join(f"{n}\n" for n in prueba), encoding="utf-8"
    )
    return reporte


def main(argv: list[str] | None = None) -> int:
    args = _argumentos(argv)
    try:
        reporte = preparar(args)
    except ErrorDatos as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for problema in reporte["problemas"]:
        print(f"problema: {problema}", file=sys.stderr)
    print(f"listo: {reporte['imagenes']['total']} imagenes, salidas en {args.salida}")
    return 2 if reporte["problemas"] else 0


if __name__ == "__main__":
    sys.exit(main())
