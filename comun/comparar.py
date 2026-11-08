"""compara dos colas (cpu y gpu) para comprobar que resolvieron la misma tarea.

uso:
  python -m comun.comparar resultados/cpu/cola.csv resultados/gpu/cola.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from comun.tarea import HALLAZGOS


def leer(ruta: Path) -> dict[str, dict]:
    with ruta.open(newline="", encoding="utf-8") as f:
        return {fila["imagen"]: fila for fila in csv.DictReader(f)}


def comparar(a: dict[str, dict], b: dict[str, dict], top: int = 100) -> dict:
    if set(a) != set(b):
        raise ValueError(f"las colas no tienen las mismas imagenes ({len(a)} frente a {len(b)})")
    nombres = sorted(a)
    cols = ["puntaje", *HALLAZGOS]
    va = np.array([[float(a[n][c]) for c in cols] for n in nombres])
    vb = np.array([[float(b[n][c]) for c in cols] for n in nombres])
    dif = np.abs(va - vb)
    pos_a = np.array([int(a[n]["posicion"]) for n in nombres])
    pos_b = np.array([int(b[n]["posicion"]) for n in nombres])
    top_a = {n for n in nombres if int(a[n]["posicion"]) <= top}
    top_b = {n for n in nombres if int(b[n]["posicion"]) <= top}
    # correlacion de spearman entre posiciones (las posiciones ya son rangos)
    spearman = float(np.corrcoef(pos_a, pos_b)[0, 1])
    return {
        "imagenes": len(nombres),
        "dif_max_puntaje": float(dif[:, 0].max()),
        "dif_media_puntaje": float(dif[:, 0].mean()),
        "dif_max_por_hallazgo": {h: float(dif[:, i + 1].max()) for i, h in enumerate(HALLAZGOS)},
        "misma_posicion": float((pos_a == pos_b).mean()),
        "dif_max_posicion": int(np.abs(pos_a - pos_b).max()),
        "spearman_posiciones": spearman,
        f"coinciden_top_{top}": len(top_a & top_b),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m comun.comparar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cola_a", type=Path)
    p.add_argument("cola_b", type=Path)
    p.add_argument("--top", type=int, default=100, help="tamano del grupo mas urgente a comparar")
    args = p.parse_args(argv)
    r = comparar(leer(args.cola_a), leer(args.cola_b), args.top)
    for k, v in r.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
