"""auc de cada hallazgo critico, solo sobre la particion oficial de prueba del nih.

los pesos se entrenaron con datos del nih: medir sobre imagenes de entrenamiento
daria numeros inflados (informe 1, tabla 2).

uso:
  python -m comun.evaluar resultados/cpu/cola.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from comun.tarea import COLUMNA_IMAGEN, ETIQUETAS, HALLAZGOS, RAIZ

PRUEBA = RAIZ / "datos" / "salidas" / "particion_prueba.txt"


def auc(puntajes: np.ndarray, positivos: np.ndarray) -> float:
    """area bajo la curva roc con la formula de mann-whitney (rangos promedio en empates)"""
    n_pos = int(positivos.sum())
    n_neg = len(positivos) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    orden = np.argsort(puntajes, kind="stable")
    rangos = np.empty(len(puntajes))
    valores = puntajes[orden]
    i = 0
    while i < len(valores):
        j = i
        while j + 1 < len(valores) and valores[j + 1] == valores[i]:
            j += 1
        rangos[orden[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((rangos[positivos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m comun.evaluar", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cola", type=Path)
    p.add_argument("--etiquetas", type=Path, default=ETIQUETAS)
    p.add_argument("--prueba", type=Path, default=PRUEBA)
    args = p.parse_args(argv)

    prueba = set(args.prueba.read_text(encoding="utf-8").split())
    with args.etiquetas.open(newline="", encoding="utf-8-sig") as f:
        etiquetas = {fila[COLUMNA_IMAGEN]: set(fila["Finding Labels"].split("|"))
                     for fila in csv.DictReader(f) if fila[COLUMNA_IMAGEN] in prueba}
    with args.cola.open(newline="", encoding="utf-8") as f:
        filas = [fila for fila in csv.DictReader(f) if fila["imagen"] in etiquetas]

    print(f"{len(filas)} imagenes de la particion de prueba")
    critico = np.array([bool(etiquetas[f["imagen"]] & set(HALLAZGOS.values())) for f in filas])
    for nombre, etiqueta in HALLAZGOS.items():
        pos = np.array([etiqueta in etiquetas[f["imagen"]] for f in filas])
        pun = np.array([float(f[nombre]) for f in filas])
        print(f"  auc {nombre:<14}{auc(pun, pos):.3f}  ({pos.sum()} positivos)")
    pun = np.array([float(f["puntaje"]) for f in filas])
    print(f"  auc puntaje vs 'algun hallazgo critico': {auc(pun, critico):.3f}  ({critico.sum()} positivos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
