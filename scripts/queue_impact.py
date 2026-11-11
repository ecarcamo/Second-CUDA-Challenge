import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINDINGS = {"neumotorax": "Pneumothorax", "edema": "Edema", "neumonia": "Pneumonia",
            "derrame": "Effusion", "consolidacion": "Consolidation"}
QUEUE_SIZE = 300
SIMULATIONS = 5000
TOP = 30
SEED = 23016


def load():
    labels = {}
    with (ROOT / "datos/salidas/etiquetas_subconjunto.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            labels[row["Image Index"]] = set(row["Finding Labels"].split("|"))
    test = set((ROOT / "datos/salidas/particion_prueba.txt").read_text().split())
    with (ROOT / "resultados/gpu/cola.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    return labels, test, rows


def simulate(scores, masks):
    rng = np.random.default_rng(SEED)
    positions = {k: [] for k in masks}
    for _ in range(SIMULATIONS):
        idx = rng.choice(len(scores), QUEUE_SIZE, replace=False)
        order = idx[np.argsort(-scores[idx], kind="stable")]
        for k, mask in masks.items():
            positions[k].append(np.nonzero(mask[order])[0] + 1)
    out = {}
    for k, chunks in positions.items():
        p = np.concatenate(chunks)
        out[k] = {"casos": int(masks[k].sum()), "posicion_media": round(float(p.mean()), 1),
                  "mediana": float(np.median(p)), "p90": float(np.percentile(p, 90)),
                  "primeras_30_pct": round(100 * float((p <= TOP).mean()), 1)}
    return out


def rank_score(rows, test):
    train = [r for r in rows if r["imagen"] not in test]
    held = [r for r in rows if r["imagen"] in test]
    ranked = []
    for finding in FINDINGS:
        ref = np.sort([float(r[finding]) for r in train])
        ranked.append(np.searchsorted(ref, [float(r[finding]) for r in held]) / len(ref))
    return np.max(ranked, axis=0)


def main():
    labels, test, rows = load()
    held = [r for r in rows if r["imagen"] in test]
    names = [r["imagen"] for r in held]
    masks = {k: np.array([v in labels[n] for n in names]) for k, v in FINDINGS.items()}
    masks["cualquier_critico"] = np.array([bool(labels[n] & set(FINDINGS.values())) for n in names])
    scores = np.array([float(r["puntaje"]) for r in held])
    result = {
        "imagenes_prueba": len(names),
        "tamano_cola": QUEUE_SIZE,
        "simulaciones": SIMULATIONS,
        "posicion_media_por_llegada": (QUEUE_SIZE + 1) / 2,
        "prevalencia_critico_pct": round(100 * float(masks["cualquier_critico"].mean()), 1),
        "puntaje_maximo": simulate(scores, masks),
        "puntaje_por_percentil": simulate(rank_score(rows, test), masks),
    }
    out = ROOT / "resultados/impacto_cola.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
