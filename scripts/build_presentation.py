import base64
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados"
IMAGES = ROOT / "datos" / "extraido"
TEMPLATE = ROOT / "presentacion" / "template.html"
OUTPUT = ROOT / "presentacion" / "index.html"
QUEUE_SIZE = 300
COLUMNS = 25
TILE = 64
SEED = 23016
HERO_ARRIVAL = 214
FINDINGS = {"Pneumothorax", "Edema", "Pneumonia", "Effusion", "Consolidation"}


def data_uri(mime, payload):
    return f"data:{mime};base64,{base64.b64encode(payload).decode()}"


def find_image(name):
    return next(IMAGES.rglob(name))


def load_rows():
    labels = {}
    with (ROOT / "datos/salidas/etiquetas_subconjunto.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            labels[row["Image Index"]] = row
    test = set((ROOT / "datos/salidas/particion_prueba.txt").read_text().split())
    with (RESULTS / "gpu/cola.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["imagen"] in test]
    return labels, rows


def build_queue(labels, rows):
    rng = np.random.default_rng(SEED)
    picks = [rows[i] for i in rng.choice(len(rows), QUEUE_SIZE, replace=False)]
    picks.sort(key=lambda r: -float(r["puntaje"]))
    for rank, row in enumerate(picks, 1):
        row["rank"] = rank
    arrival = rng.permutation(QUEUE_SIZE) + 1
    for row, position in zip(picks, arrival):
        row["arrival"] = int(position)
    impact = json.loads((RESULTS / "impacto_cola.json").read_text(encoding="utf-8"))
    target = impact["puntaje_maximo"]["edema"]["mediana"]
    edema = [r for r in picks if "Edema" in labels[r["imagen"]]["Finding Labels"]]
    hero = min(edema, key=lambda r: (abs(r["rank"] - target), labels[r["imagen"]]["View Position"] != "PA"))
    other = next(r for r in picks if r["arrival"] == HERO_ARRIVAL)
    other["arrival"], hero["arrival"] = hero["arrival"], HERO_ARRIVAL
    return picks, hero, impact


def build_sprite(picks):
    rows = (len(picks) + COLUMNS - 1) // COLUMNS
    sheet = np.zeros((rows * TILE, COLUMNS * TILE), dtype=np.uint8)
    for i, row in enumerate(picks):
        image = cv2.imread(str(find_image(row["imagen"])), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(image, (TILE, TILE), interpolation=cv2.INTER_AREA)
        y, x = divmod(i, COLUMNS)
        sheet[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE] = small
    ok, encoded = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 72])
    assert ok
    return data_uri("image/jpeg", encoded.tobytes())


def timings():
    out = {}
    for key, path in (("cpu", "cpu"), ("gpu", "gpu"), ("overlap", "variantes_gpu/solapado"),
                      ("float32", "variantes_gpu/float32"), ("same_threads", "variantes_gpu/hilos_cpu"),
                      ("latency_cpu", "latencia/cpu"), ("latency_gpu", "latencia/gpu")):
        data = json.loads((RESULTS / path / "tiempos.json").read_text(encoding="utf-8"))
        out[key] = {"median": data["mediana_s"], "images": data["imagenes"]}
    return out


def main():
    labels, rows = load_rows()
    picks, hero, impact = build_queue(labels, rows)
    meta = labels[hero["imagen"]]
    image = cv2.imread(str(find_image(hero["imagen"])), cv2.IMREAD_GRAYSCALE)
    ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    assert ok
    payload = {
        "sprite": build_sprite(picks),
        "columns": COLUMNS,
        "tile": TILE,
        "items": [{"a": r["arrival"], "r": r["rank"],
                   "c": bool(set(labels[r["imagen"]]["Finding Labels"].split("|")) & FINDINGS)} for r in picks],
        "heroIndex": picks.index(hero),
        "hero": {"file": hero["imagen"], "arrival": hero["arrival"], "rank": hero["rank"],
                 "score": float(hero["puntaje"]), "age": int(meta["Patient Age"]), "sex": meta["Patient Gender"],
                 "labels": meta["Finding Labels"], "view": meta["View Position"],
                 "image": data_uri("image/png", encoded.tobytes())},
        "impact": impact,
        "timings": timings(),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    OUTPUT.write_text(html, encoding="utf-8")
    summary = {k: payload["hero"][k] for k in ("file", "arrival", "rank", "score", "age", "sex", "labels", "view")}
    print(json.dumps(summary, ensure_ascii=False))
    print("bytes", OUTPUT.stat().st_size)
    print("critical in queue", sum(i["c"] for i in payload["items"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
