import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "resultados"
OUTPUT = ROOT / "informe_final" / "figuras"
INK = "#1A1A1A"
MUTED = "#5A5A5A"
GRID = "#D9D9D9"
SERIES = (("Lectura y decodificación (CPU)", "#6B7F9E"), ("Copia a la GPU", "#E0862A"),
          ("Preprocesado", "#7CC48F"), ("Inferencia", "#1F7A4D"))
FINDING_NAMES = {"edema": "Edema", "consolidacion": "Consolidación", "derrame": "Derrame",
                 "cualquier_critico": "Cualquier crítico", "neumonia": "Neumonía", "neumotorax": "Neumotórax"}


def setup():
    plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                         "pdf.fonttype": 42})


def number(value, digits=1):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def stage_shares(data):
    median = data["mediana_s"]
    parts = (median["lectura"] + median["decodificacion"], median.get("copia", 0.0), median["preprocesado"],
             median["inferencia"])
    return [100 * p / median["total"] for p in parts], median["total"]


def stages_figure():
    cpu = json.loads((RESULTS / "cpu/tiempos.json").read_text(encoding="utf-8"))
    gpu = json.loads((RESULTS / "gpu/tiempos.json").read_text(encoding="utf-8"))
    rows = [("GPU", *stage_shares(gpu)), ("CPU", *stage_shares(cpu))]
    fig, ax = plt.subplots(figsize=(6.9, 1.55))
    for y, (name, shares, total) in enumerate(rows):
        left = 0.0
        for share, (_, color) in zip(shares, SERIES):
            if share > 0:
                ax.barh(y, share, left=left, height=0.56, color=color, edgecolor="white", linewidth=1.2)
            if share >= 10:
                ax.text(left + share / 2, y, f"{number(share)} %", ha="center", va="center", fontsize=8.5,
                        color="white")
            left += share
    ax.set_yticks([0, 1], [f"{name}, {number(total)} s" for name, _, total in rows])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Porcentaje del tiempo total")
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in SERIES]
    ax.legend(handles, [name for name, _ in SERIES], ncol=4, frameon=False, loc="lower center",
              bbox_to_anchor=(0.42, 1.0), fontsize=8, handlelength=1, columnspacing=1.2)
    fig.savefig(OUTPUT / "etapas.pdf", bbox_inches="tight")
    plt.close(fig)


def queue_figure():
    impact = json.loads((RESULTS / "impacto_cola.json").read_text(encoding="utf-8"))
    arrival = impact["posicion_media_por_llegada"]
    items = sorted(((FINDING_NAMES[k], v["posicion_media"]) for k, v in impact["puntaje_maximo"].items()),
                   key=lambda item: -item[1])
    fig, ax = plt.subplots(figsize=(6.9, 1.95))
    for y, (name, position) in enumerate(items):
        ax.plot([position, arrival], [y, y], color=GRID, linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter(position, y, s=46, color="#1F7A4D", zorder=2)
        ax.text(position - 3, y, number(position), ha="right", va="center", fontsize=8.5, color=INK)
    ax.axvline(arrival, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(arrival + 1.5, len(items) - 0.55, f"Orden de llegada: {number(arrival)}", fontsize=8.5, color=MUTED,
            va="center")
    ax.set_yticks(range(len(items)), [name for name, _ in items])
    ax.set_xlim(60, 185)
    ax.set_ylim(-0.6, len(items) - 0.2)
    ax.set_xlabel("Posición media en una cola de 300 placas (menor es mejor)")
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.savefig(OUTPUT / "cola.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    setup()
    stages_figure()
    queue_figure()
    return 0


if __name__ == "__main__":
    sys.exit(main())
