from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "resultados"
CPU_STAGES = ("lectura", "decodificacion", "preprocesado", "inferencia", "puntaje", "ordenamiento")
GPU_STAGES = ("lectura", "decodificacion", "copia", "preprocesado", "inferencia", "puntaje", "ordenamiento",
              "copia_resultados")
SHIFT_SIZES = (300, 1000)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def fmt(x: float, d: int = 2) -> str:
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "(no existe)"


def main() -> int:
    cpu, gpu = load(RES / "cpu" / "tiempos.json"), load(RES / "gpu" / "tiempos.json")
    if not cpu or not gpu:
        print("faltan resultados/cpu o resultados/gpu", file=sys.stderr)
        return 1
    out = []
    w = out.append
    tc, tg = cpu["mediana_s"]["total"], gpu["mediana_s"]["total"]
    n = cpu["imagenes"]

    w("# Resumen de mediciones\n")
    w(f"Generado a partir de `resultados/`. {n} imágenes; mediana de {cpu['repeticiones']} corridas tras "
      f"{cpu['calentamiento']} de calentamiento en cada versión.\n")
    eq = gpu["equipo"]
    w(f"- Equipo: {eq['cpu']}, {eq['ram_gb']} GB de RAM, {eq['gpu']} ({eq['vram_gb']} GB), driver {eq['driver']}, "
      f"CUDA {eq['cuda_torch']}, cuDNN {eq['cudnn']}.")
    w(f"- Librerías: torch {gpu['librerias']['torch']}, torchxrayvision {gpu['librerias']['torchxrayvision']}, "
      f"CuPy {gpu['librerias']['cupy']}, OpenCV {gpu['librerias']['opencv']}, NumPy {gpu['librerias']['numpy']}.")
    w(f"- CPU: {cpu['hilos']} hilos, lote {cpu['lote']}. GPU: lote {gpu['lote']}, transferencia {gpu['transferencia']}, "
      f"preprocesado {gpu['preprocesado']}, memoria {gpu['memoria_host']}, modo {gpu['modo']}, {gpu['precision']}.\n")

    w("## Resultado principal\n")
    w("| Medida | Valor |\n|---|---|")
    w(f"| T_CPU | {fmt(tc, 1)} s ({fmt(tc / 60, 1)} min), {fmt(n / tc, 1)} img/s |")
    w(f"| T_GPU | {fmt(tg, 1)} s ({fmt(tg / 60, 2)} min), {fmt(n / tg, 1)} img/s |")
    w(f"| Speedup total T_CPU / T_GPU | {fmt(tc / tg, 1)}× |")
    ci, gi = cpu["mediana_s"]["inferencia"], gpu["mediana_s"]["inferencia"]
    w(f"| Speedup de la inferencia | {fmt(ci / gi, 1)}× |")
    cp_, gp_ = cpu["mediana_s"]["preprocesado"], gpu["mediana_s"]["preprocesado"]
    w(f"| Speedup del preprocesado (kernel CuPy) | {fmt(cp_ / gp_, 1)}× |")
    copy = gpu["mediana_s"]["copia"]
    w(f"| Copia a la GPU | {fmt(copy, 2)} s ({fmt(100 * copy / tg, 1)} % de T_GPU), "
      f"{fmt(gpu['bytes_copiados_a_gpu'] / 2**30, 1)} GiB a {fmt(gpu['ancho_de_banda_copia_gb_s'], 1)} GB/s |")
    w(f"| Copia de resultados a la CPU | {fmt(1000 * gpu['mediana_s']['copia_resultados'], 2)} ms |")
    w(f"| VRAM pico (reservada) | {fmt(gpu['vram_pico_reservada_mib'], 0)} MiB |\n")

    w("## Tiempo por etapa (mediana, segundos)\n")
    w("| Etapa | CPU | % | GPU | % | Speedup |\n|---|---|---|---|---|---|")
    for s in GPU_STAGES:
        c = cpu["mediana_s"].get(s)
        g = gpu["mediana_s"][s]
        cs = f"{fmt(c, 3)} | {fmt(100 * c / tc, 1)}" if c is not None else "— | —"
        sp = fmt(c / g, 1) + "×" if c and g else "—"
        w(f"| {s} | {cs} | {fmt(g, 3)} | {fmt(100 * g / tg, 1)} | {sp} |")
    w(f"| total | {fmt(tc, 1)} | 100 | {fmt(tg, 1)} | 100 | {fmt(tc / tg, 1)}× |\n")

    host_cpu = cpu["mediana_s"]["lectura"] + cpu["mediana_s"]["decodificacion"]
    host_gpu = gpu["mediana_s"]["lectura"] + gpu["mediana_s"]["decodificacion"]
    s = host_cpu / tc
    w("## Ley de Amdahl\n")
    w(f"- Lectura y decodificación se quedan en la CPU en las dos versiones: {fmt(host_cpu, 1)} s de T_CPU, "
      f"fracción serial s = {fmt(100 * s, 2)} %. Speedup máximo teórico 1/s = {fmt(1 / s, 1)}×.")
    w(f"- En la versión GPU esas mismas etapas tardan {fmt(host_gpu, 1)} s, el {fmt(100 * host_gpu / tg, 1)} % de T_GPU "
      f"(en la CPU eran el {fmt(100 * s, 1)} %): ahora son una parte importante del tiempo.")
    p = 1 - s
    w(f"- Con la parte paralela acelerada {fmt(ci / gi, 1)}× (la de la inferencia), Amdahl predice "
      f"1 / (s + (1 − s) / {fmt(ci / gi, 1)}) = {fmt(1 / (s + p / (ci / gi)), 1)}×; se midió {fmt(tc / tg, 1)}×.")
    w(f"- La lectura y la decodificación usan {cpu['hilos']} hilos en la versión CPU (su mejor configuración) y "
      f"{gpu['hilos']} en la GPU, porque ahí la CPU solo decodifica. Por eso esa parte también cambia "
      f"({fmt(host_cpu / host_gpu, 1)}×).\n")

    w("## Con 10 veces más datos\n")
    w("El trabajo crece lineal con las imágenes y se procesa por lotes, así que la RAM y la VRAM no cambian.\n")
    w("| Imágenes | T_CPU | T_GPU |\n|---|---|---|")
    for m, label in ((n, "subconjunto"), (112_120, "NIH completo"), (10 * n, "10 × subconjunto")):
        w(f"| {m:,} ({label}) | ".replace(",", " ") + f"{fmt(tc * m / n / 3600, 2)} h | {fmt(tg * m / n / 60, 1)} min |")
    w("")

    w("## Para el IGSS\n")
    w("| Cola | CPU | GPU |\n|---|---|---|")
    for m in SHIFT_SIZES:
        w(f"| {m} radiografías | {fmt(tc * m / n, 1)} s | {fmt(tg * m / n, 1)} s |")
    latency_cpu, latency_gpu = load(RES / "latencia" / "cpu" / "tiempos.json"), load(RES / "latencia" / "gpu" / "tiempos.json")
    if latency_cpu and latency_gpu:
        one_cpu = 1000 * latency_cpu["mediana_s"]["total"] / latency_cpu["imagenes"]
        one_gpu = 1000 * latency_gpu["mediana_s"]["total"] / latency_gpu["imagenes"]
        w(f"| Una radiografía nueva (lote 1, medido) | {fmt(one_cpu, 1)} ms | {fmt(one_gpu, 1)} ms |")
    w("")

    sweep = sorted(p for p in RES.glob("barrido_cpu/*/tiempos.json") if p.parent.name != "pesos_originales")
    if sweep:
        w("## Barrido de hilos en la CPU (2 000 imágenes, 1 corrida)\n")
        w("| Configuración | Hilos | img/s |\n|---|---|---|")
        for path in sweep:
            d = load(path)
            w(f"| {path.parent.name} | {d['hilos']} | {fmt(d['imagenes_por_s'], 1)} |")
        w(f"\nMejor: `{text(RES / 'barrido_cpu' / 'mejor.txt')}`. `nucleos_p` = `taskset -c 0-15` (solo núcleos P).\n")

    orig_c = load(RES / "barrido_cpu" / "pesos_originales" / "tiempos.json")
    orig_g = load(RES / "variantes_gpu" / "pesos_originales" / "tiempos.json")
    clean_g = load(RES / "variantes_gpu" / "pesos_limpios_2000" / "tiempos.json")
    best = load(RES / "barrido_cpu" / text(RES / "barrido_cpu" / "mejor.txt") / "tiempos.json")
    if orig_c and best:
        w("## Pesos desnormalizados\n")
        count = f"{cpu.get('pesos_desnormalizados_a_cero', 0):,}".replace(",", " ")
        w(f"De los pesos `{cpu['pesos']}`, {count} valores son desnormalizados "
          f"(menores que 1,18e-38). En x86 cada operación con ellos es muy lenta. Se pasan a cero en las dos versiones; "
          f"la salida del modelo no cambia.\n")
        w("| Versión (2 000 imágenes) | Pesos originales | Pesos limpios |\n|---|---|---|")
        w(f"| CPU | {fmt(orig_c['imagenes_por_s'], 1)} img/s | {fmt(best['imagenes_por_s'], 1)} img/s |")
        if orig_g and clean_g:
            w(f"| GPU | {fmt(orig_g['imagenes_por_s'], 1)} img/s | {fmt(clean_g['imagenes_por_s'], 1)} img/s |")
        w("")

    variants = sorted(RES.glob("variantes_gpu/*/tiempos.json"))
    if variants:
        w("## Variantes de la GPU\n")
        w("| Variante | img/s | T total (s) | Copia (s) | Preprocesado (s) | Decodificación (s) | VRAM pico (MiB) |")
        w("|---|---|---|---|---|---|---|")
        w(f"| base (gpu) | {fmt(gpu['imagenes_por_s'], 1)} | {fmt(tg, 1)} | {fmt(copy, 2)} | {fmt(gp_, 3)} | "
          f"{fmt(gpu['mediana_s']['decodificacion'], 1)} | {fmt(gpu['vram_pico_reservada_mib'], 0)} |")
        for path in variants:
            d = load(path)
            m = d["mediana_s"]
            w(f"| {path.parent.name} ({d['imagenes']} img) | {fmt(d['imagenes_por_s'], 1)} | {fmt(m['total'], 1)} | "
              f"{fmt(m['copia'], 2)} | {fmt(m['preprocesado'], 3)} | {fmt(m['decodificacion'], 1)} | "
              f"{fmt(d['vram_pico_reservada_mib'], 0)} |")
        w("")

    w("## Misma tarea: CPU frente a GPU\n")
    w("```\n" + text(RES / "comparacion_cpu_gpu.txt") + "\n```\n")
    w("## AUC sobre la partición de prueba\n")
    w("CPU:\n```\n" + text(RES / "auc_cpu.txt") + "\n```\nGPU:\n```\n" + text(RES / "auc_gpu.txt") + "\n```\n")

    temps = RES / "logs" / "temperaturas.csv"
    if temps.is_file():
        rows = list(csv.DictReader(temps.open(encoding="utf-8")))
        cpu_t = [float(r["cpu_c"]) for r in rows if r["cpu_c"] not in ("", "None")]
        gpu_t = [float(r["gpu_c"]) for r in rows if r["gpu_c"] not in ("", "None")]
        paused = sum(int(r["pausado"]) for r in rows)
        w("## Temperaturas\n")
        w(f"- {len(rows)} lecturas cada 15 s. CPU máx. {fmt(max(cpu_t), 0)} °C, GPU máx. {fmt(max(gpu_t), 0)} °C.")
        w(f"- Lecturas con el proceso pausado por el vigilante: {paused}. Pausas entre corridas: "
          f"{len(cpu.get('pausas_termicas', []))} en CPU y {len(gpu.get('pausas_termicas', []))} en GPU.\n")

    (RES / "RESUMEN.md").write_text("\n".join(out), encoding="utf-8")
    print(RES / "RESUMEN.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
