# Caso 1: Radiografías que no esperan

Caso práctico «CUDA en el mundo real» (CC3069, Universidad del Valle de Guatemala, grupo 1). Ordena una cola de radiografías de tórax por urgencia y compara la misma tarea en CPU y en GPU sobre 44 999 imágenes reales de NIH ChestX-ray14.

| Medida | Valor |
|---|---|
| T_CPU | 1 445,8 s (24,1 min) |
| T_GPU | 173,3 s (2,9 min) |
| Speedup | 8,3× (10,3× con decodificación solapada) |
| Copia a la GPU | 3,96 s, 2,3 % de T_GPU |
| GPU | NVIDIA GeForce RTX 4050 Laptop, 6 GB |

El detalle completo está en [resultados/RESUMEN.md](resultados/RESUMEN.md).

## Requisitos

- Linux con GPU NVIDIA y driver compatible con CUDA 13.
- Python 3.11 y [uv](https://docs.astral.sh/uv/).
- Unos 40 GB libres en disco.

## Cómo correrlo

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv -r gpu/requirements.txt
scripts/download.sh
scripts/run_all.sh
.venv/bin/python scripts/summarize.py
```

`scripts/run_all.sh` prepara los datos, corre los tests y hace todas las mediciones. En la laptop de referencia tarda unas 6 horas. Para una prueba rápida, una vez preparados los datos:

```bash
.venv/bin/python -m gpu.run --limite 500 -r 1 --sin-pausas --salida resultados/prueba_gpu
```

## Análisis adicionales

```bash
.venv/bin/python -m gpu.run --limite 1000 --lote 1 -r 5 --sin-pausas --salida resultados/latencia/gpu
taskset -c 0-15 .venv/bin/python -m cpu.ejecutar --limite 1000 --lote 1 --hilos 8 -r 5 --sin-pausas --salida resultados/latencia/cpu
.venv/bin/python scripts/queue_impact.py
uv run --no-project --with matplotlib python scripts/make_figures.py
.venv/bin/python scripts/build_presentation.py
```

## Estructura

| Carpeta | Contenido |
|---|---|
| `datos/` | Descarga, verificación y preparación del subconjunto |
| `comun/` | Tarea común a las dos versiones, modelo, comparación y evaluación |
| `cpu/` | Versión de CPU ([README](cpu/README.md)) |
| `gpu/` | Versión de GPU y kernel propio en CuPy ([README](gpu/README.md)) |
| `scripts/` | Descarga, corrida completa, resumen, impacto, figuras y presentación |
| `resultados/` | Mediciones usadas en el informe |
| `informe1/`, `informe_final/` | Informes en LaTeX |
| `presentacion/` | Presentación interactiva |
| `entrega/` | Informe final y diapositivas en PDF |

Para compilar el informe, dentro de `informe_final/`:

```bash
tectonic informe_final.tex
```

## Fuentes

- Datos: NIH Clinical Center, [ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC). Wang et al. (2017), CVPR. Uso sin restricciones, con enlace, cita y reconocimiento al NIH Clinical Center.
- Modelo: [TorchXRayVision](https://github.com/mlmed/torchxrayvision), Apache 2.0. Cohen et al. (2022).
- Volúmenes del IGSS: [Boletín estadístico de prestaciones en salud 2024](https://www.igssgt.org/informes-y-estadisticas/boletin-estadistico-de-salud/).
