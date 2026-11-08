# Versión de GPU

La misma tarea que la versión de CPU, sobre las mismas imágenes y con el mismo modelo:

```
leer png (disco, CPU) -> decodificar (OpenCV, CPU, hilos) -> copia a la GPU en uint8 (PCIe)
-> recorte, 224 x 224 y normalización (kernel propio en CuPy) -> DenseNet-121 (cuDNN, FP32)
-> puntaje (GPU) -> cola ordenada (GPU) -> copia de la cola a la CPU
```

- **Lectura y decodificación:** se usan las mismas funciones de `comun/tarea.py` que en la CPU. Cada hilo decodifica una imagen y la escribe directo en un búfer de memoria fijada (pinned) del lote.
- **Kernel propio (`gpu/preprocess.py`):** un hilo por pixel de salida (224 × 224 por imagen, una imagen por bloque en el eje z). Cada hilo suma su caja de pixeles de entrada (`intervalos_area`), divide entre el tamaño de la caja y aplica `x · 2048/255 − 1024`. Hace las mismas operaciones que `cpu/preprocesado.py`; la diferencia con la CPU es menor a 0,001.
- **Referencia en PyTorch:** `--preprocesado torch` hace la misma operación con `F.adaptive_avg_pool2d`, para comparar.
- **FP32 de verdad:** PyTorch usa TF32 en las convoluciones por defecto. Aquí se apaga (`fp32_precision = "ieee"`) para que la GPU calcule con la misma precisión que la CPU.
- **Pesos desnormalizados:** `comun/model.py` pasa a cero los valores de los pesos menores que 1,18e-38 (unos 2,56 millones de los 6,97 millones). La salida del modelo no cambia, pero en x86 la CPU es unas 4 veces más rápida. Se hace en las dos versiones; `--pesos-originales` lo desactiva.
- **Medición:** la lectura y la decodificación se miden con el reloj de la CPU; la copia, el preprocesado, la inferencia, el puntaje, el ordenamiento y la copia de resultados, con eventos de CUDA. Se reportan los bytes copiados, el ancho de banda de la copia y la VRAM pico.

## Cómo correrlo

Mismo entorno que la CPU, más CuPy:

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv -r gpu/requirements.txt
```

Prueba rápida (500 imágenes, 1 corrida):

```bash
.venv/bin/python -m gpu.run --limite 500 -r 1 --sin-pausas --salida resultados/prueba_gpu
```

Medición final (44 999 imágenes, 1 corrida de calentamiento y la mediana de 5):

```bash
.venv/bin/python -m gpu.run --salida resultados/gpu
```

| Parámetro | Default | Qué hace |
|---|---|---|
| `--lote` | 64 | Imágenes por lote |
| `--hilos` | todos | Hilos de la CPU para leer y decodificar |
| `--transferencia` | `uint8` | `float32` convierte en la CPU y copia 4 veces más bytes |
| `--preprocesado` | `kernel` | `torch` usa `F.adaptive_avg_pool2d` en vez del kernel propio |
| `--memoria` | `fijada` | `paginable` copia desde memoria normal |
| `--modo` | `etapas` | `etapas` sincroniza en cada lote para medir cada etapa; `solapado` decodifica el lote siguiente mientras la GPU procesa el actual |
| `-r`, `--calentamiento`, `--limite`, `--pesos` | | Igual que en la CPU |
| `--sin-pausas` | | No espera a que la laptop se enfríe entre corridas |
| `--pesos-originales` | | No pasa a cero los pesos desnormalizados |

Las salidas (`cola.csv` y `tiempos.json`) tienen el mismo formato que las de la CPU, con dos etapas más (`copia` y `copia_resultados`).

## Todo de una vez

`scripts/run_all.sh` espera a que estén los datos, corre `datos.preparar_subconjunto`, los tests, el barrido de hilos de la CPU, la medición final de CPU y GPU, las variantes de la GPU, `comun.comparar` y `comun.evaluar`. Deja los registros en `resultados/logs/`. Mientras corre, `scripts/thermal_guard.py` anota la temperatura cada 15 s y pausa las mediciones si la CPU pasa de 99 °C o la GPU de 87 °C durante un minuto. Al final, `scripts/summarize.py` escribe `resultados/RESUMEN.md`.

```bash
scripts/download.sh
scripts/run_all.sh
.venv/bin/python scripts/summarize.py
```

## Tests

```bash
.venv/bin/python -m pytest gpu/tests
```

Comparan el kernel con la CPU y con PyTorch (imágenes cuadradas y no cuadradas, uint8 y float32) y corren las dos versiones completas sobre imágenes pequeñas para comprobar que dan la misma cola con cada variante.
