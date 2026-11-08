# Versión de CPU

Pipeline completo en la CPU sobre el subconjunto images_001 a images_005 de NIH ChestX-ray14:

```
leer png (disco) -> decodificar (opencv, gris) -> recorte al centro -> 224 x 224 (promedio por área)
-> normalizar a [-1024, 1024] -> DenseNet-121 (TorchXRayVision, FP32) -> puntaje -> cola ordenada
```

- **Puntaje de urgencia:** máximo de las probabilidades de neumotórax, edema, neumonía, derrame y consolidación.
- **Pesos:** `densenet121-res224-all`. Sus 18 salidas están entrenadas; en `-nih`, las últimas 4 no.
- **Hilos:** por defecto, todos los núcleos lógicos (`os.cpu_count()`, 32 en el i9-13980HX). Se usan en PyTorch y en el pool que lee, decodifica y preprocesa. El número usado queda en `tiempos.json`.

## Lo que comparte con la versión de GPU (`comun/tarea.py`)

Para que las dos versiones resuelvan exactamente la misma tarea, en `comun/tarea.py` está:

| Qué | Valor |
|---|---|
| Lista de imágenes | La del CSV filtrado de Diego (`datos/salidas/etiquetas_subconjunto.csv`), ordenada por nombre |
| Lectura y decodificación | `leer_bytes` + `decodificar` (OpenCV, `IMREAD_GRAYSCALE`). En las dos versiones se queda en la CPU |
| Recorte | Cuadrado al centro, igual que `xrv.datasets.XRayCenterCrop` (`ventana_recorte`) |
| Reducción a 224 × 224 | Promedio por área, igual que `F.adaptive_avg_pool2d` de PyTorch. Para el pixel de salida `i`, se promedian los pixeles de entrada `[floor(i·N/224), ceil((i+1)·N/224))` en cada eje (`intervalos_area`) |
| Normalización | `x · 2048/255 − 1024`, igual que `xrv.utils.normalize(img, 255)` |
| Modelo y salidas | `PESOS` y `indices_hallazgos` |

El kernel de GPU (un hilo por pixel de salida) solo tiene que recorrer la caja de `intervalos_area` y aplicar la misma normalización. La referencia en PyTorch es `F.adaptive_avg_pool2d(x, 224) * 2048/255 - 1024`. La versión de CPU coincide con esa referencia con una diferencia menor a 0,001 (ver `cpu/tests`).

Respecto al preprocesado oficial de TorchXRayVision (skimage con suavizado), los puntajes cambian poco: en 64 radiografías reales, la diferencia media de probabilidad fue 0,014 y la correlación de Spearman entre puntajes fue 0,997. Se eligió el promedio por área porque se puede programar igual, de forma exacta, en CPU y en GPU.

## Cómo correrlo

Requisitos: Python 3.11 y los datos preparados con el script de Diego (`datos/extraido/` y `datos/salidas/`).

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv -r cpu/requirements.txt
```

Prueba rápida (500 imágenes, 1 corrida):

```bash
.venv/bin/python -m cpu.ejecutar --limite 500 -r 1 --calentamiento 0 --salida resultados/prueba
```

Medición final (44 999 imágenes, 1 corrida de calentamiento y la mediana de 5):

```bash
.venv/bin/python -m cpu.ejecutar --salida resultados/cpu
```

En el i9-13980HX la mejor configuración fue 8 hilos solo en los núcleos P (ver `resultados/RESUMEN.md`, barrido de hilos), así que la medición final se corrió así:

```bash
taskset -c 0-15 .venv/bin/python -m cpu.ejecutar --hilos 8 --salida resultados/cpu
```

Entre corridas se espera a que la laptop se enfríe (`--sin-pausas` lo desactiva). Los pesos desnormalizados se pasan a cero (`comun/model.py`); `--pesos-originales` lo desactiva.

| Parámetro | Default | Qué hace |
|---|---|---|
| `--lote` | 64 | Imágenes por lote. Así nunca se carga todo el subconjunto (unos 45 GB decodificados) en la RAM |
| `--hilos` | todos | Hilos de PyTorch y del pool |
| `-r`, `--repeticiones` | 5 | Corridas medidas; se reporta la mediana |
| `--calentamiento` | 1 | Corridas que no se miden |
| `--limite` | todas | Solo las primeras N imágenes |
| `--pesos` | `densenet121-res224-all` | Pesos de TorchXRayVision |

La primera vez, TorchXRayVision descarga los pesos (unos 30 MB) a `~/.torchxrayvision/`.

## Salidas (`--salida`)

- `cola.csv`: una fila por imagen, de la más urgente a la menos urgente. Columnas: `posicion, imagen, puntaje, hallazgo_principal` y la probabilidad de cada uno de los 5 hallazgos.
- `tiempos.json`: tiempo de cada etapa (lectura, decodificación, preprocesado, inferencia, puntaje y ordenamiento) en cada corrida, la mediana, la fracción de cada etapa, las imágenes por segundo, el equipo y las versiones de las librerías.

## Comparar con la GPU y medir precisión

```bash
.venv/bin/python -m comun.comparar resultados/cpu/cola.csv resultados/gpu/cola.csv
.venv/bin/python -m comun.evaluar resultados/cpu/cola.csv
```

`comparar` exige que las dos colas tengan las mismas imágenes y reporta la diferencia máxima de puntajes, cuántas quedan en la misma posición, la correlación de Spearman y cuántas coinciden entre las 100 más urgentes. `evaluar` calcula el AUC de cada hallazgo solo sobre la partición oficial de prueba (`test_list.txt`), porque los pesos se entrenaron con datos del NIH.

## Notas

- **Caché del disco:** después del calentamiento, el sistema operativo puede tener los PNG en la caché de memoria. Con 15 GB de RAM y 17 GB de PNG, solo una parte cabe. Afecta igual a CPU y a GPU, porque las dos leen igual.
- **Mac:** el PyTorch para macOS ARM no trae oneDNN, así que la inferencia da casi lo mismo con 4, 8 o 12 hilos (unas 26 img/s en el M4 Pro). En el i9 (x86, con oneDNN) sí debería escalar con los hilos. Los tiempos de la Mac no se usan en el informe.
- OpenCV puede imprimir `libpng warning: iCCP: ...` en algunas imágenes. Es un aviso del perfil de color y no afecta los píxeles.
- **Tests:** `.venv/bin/python -m pytest cpu/tests`.
