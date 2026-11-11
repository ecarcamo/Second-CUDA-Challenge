# Resumen de mediciones

Generado a partir de `resultados/`. 44999 imágenes; mediana de 5 corridas tras 1 de calentamiento en cada versión.

- Equipo: 13th Gen Intel(R) Core(TM) i9-13980HX, 15.2 GB de RAM, NVIDIA GeForce RTX 4050 Laptop GPU (5.72 GB), driver 615.78.08, CUDA 13.2, cuDNN 92600.
- Librerías: torch 2.15.0+cu132, torchxrayvision 1.5.7, CuPy 14.2.0, OpenCV 5.0.0, NumPy 2.4.6.
- CPU: 8 hilos, lote 64. GPU: lote 64, transferencia uint8, preprocesado kernel, memoria fijada, modo etapas, fp32 (tf32 apagado).

## Resultado principal

| Medida | Valor |
|---|---|
| T_CPU | 1 445,8 s (24,1 min), 31,1 img/s |
| T_GPU | 173,3 s (2,89 min), 259,7 img/s |
| Speedup total T_CPU / T_GPU | 8,3× |
| Speedup de la inferencia | 10,5× |
| Speedup del preprocesado (kernel CuPy) | 219,2× |
| Copia a la GPU | 3,96 s (2,3 % de T_GPU), 43,9 GiB a 11,9 GB/s |
| Copia de resultados a la CPU | 0,32 ms |
| VRAM pico (reservada) | 2 532 MiB |

## Tiempo por etapa (mediana, segundos)

| Etapa | CPU | % | GPU | % | Speedup |
|---|---|---|---|---|---|
| lectura | 19,111 | 1,3 | 18,419 | 10,6 | 1,0× |
| decodificacion | 53,479 | 3,7 | 27,605 | 15,9 | 1,9× |
| copia | — | — | 3,958 | 2,3 | — |
| preprocesado | 79,246 | 5,5 | 0,362 | 0,2 | 219,2× |
| inferencia | 1 293,656 | 89,5 | 122,821 | 70,9 | 10,5× |
| puntaje | 0,071 | 0,0 | 0,136 | 0,1 | 0,5× |
| ordenamiento | 0,021 | 0,0 | 0,000 | 0,0 | 212,7× |
| copia_resultados | — | — | 0,000 | 0,0 | — |
| total | 1 445,8 | 100 | 173,3 | 100 | 8,3× |

## Ley de Amdahl

- Lectura y decodificación se quedan en la CPU en las dos versiones: 72,6 s de T_CPU, fracción serial s = 5,02 %. Speedup máximo teórico 1/s = 19,9×.
- En la versión GPU esas mismas etapas tardan 46,0 s, el 26,6 % de T_GPU (en la CPU eran el 5,0 %): ahora son una parte importante del tiempo.
- Con la parte paralela acelerada 10,5× (la de la inferencia), Amdahl predice 1 / (s + (1 − s) / 10,5) = 7,1×; se midió 8,3×.
- La lectura y la decodificación usan 8 hilos en la versión CPU (su mejor configuración) y 32 en la GPU, porque ahí la CPU solo decodifica. Por eso esa parte también cambia (1,6×).

## Con 10 veces más datos

El trabajo crece lineal con las imágenes y se procesa por lotes, así que la RAM y la VRAM no cambian.

| Imágenes | T_CPU | T_GPU |
|---|---|---|
| 44 999 (subconjunto) | 0,40 h | 2,9 min |
| 112 120 (NIH completo) | 1,00 h | 7,2 min |
| 449 990 (10 × subconjunto) | 4,02 h | 28,9 min |

## Para el IGSS

| Cola | CPU | GPU |
|---|---|---|
| 300 radiografías | 9,6 s | 1,2 s |
| 1000 radiografías | 32,1 s | 3,9 s |
| Una radiografía nueva (lote 1, medido) | 45,7 ms | 13,3 ms |

## Barrido de hilos en la CPU (2 000 imágenes, 1 corrida)

| Configuración | Hilos | img/s |
|---|---|---|
| nucleos_p_16 | 16 | 29,1 |
| nucleos_p_8 | 8 | 31,1 |
| todos_16 | 16 | 20,7 |
| todos_24 | 24 | 21,0 |
| todos_32 | 32 | 14,9 |
| todos_8 | 8 | 28,9 |

Mejor: `nucleos_p_8`. `nucleos_p` = `taskset -c 0-15` (solo núcleos P).

## Pesos desnormalizados

De los pesos `densenet121-res224-all`, 2 564 659 valores son desnormalizados (menores que 1,18e-38). En x86 cada operación con ellos es muy lenta. Se pasan a cero en las dos versiones; la salida del modelo no cambia.

| Versión (2 000 imágenes) | Pesos originales | Pesos limpios |
|---|---|---|
| CPU | 14,6 img/s | 31,1 img/s |
| GPU | 280,6 img/s | 280,0 img/s |

## Variantes de la GPU

| Variante | img/s | T total (s) | Copia (s) | Preprocesado (s) | Decodificación (s) | VRAM pico (MiB) |
|---|---|---|---|---|---|---|
| base (gpu) | 259,7 | 173,3 | 3,96 | 0,362 | 27,6 | 2 532 |
| float32 (44999 img) | 238,7 | 188,5 | 15,33 | 1,160 | 31,5 | 2 724 |
| hilos_cpu (44999 img) | 226,8 | 198,4 | 3,87 | 0,361 | 53,6 | 2 532 |
| lote_128 (44999 img) | 265,2 | 169,7 | 3,92 | 0,340 | 22,3 | 3 984 |
| lote_32 (44999 img) | 263,4 | 170,8 | 4,15 | 0,389 | 29,9 | 1 820 |
| paginable (44999 img) | 258,6 | 174,0 | 6,75 | 0,541 | 25,0 | 2 532 |
| pesos_limpios_2000 (2000 img) | 280,0 | 7,1 | 0,17 | 0,016 | 0,8 | 2 530 |
| pesos_originales (2000 img) | 280,6 | 7,1 | 0,17 | 0,016 | 0,7 | 2 530 |
| solapado (44999 img) | 322,0 | 139,8 | 5,28 | 0,448 | 50,7 | 2 596 |
| torch (44999 img) | 255,8 | 175,9 | 3,94 | 2,652 | 27,4 | 2 420 |

## Misma tarea: CPU frente a GPU

```
imagenes: 44999
dif_max_puntaje: 2.000000000002e-06
dif_media_puntaje: 8.642414275876313e-08
dif_max_por_hallazgo: {'neumotorax': 2.000000000002e-06, 'edema': 2.000000000002e-06, 'neumonia': 2.000000000002e-06, 'derrame': 1.0000000000287557e-06, 'consolidacion': 1.0000000000287557e-06}
misma_posicion: 0.9927331718482633
dif_max_posicion: 2
spearman_posiciones: 0.9999999999774798
coinciden_top_100: 100
```

## AUC sobre la partición de prueba

CPU:
```
7724 imagenes de la particion de prueba
  auc neumotorax    0.630  (758 positivos)
  auc edema         0.774  (264 positivos)
  auc neumonia      0.614  (166 positivos)
  auc derrame       0.761  (1185 positivos)
  auc consolidacion 0.676  (529 positivos)
  auc puntaje vs 'algun hallazgo critico': 0.662  (2471 positivos)
```
GPU:
```
7724 imagenes de la particion de prueba
  auc neumotorax    0.630  (758 positivos)
  auc edema         0.774  (264 positivos)
  auc neumonia      0.614  (166 positivos)
  auc derrame       0.761  (1185 positivos)
  auc consolidacion 0.676  (529 positivos)
  auc puntaje vs 'algun hallazgo critico': 0.662  (2471 positivos)
```

## Temperaturas

- 1547 lecturas cada 15 s. CPU máx. 100 °C, GPU máx. 75 °C.
- Lecturas con el proceso pausado por el vigilante: 0. Pausas entre corridas: 5 en CPU y 5 en GPU.
