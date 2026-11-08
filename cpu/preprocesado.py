"""preprocesado en cpu con numpy: recorte al centro, reduccion por area a 224 x 224
y normalizacion a [-1024, 1024]. es la misma operacion que hace el kernel de la gpu."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from comun.tarea import DESPLAZAMIENTO, ESCALA, TAMANO, intervalos_area, ventana_recorte


def preprocesar(img: np.ndarray, tamano: int = TAMANO) -> np.ndarray:
    """uint8 (alto, ancho) -> float32 (tamano, tamano) normalizado.

    cada pixel de salida es el promedio de su caja de pixeles de entrada
    (intervalos_area). las sumas de cada caja salen de sumas acumuladas en
    enteros, asi que son exactas. la normalizacion es lineal, por eso se aplica
    despues de promediar, sobre 224 x 224 y no sobre 1024 x 1024.
    """
    fila, col, lado = ventana_recorte(*img.shape)
    recorte = img[fila:fila + lado, col:col + lado]
    ini, fin = intervalos_area(lado, tamano)

    # suma por filas: acumulada con un cero al inicio, luego c[fin] - c[ini]
    acum = np.zeros((lado + 1, lado), dtype=np.int32)
    np.cumsum(recorte, axis=0, dtype=np.int32, out=acum[1:])
    filas = acum[fin] - acum[ini]  # (tamano, lado)

    # suma por columnas sobre el resultado anterior
    acum = np.zeros((tamano, lado + 1), dtype=np.int32)
    np.cumsum(filas, axis=1, out=acum[:, 1:])
    cajas = acum[:, fin] - acum[:, ini]  # (tamano, tamano)

    largo = (fin - ini).astype(np.float32)
    media = cajas.astype(np.float32) / np.outer(largo, largo)
    return media * np.float32(ESCALA) + np.float32(DESPLAZAMIENTO)


def preprocesar_lote(imgs: list[np.ndarray], pool: ThreadPoolExecutor | None = None,
                     tamano: int = TAMANO) -> np.ndarray:
    """lista de imagenes -> float32 (n, 1, tamano, tamano). con pool, una imagen por hilo."""
    salida = np.empty((len(imgs), 1, tamano, tamano), dtype=np.float32)

    def una(k: int) -> None:
        salida[k, 0] = preprocesar(imgs[k], tamano)

    if pool is None:
        for k in range(len(imgs)):
            una(k)
    else:
        list(pool.map(una, range(len(imgs))))
    return salida
