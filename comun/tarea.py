"""definicion de la tarea que comparten la version cpu y la gpu.

todo lo que tiene que ser identico en las dos versiones vive aqui:
la lista de imagenes, la lectura y decodificacion del png (en las dos se queda
en la cpu), los parametros del preprocesado, el modelo y el puntaje.
"""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
EXTRAIDO = RAIZ / "datos" / "extraido"
ETIQUETAS = RAIZ / "datos" / "salidas" / "etiquetas_subconjunto.csv"
COLUMNA_IMAGEN = "Image Index"

# modelo (informe 1, seccion 3.1)
PESOS = "densenet121-res224-all"
TAMANO = 224  # entrada de la red
MAXVAL = 255  # png de 8 bits
# [0, 255] -> [-1024, 1024], igual que xrv.utils.normalize(img, 255)
ESCALA = 2048.0 / MAXVAL
DESPLAZAMIENTO = -1024.0

# puntaje de urgencia: maximo entre estos 5 hallazgos (nombre en espanol -> salida del modelo)
HALLAZGOS = {
    "neumotorax": "Pneumothorax",
    "edema": "Edema",
    "neumonia": "Pneumonia",
    "derrame": "Effusion",
    "consolidacion": "Consolidation",
}


# ---------- lista de imagenes ----------

def listar_imagenes(extraido: Path = EXTRAIDO, etiquetas: Path | None = ETIQUETAS,
                    limite: int | None = None) -> list[Path]:
    """rutas de las imagenes del subconjunto, en orden fijo (por nombre).

    si existe el csv filtrado de diego se usa como lista oficial; si no, todos
    los png de la carpeta. asi cpu y gpu procesan exactamente lo mismo.
    """
    rutas = {p.name: p for p in extraido.rglob("*.png")}
    if not rutas:
        raise FileNotFoundError(f"no hay png en {extraido}")
    if etiquetas is not None and etiquetas.is_file():
        with etiquetas.open(newline="", encoding="utf-8-sig") as f:
            nombres = sorted(fila[COLUMNA_IMAGEN] for fila in csv.DictReader(f))
        faltan = [n for n in nombres if n not in rutas]
        if faltan:
            raise FileNotFoundError(f"{len(faltan)} imagenes del csv no estan en {extraido}, ej. {faltan[:3]}")
    else:
        nombres = sorted(rutas)
    if limite is not None:
        nombres = nombres[:limite]
    return [rutas[n] for n in nombres]


# ---------- lectura y decodificacion (cpu en las dos versiones) ----------

def leer_bytes(ruta: Path) -> np.ndarray:
    return np.frombuffer(ruta.read_bytes(), dtype=np.uint8)


def decodificar(datos: np.ndarray) -> np.ndarray:
    """png -> matriz uint8 en escala de grises (alto, ancho).

    opencv convierte a gris los png en color (0.299 r + 0.587 g + 0.114 b),
    descarta el canal alfa y baja a 8 bits los de 16. libera el gil, asi que
    se puede llamar desde varios hilos.
    """
    img = cv2.imdecode(datos, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("no se pudo decodificar el png")
    return img


# ---------- parametros del preprocesado ----------

def ventana_recorte(alto: int, ancho: int) -> tuple[int, int, int]:
    """recorte cuadrado al centro, igual que xrv.datasets.XRayCenterCrop.

    devuelve (fila_inicio, columna_inicio, lado).
    """
    lado = min(alto, ancho)
    return alto // 2 - lado // 2, ancho // 2 - lado // 2, lado


def intervalos_area(entrada: int, salida: int = TAMANO) -> tuple[np.ndarray, np.ndarray]:
    """para cada pixel de salida, el rango [inicio, fin) de pixeles de entrada que promedia.

    es la reduccion por area de pytorch (F.adaptive_avg_pool2d / interpolate
    mode='area'): inicio = floor(i * entrada / salida), fin = ceil((i + 1) * entrada / salida).
    el kernel de la gpu usa exactamente estos rangos, un hilo por pixel de salida.
    """
    i = np.arange(salida)
    inicio = (i * entrada) // salida
    fin = -((-(i + 1) * entrada) // salida)  # techo con enteros
    return inicio, fin


@lru_cache(maxsize=8)
def matriz_area(entrada: int, salida: int = TAMANO) -> np.ndarray:
    """matriz (salida x entrada) que promedia cada intervalo; la reduccion es separable:
    salida = M @ img @ M.T"""
    inicio, fin = intervalos_area(entrada, salida)
    m = np.zeros((salida, entrada), dtype=np.float32)
    for i, (a, b) in enumerate(zip(inicio, fin)):
        m[i, a:b] = 1.0 / (b - a)
    return m


# ---------- puntaje ----------

def indices_hallazgos(patologias: list[str]) -> list[int]:
    """posicion de cada hallazgo critico en la salida del modelo"""
    faltan = [v for v in HALLAZGOS.values() if v not in patologias]
    if faltan:
        raise ValueError(f"el modelo no tiene salida para: {faltan}")
    return [patologias.index(v) for v in HALLAZGOS.values()]
