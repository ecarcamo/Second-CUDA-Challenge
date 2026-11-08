"""datos falsos para los tests. ningun nombre de aqui es del dataset real"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from PIL import Image

from datos.config import HALLAZGOS_CRITICOS

ETIQUETAS_FALSAS = {clave: f"falso_{clave}" for clave in HALLAZGOS_CRITICOS}


def config_base() -> dict:
    return {
        "archivos": [f"falso_{i:03d}.tar.gz" for i in range(1, 6)],
        "urls": {},
        "checksum_algoritmo": None,
        "checksums": {},
        "columna_imagen": "imagen",
        "formato_hallazgos": "columna_unica",
        "columna_hallazgos": "hallazgos",
        "separador_hallazgos": "|",
        "valor_positivo": "",
        "hallazgos": dict(ETIQUETAS_FALSAS),
    }


def escribir_config(carpeta: Path, **cambios) -> Path:
    ruta = carpeta / "dataset.json"
    ruta.write_text(json.dumps({**config_base(), **cambios}), encoding="utf-8")
    return ruta


def png_bytes(modo: str = "L", tamano: tuple[int, int] = (1024, 1024)) -> bytes:
    buffer = io.BytesIO()
    Image.new(modo, tamano).save(buffer, format="PNG")
    return buffer.getvalue()


def crear_png(ruta: Path, modo: str = "L", tamano: tuple[int, int] = (1024, 1024)) -> Path:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_bytes(png_bytes(modo, tamano))
    return ruta


def crear_tar(ruta: Path, miembros: dict[str, bytes]) -> Path:
    with tarfile.open(ruta, "w:gz") as tar:
        for nombre, contenido in miembros.items():
            info = tarfile.TarInfo(nombre)
            info.size = len(contenido)
            tar.addfile(info, io.BytesIO(contenido))
    return ruta


def escribir_csv(ruta: Path, filas: list[tuple[str, str]]) -> Path:
    lineas = ["imagen,hallazgos,edad"] + [f"{img},{hall},40" for img, hall in filas]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    return ruta


# csv donde aparecen las 5 etiquetas falsas al menos una vez
FILAS_CSV = [
    ("a.png", "falso_edema|falso_derrame"),
    ("b.png", "sin_hallazgo"),
    ("c.png", "falso_neumotorax"),
    ("d.png", "falso_neumonia|falso_consolidacion"),
]
