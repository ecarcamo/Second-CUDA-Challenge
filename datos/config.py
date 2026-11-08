"""carga y valida datos/dataset.json.

ese json lo llena el equipo con los datos reales del nih (fase 0 del plan).
aqui no se inventa ningun valor del dataset: si falta algo, se corta con error.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePath
from urllib.parse import urlparse

from datos.errores import ErrorDatos

# los 5 hallazgos criticos del informe 1, seccion 3.1
HALLAZGOS_CRITICOS = ("neumotorax", "edema", "neumonia", "derrame", "consolidacion")
# images_001 a images_005
CANTIDAD_ARCHIVOS = 5
COLUMNA_UNICA = "columna_unica"
COLUMNA_POR_HALLAZGO = "columna_por_hallazgo"


@dataclass(frozen=True)
class Config:
    archivos: tuple[str, ...]
    urls: dict[str, str]
    checksum_algoritmo: str | None
    checksums: dict[str, str]
    columna_imagen: str
    formato_hallazgos: str
    columna_hallazgos: str
    separador_hallazgos: str
    valor_positivo: str
    hallazgos: dict[str, str]


def es_nombre_plano(nombre: object) -> bool:
    """true si es solo un nombre de archivo, sin carpetas ni '..'"""
    return (
        isinstance(nombre, str)
        and nombre not in ("", ".", "..")
        and "/" not in nombre
        and "\\" not in nombre
        and PurePath(nombre).name == nombre
    )


def es_https(url: object) -> bool:
    if not isinstance(url, str):
        return False
    partes = urlparse(url)
    return partes.scheme == "https" and bool(partes.netloc)


def _leer_json(ruta: Path) -> dict:
    try:
        datos = json.loads(ruta.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ErrorDatos(f"no existe el config: {ruta}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ErrorDatos(f"el config no es json valido: {e}") from None
    if not isinstance(datos, dict):
        raise ErrorDatos("el config debe ser un objeto json")
    return datos


def _texto(datos: dict, clave: str, errores: list[str]) -> str:
    valor = datos.get(clave)
    if isinstance(valor, str) and valor != "":
        return valor
    errores.append(f"'{clave}' debe ser texto no vacio")
    return ""


def _archivos(datos: dict, errores: list[str]) -> tuple[str, ...]:
    archivos = datos.get("archivos")
    if not isinstance(archivos, list) or len(archivos) != CANTIDAD_ARCHIVOS:
        errores.append(f"'archivos' debe traer exactamente {CANTIDAD_ARCHIVOS} nombres")
        return ()
    if not all(es_nombre_plano(a) for a in archivos):
        errores.append("'archivos' solo acepta nombres de archivo, sin carpetas")
    elif len(set(archivos)) != len(archivos):
        errores.append("'archivos' tiene nombres repetidos")
    return tuple(archivos)


def _urls(datos: dict, errores: list[str]) -> dict[str, str]:
    urls = datos.get("urls", {})
    if not isinstance(urls, dict):
        errores.append("'urls' debe ser un objeto nombre -> url")
        return {}
    for nombre, url in urls.items():
        if not es_nombre_plano(nombre):
            errores.append(f"'urls': nombre invalido {nombre!r}")
        if not es_https(url):
            errores.append(f"'urls': solo se acepta https ({nombre})")
    return urls


def _checksums(datos: dict, archivos: tuple[str, ...], errores: list[str]):
    checksums = datos.get("checksums", {})
    if not isinstance(checksums, dict):
        errores.append("'checksums' debe ser un objeto nombre -> hash")
        return None, {}
    if not checksums:
        return None, {}
    algoritmo = datos.get("checksum_algoritmo")
    if not (isinstance(algoritmo, str) and algoritmo in hashlib.algorithms_guaranteed):
        errores.append("'checksum_algoritmo' no es un algoritmo que soporte hashlib")
    ajenos = [n for n in checksums if n not in archivos]
    if ajenos:
        errores.append(f"'checksums' trae nombres que no estan en 'archivos': {ajenos}")
    if not all(isinstance(v, str) and v.strip() for v in checksums.values()):
        errores.append("'checksums': cada hash debe ser texto")
    return algoritmo, checksums


def _hallazgos(datos: dict, errores: list[str]) -> dict[str, str]:
    hallazgos = datos.get("hallazgos")
    if not isinstance(hallazgos, dict) or set(hallazgos) != set(HALLAZGOS_CRITICOS):
        errores.append(f"'hallazgos' debe traer exactamente: {', '.join(HALLAZGOS_CRITICOS)}")
        return {}
    vacios = [k for k, v in hallazgos.items() if not (isinstance(v, str) and v != "")]
    if vacios:
        errores.append(f"'hallazgos' sin valor en: {', '.join(vacios)}")
    return hallazgos


def cargar(ruta: Path) -> Config:
    datos = _leer_json(ruta)
    errores: list[str] = []
    archivos = _archivos(datos, errores)
    urls = _urls(datos, errores)
    algoritmo, checksums = _checksums(datos, archivos, errores)
    columna_imagen = _texto(datos, "columna_imagen", errores)

    # cada formato del csv pide campos distintos
    formato = datos.get("formato_hallazgos")
    columna_hallazgos = separador = valor_positivo = ""
    if formato == COLUMNA_UNICA:
        columna_hallazgos = _texto(datos, "columna_hallazgos", errores)
        separador = _texto(datos, "separador_hallazgos", errores)
    elif formato == COLUMNA_POR_HALLAZGO:
        valor_positivo = _texto(datos, "valor_positivo", errores)
    else:
        errores.append(f"'formato_hallazgos' debe ser '{COLUMNA_UNICA}' o '{COLUMNA_POR_HALLAZGO}'")

    hallazgos = _hallazgos(datos, errores)
    if errores:
        raise ErrorDatos("config incompleto o invalido:\n  - " + "\n  - ".join(errores))
    return Config(
        archivos=archivos,
        urls=urls,
        checksum_algoritmo=algoritmo,
        checksums=checksums,
        columna_imagen=columna_imagen,
        formato_hallazgos=formato,
        columna_hallazgos=columna_hallazgos,
        separador_hallazgos=separador,
        valor_positivo=valor_positivo,
        hallazgos=hallazgos,
    )
