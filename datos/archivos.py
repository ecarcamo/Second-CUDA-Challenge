"""validacion, descarga y descompresion segura de los comprimidos del nih"""
from __future__ import annotations

import hashlib
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import IO, Callable
from urllib.request import urlopen

from datos.config import es_https
from datos.errores import ErrorDatos

_BLOQUE = 1024 * 1024
_TIMEOUT_S = 60


def validar_presentes(carpeta: Path, nombres) -> list[Path]:
    faltan = [n for n in nombres if not (carpeta / n).is_file()]
    if faltan:
        raise ErrorDatos(f"faltan archivos en {carpeta}: {', '.join(faltan)}")
    return [carpeta / n for n in nombres]


def _bajar(abrir: Callable, url: str, destino: Path) -> None:
    try:
        with abrir(url, timeout=_TIMEOUT_S) as respuesta:
            # una redireccion podria mandarnos a http, eso no
            if not es_https(respuesta.geturl()):
                raise ErrorDatos(f"la descarga redirigio fuera de https: {url}")
            with destino.open("wb") as f:
                shutil.copyfileobj(respuesta, f, _BLOQUE)
    except OSError as e:
        raise ErrorDatos(f"fallo la descarga de {url}: {e}") from None


def descargar(urls: dict[str, str], carpeta: Path, abrir: Callable = urlopen) -> list[str]:
    """baja lo que falte y devuelve los nombres que si se bajaron"""
    carpeta.mkdir(parents=True, exist_ok=True)
    bajados = []
    for nombre, url in urls.items():
        if not es_https(url):
            raise ErrorDatos(f"solo se permite https: {nombre}")
        destino = carpeta / nombre
        if destino.is_file():
            continue  # ya estaba, no se vuelve a bajar
        parcial = destino.with_name(destino.name + ".parcial")
        try:
            _bajar(abrir, url, parcial)
        except BaseException:
            # nada de archivos a medias, ni con ctrl+c
            parcial.unlink(missing_ok=True)
            raise
        parcial.replace(destino)
        bajados.append(nombre)
    return bajados


def verificar_checksum(ruta: Path, algoritmo: str, esperado: str) -> None:
    calculado = hashlib.new(algoritmo)
    with ruta.open("rb") as f:
        for bloque in iter(lambda: f.read(_BLOQUE), b""):
            calculado.update(bloque)
    if calculado.hexdigest().lower() != esperado.strip().lower():
        raise ErrorDatos(f"el checksum no coincide: {ruta.name}")


def _es_png(nombre: str) -> bool:
    return nombre.lower().endswith(".png")


def _ruta_segura(destino: Path, nombre: str) -> Path:
    """evita que un miembro del comprimido escriba fuera de destino"""
    partes = PurePosixPath(nombre.replace("\\", "/")).parts
    inseguro = (
        not partes
        or nombre.startswith(("/", "\\"))
        or ".." in partes
        or ":" in partes[0]  # letras de unidad tipo c:
    )
    if not inseguro:
        final = destino.joinpath(*partes).resolve()
        inseguro = not final.is_relative_to(destino.resolve())
    if inseguro:
        raise ErrorDatos(f"ruta insegura dentro del comprimido: {nombre!r}")
    return final


def _copiar(origen: IO[bytes], ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with ruta.open("wb") as f:
        shutil.copyfileobj(origen, f, _BLOQUE)


def _extraer_tar(archivo: Path, destino: Path) -> tuple[int, int]:
    pngs = otros = 0
    with tarfile.open(archivo, "r:*") as tar:
        for miembro in tar:
            if miembro.isdir():
                continue
            # enlaces, dispositivos y demas quedan fuera
            if not miembro.isfile():
                raise ErrorDatos(f"miembro no permitido (enlace o especial): {miembro.name!r}")
            if not _es_png(miembro.name):
                otros += 1
                continue
            ruta = _ruta_segura(destino, miembro.name)
            _copiar(tar.extractfile(miembro), ruta)
            pngs += 1
    return pngs, otros


def _extraer_zip(archivo: Path, destino: Path) -> tuple[int, int]:
    pngs = otros = 0
    with zipfile.ZipFile(archivo) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ErrorDatos(f"miembro no permitido (enlace): {info.filename!r}")
            if not _es_png(info.filename):
                otros += 1
                continue
            ruta = _ruta_segura(destino, info.filename)
            with zf.open(info) as origen:
                _copiar(origen, ruta)
            pngs += 1
    return pngs, otros


def extraer_pngs(archivo: Path, destino: Path) -> tuple[int, int]:
    """saca solo los png; devuelve (pngs extraidos, otros archivos ignorados)"""
    destino.mkdir(parents=True, exist_ok=True)
    # se detecta por contenido, no por extension
    try:
        if zipfile.is_zipfile(archivo):
            return _extraer_zip(archivo, destino)
        if tarfile.is_tarfile(archivo):
            return _extraer_tar(archivo, destino)
    except (tarfile.TarError, zipfile.BadZipFile, EOFError, OSError) as e:
        # puede ser comprimido dañado o disco lleno, el mensaje trae el detalle
        raise ErrorDatos(f"no se pudo descomprimir {archivo.name}: {e}") from None
    raise ErrorDatos(f"formato no soportado (ni tar ni zip): {archivo.name}")
