"""filtra el csv de etiquetas al subconjunto, cuenta positivos y lee la particion de prueba"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from datos.config import COLUMNA_UNICA, HALLAZGOS_CRITICOS, Config
from datos.errores import ErrorDatos

Fila = dict[str, str]


def _columnas_requeridas(cfg: Config) -> list[str]:
    if cfg.formato_hallazgos == COLUMNA_UNICA:
        return [cfg.columna_imagen, cfg.columna_hallazgos]
    return [cfg.columna_imagen, *cfg.hallazgos.values()]


def _etiquetas_de(fila: Fila, cfg: Config) -> set[str]:
    texto = fila.get(cfg.columna_hallazgos) or ""
    return {e.strip() for e in texto.split(cfg.separador_hallazgos)}


def _validar_etiquetas(filas: Iterable[Fila], cfg: Config) -> None:
    # en columna_por_hallazgo ya se validaron las columnas, aqui solo aplica columna_unica
    if cfg.formato_hallazgos != COLUMNA_UNICA:
        return
    vistas = set().union(*(_etiquetas_de(f, cfg) for f in filas))
    faltan = [e for e in cfg.hallazgos.values() if e not in vistas]
    if faltan:
        raise ErrorDatos(f"estas etiquetas del config no aparecen en el csv: {faltan}")


def leer_csv(ruta: Path, cfg: Config) -> tuple[list[str], dict[str, Fila]]:
    """devuelve (columnas, filas por nombre de imagen)"""
    try:
        with ruta.open(newline="", encoding="utf-8-sig") as f:
            lector = csv.DictReader(f)
            columnas = list(lector.fieldnames or [])
            faltan = [c for c in _columnas_requeridas(cfg) if c not in columnas]
            if faltan:
                raise ErrorDatos(f"el csv no tiene las columnas {faltan}; trae {columnas}")
            filas: dict[str, Fila] = {}
            for fila in lector:
                nombre = (fila.get(cfg.columna_imagen) or "").strip()
                if nombre in filas:
                    raise ErrorDatos(f"imagen repetida en el csv: {nombre!r}")
                filas[nombre] = fila
    except FileNotFoundError:
        raise ErrorDatos(f"no existe el csv: {ruta}") from None
    except (UnicodeDecodeError, csv.Error) as e:
        raise ErrorDatos(f"no se pudo leer el csv: {e}") from None
    _validar_etiquetas(filas.values(), cfg)
    return columnas, filas


def hallazgos_de(fila: Fila, cfg: Config) -> set[str]:
    """claves de HALLAZGOS_CRITICOS que salen positivas en la fila"""
    if cfg.formato_hallazgos == COLUMNA_UNICA:
        etiquetas = _etiquetas_de(fila, cfg)
        return {clave for clave, etiqueta in cfg.hallazgos.items() if etiqueta in etiquetas}
    return {
        clave
        for clave, columna in cfg.hallazgos.items()
        if (fila.get(columna) or "").strip() == cfg.valor_positivo
    }


def contar_positivos(filas: Iterable[Fila], cfg: Config) -> dict[str, int]:
    conteo = dict.fromkeys(HALLAZGOS_CRITICOS, 0)
    for fila in filas:
        for clave in hallazgos_de(fila, cfg):
            conteo[clave] += 1
    return conteo


def filtrar(filas: dict[str, Fila], nombres: Iterable[str]) -> tuple[list[Fila], list[str]]:
    """devuelve (filas del subconjunto, imagenes que no tienen fila)"""
    unicos = sorted(set(nombres))
    return [filas[n] for n in unicos if n in filas], [n for n in unicos if n not in filas]


def leer_lista_prueba(ruta: Path) -> set[str]:
    try:
        texto = ruta.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ErrorDatos(f"no existe la lista de prueba: {ruta}") from None
    except UnicodeDecodeError as e:
        raise ErrorDatos(f"no se pudo leer la lista de prueba: {e}") from None
    nombres = [linea.strip() for linea in texto.splitlines() if linea.strip()]
    if not nombres:
        raise ErrorDatos("la lista de prueba esta vacia")
    # si no es un png por linea, mejor avisar que adivinar el formato
    raros = [n for n in nombres if not n.lower().endswith(".png")]
    if raros:
        raise ErrorDatos(f"la lista de prueba no parece un png por linea, ej: {raros[0]!r}")
    return set(nombres)


def escribir_csv(ruta: Path, columnas: list[str], filas: list[Fila]) -> None:
    with ruta.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(filas)
