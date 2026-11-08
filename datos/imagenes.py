"""revisa tamaño, modo e integridad de cada png sin decodificar los pixeles"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# informe 1, tabla 1: png de 1024 x 1024
TAMANO_ESPERADO = (1024, 1024)


@dataclass
class ResultadoImagenes:
    nombres: list[str] = field(default_factory=list)
    por_modo: dict[str, int] = field(default_factory=dict)
    tamano_distinto: list[dict] = field(default_factory=list)
    color: list[str] = field(default_factory=list)
    alfa: list[str] = field(default_factory=list)
    # la paleta va aparte para no clasificarla a ojo
    paleta: list[str] = field(default_factory=list)
    ilegibles: list[dict] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.nombres)


def listar_pngs(carpeta: Path) -> list[Path]:
    return sorted(p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() == ".png")


def _leer_cabecera(ruta: Path):
    # formats=["PNG"] para que pillow no pruebe otros parsers
    with Image.open(ruta, formats=["PNG"]) as im:
        datos = im.mode, im.size, im.getbands()
        im.verify()  # revisa los crc de los chunks, no decodifica
    return datos


def verificar(carpeta: Path) -> ResultadoImagenes:
    res = ResultadoImagenes()
    for ruta in listar_pngs(carpeta):
        res.nombres.append(ruta.name)
        try:
            modo, tamano, bandas = _leer_cabecera(ruta)
        except Exception as e:  # pillow tira errores muy variados con archivos rotos
            res.ilegibles.append({"nombre": ruta.name, "motivo": f"{type(e).__name__}: {e}"})
            continue
        res.por_modo[modo] = res.por_modo.get(modo, 0) + 1
        if tamano != TAMANO_ESPERADO:
            res.tamano_distinto.append({"nombre": ruta.name, "ancho": tamano[0], "alto": tamano[1]})
        if {"R", "G", "B"} <= set(bandas):
            res.color.append(ruta.name)
        if "A" in bandas:
            res.alfa.append(ruta.name)
        if modo == "P":
            res.paleta.append(ruta.name)
    return res
