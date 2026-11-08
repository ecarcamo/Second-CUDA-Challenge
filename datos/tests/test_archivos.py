import hashlib
import io
import tarfile
import zipfile

import pytest

from datos import archivos
from datos.errores import ErrorDatos
from datos.tests.utiles import crear_tar, png_bytes


class RespuestaFalsa(io.BytesIO):
    """imita lo que devuelve urlopen, sin tocar la red"""

    def __init__(self, contenido: bytes, url_final: str):
        super().__init__(contenido)
        self._url_final = url_final

    def geturl(self) -> str:
        return self._url_final


def abrir_falso(contenido=b"contenido", redirigir_a=None):
    def abrir(url, timeout):
        return RespuestaFalsa(contenido, redirigir_a or url)
    return abrir


# --- validar_presentes ---

def test_validar_presentes_nombra_el_faltante(tmp_path):
    (tmp_path / "uno.tar.gz").write_bytes(b"x")
    with pytest.raises(ErrorDatos, match="dos.tar.gz"):
        archivos.validar_presentes(tmp_path, ["uno.tar.gz", "dos.tar.gz"])


# --- descarga (red simulada) ---

def test_descargar_baja_y_no_deja_parciales(tmp_path):
    bajados = archivos.descargar({"a.bin": "https://ejemplo.invalid/a"}, tmp_path, abrir_falso(b"hola"))
    assert bajados == ["a.bin"]
    assert (tmp_path / "a.bin").read_bytes() == b"hola"
    assert not list(tmp_path.glob("*.parcial"))


def test_descargar_salta_lo_que_ya_existe(tmp_path):
    (tmp_path / "a.bin").write_bytes(b"viejo")
    bajados = archivos.descargar({"a.bin": "https://ejemplo.invalid/a"}, tmp_path, abrir_falso(b"nuevo"))
    assert bajados == []
    assert (tmp_path / "a.bin").read_bytes() == b"viejo"


def test_descargar_rechaza_http(tmp_path):
    with pytest.raises(ErrorDatos, match="https"):
        archivos.descargar({"a.bin": "http://ejemplo.invalid/a"}, tmp_path, abrir_falso())


def test_descargar_rechaza_redireccion_a_http(tmp_path):
    abrir = abrir_falso(redirigir_a="http://ejemplo.invalid/otro")
    with pytest.raises(ErrorDatos, match="redirigio"):
        archivos.descargar({"a.bin": "https://ejemplo.invalid/a"}, tmp_path, abrir)
    assert not list(tmp_path.iterdir())


def test_descargar_con_fallo_de_red_limpia_el_parcial(tmp_path):
    def abrir_que_falla(url, timeout):
        raise OSError("sin conexion")
    with pytest.raises(ErrorDatos, match="sin conexion"):
        archivos.descargar({"a.bin": "https://ejemplo.invalid/a"}, tmp_path, abrir_que_falla)
    assert not list(tmp_path.iterdir())


# --- checksum ---

def test_checksum_correcto_y_incorrecto(tmp_path):
    ruta = tmp_path / "a.bin"
    ruta.write_bytes(b"datos")
    archivos.verificar_checksum(ruta, "sha256", hashlib.sha256(b"datos").hexdigest())
    with pytest.raises(ErrorDatos, match="checksum"):
        archivos.verificar_checksum(ruta, "sha256", "0" * 64)


# --- descompresion ---

def test_extraer_tar_saca_solo_png(tmp_path):
    tar = crear_tar(tmp_path / "a.tar.gz", {
        "images/uno.png": png_bytes(tamano=(4, 4)),
        "images/dos.png": png_bytes(tamano=(4, 4)),
        "LEEME.txt": b"texto",
    })
    destino = tmp_path / "salida"
    assert archivos.extraer_pngs(tar, destino) == (2, 1)
    assert sorted(p.name for p in destino.rglob("*.png")) == ["dos.png", "uno.png"]
    assert not list(destino.rglob("*.txt"))


def test_extraer_zip_valido(tmp_path):
    ruta = tmp_path / "a.zip"
    with zipfile.ZipFile(ruta, "w") as zf:
        zf.writestr("images/uno.png", png_bytes(tamano=(4, 4)))
    assert archivos.extraer_pngs(ruta, tmp_path / "salida") == (1, 0)


def test_extraer_rechaza_ruta_con_puntos(tmp_path):
    tar = crear_tar(tmp_path / "malo.tar.gz", {"../fuera.png": b"x"})
    destino = tmp_path / "salida"
    with pytest.raises(ErrorDatos, match="insegura"):
        archivos.extraer_pngs(tar, destino)
    assert not (tmp_path / "fuera.png").exists()


def test_extraer_rechaza_ruta_absoluta(tmp_path):
    ruta = tmp_path / "malo.zip"
    with zipfile.ZipFile(ruta, "w") as zf:
        zf.writestr("/absoluta.png", b"x")
    with pytest.raises(ErrorDatos, match="insegura"):
        archivos.extraer_pngs(ruta, tmp_path / "salida")


def test_extraer_rechaza_enlace_simbolico(tmp_path):
    ruta = tmp_path / "enlace.tar.gz"
    with tarfile.open(ruta, "w:gz") as tar:
        info = tarfile.TarInfo("images/enlace.png")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ErrorDatos, match="no permitido"):
        archivos.extraer_pngs(ruta, tmp_path / "salida")


def test_extraer_formato_no_soportado(tmp_path):
    ruta = tmp_path / "texto.tar.gz"
    ruta.write_text("no soy un comprimido", encoding="utf-8")
    with pytest.raises(ErrorDatos, match="no soportado"):
        archivos.extraer_pngs(ruta, tmp_path / "salida")
