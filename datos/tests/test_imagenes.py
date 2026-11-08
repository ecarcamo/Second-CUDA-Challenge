from PIL import Image

from datos import imagenes
from datos.tests.utiles import crear_png


def test_verificar_clasifica_cada_caso(tmp_path):
    crear_png(tmp_path / "gris.png", "L")
    crear_png(tmp_path / "color.png", "RGB")
    crear_png(tmp_path / "alfa.png", "RGBA")
    crear_png(tmp_path / "chica.png", "L", (512, 512))
    crear_png(tmp_path / "paleta.png", "P")
    (tmp_path / "basura.png").write_bytes(b"esto no es un png")
    # png con ruido para que pese algo y cortado a la mitad
    ruido = tmp_path / "ruido.png"
    Image.effect_noise((256, 256), 64).save(ruido, format="PNG")
    contenido = ruido.read_bytes()
    ruido.unlink()
    (tmp_path / "truncado.png").write_bytes(contenido[: len(contenido) // 2])
    (tmp_path / "nota.txt").write_text("se ignora", encoding="utf-8")

    res = imagenes.verificar(tmp_path)

    assert res.total == 7
    assert res.por_modo == {"L": 2, "RGB": 1, "RGBA": 1, "P": 1}
    assert res.tamano_distinto == [{"nombre": "chica.png", "ancho": 512, "alto": 512}]
    assert res.color == ["alfa.png", "color.png"]
    assert res.alfa == ["alfa.png"]
    assert res.paleta == ["paleta.png"]
    assert sorted(i["nombre"] for i in res.ilegibles) == ["basura.png", "truncado.png"]


def test_busca_png_en_subcarpetas(tmp_path):
    crear_png(tmp_path / "images" / "uno.png", "L", (8, 8))
    assert imagenes.verificar(tmp_path).nombres == ["uno.png"]


def test_carpeta_vacia(tmp_path):
    assert imagenes.verificar(tmp_path).total == 0
