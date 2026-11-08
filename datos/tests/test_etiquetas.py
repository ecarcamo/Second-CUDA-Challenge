import pytest

from datos import config, etiquetas
from datos.config import HALLAZGOS_CRITICOS
from datos.errores import ErrorDatos
from datos.tests.utiles import FILAS_CSV, escribir_config, escribir_csv


@pytest.fixture
def cfg(tmp_path):
    return config.cargar(escribir_config(tmp_path))


def test_filtra_y_lista_imagenes_sin_fila(tmp_path, cfg):
    _, filas = etiquetas.leer_csv(escribir_csv(tmp_path / "e.csv", FILAS_CSV), cfg)
    filtradas, sin_fila = etiquetas.filtrar(filas, ["b.png", "a.png", "x.png"])
    assert [f["imagen"] for f in filtradas] == ["a.png", "b.png"]
    assert sin_fila == ["x.png"]


def test_cuenta_positivos_columna_unica(tmp_path, cfg):
    _, filas = etiquetas.leer_csv(escribir_csv(tmp_path / "e.csv", FILAS_CSV), cfg)
    assert etiquetas.contar_positivos(filas.values(), cfg) == dict.fromkeys(HALLAZGOS_CRITICOS, 1)
    assert etiquetas.contar_positivos([filas["b.png"]], cfg) == dict.fromkeys(HALLAZGOS_CRITICOS, 0)


def test_imagen_repetida_en_csv_falla(tmp_path, cfg):
    ruta = escribir_csv(tmp_path / "e.csv", FILAS_CSV + [("a.png", "otra")])
    with pytest.raises(ErrorDatos, match="repetida"):
        etiquetas.leer_csv(ruta, cfg)


def test_columna_faltante_lista_las_que_si_hay(tmp_path, cfg):
    ruta = tmp_path / "e.csv"
    ruta.write_text("nombre,otra\na.png,x\n", encoding="utf-8")
    with pytest.raises(ErrorDatos, match="trae"):
        etiquetas.leer_csv(ruta, cfg)


def test_etiqueta_del_config_que_no_existe_falla(tmp_path, cfg):
    sin_consolidacion = [f for f in FILAS_CSV if f[0] != "d.png"]
    with pytest.raises(ErrorDatos, match="falso_neumonia"):
        etiquetas.leer_csv(escribir_csv(tmp_path / "e.csv", sin_consolidacion), cfg)


def test_formato_columna_por_hallazgo(tmp_path):
    columnas = {clave: f"col_{clave}" for clave in HALLAZGOS_CRITICOS}
    cfg = config.cargar(escribir_config(
        tmp_path, formato_hallazgos="columna_por_hallazgo", valor_positivo="1", hallazgos=columnas
    ))
    ruta = tmp_path / "e.csv"
    encabezado = ",".join(["imagen", *columnas.values()])
    ruta.write_text(f"{encabezado}\na.png,1,0,0,1,0\nb.png,0,0,0,0,0\n", encoding="utf-8")
    _, filas = etiquetas.leer_csv(ruta, cfg)
    assert etiquetas.hallazgos_de(filas["a.png"], cfg) == {"neumotorax", "derrame"}
    assert etiquetas.hallazgos_de(filas["b.png"], cfg) == set()


def test_lista_prueba_valida(tmp_path):
    ruta = tmp_path / "prueba.txt"
    ruta.write_text("a.png\n\nc.png\n", encoding="utf-8")
    assert etiquetas.leer_lista_prueba(ruta) == {"a.png", "c.png"}


def test_lista_prueba_con_formato_inesperado_falla(tmp_path):
    ruta = tmp_path / "prueba.txt"
    ruta.write_text("a.png,1\n", encoding="utf-8")
    with pytest.raises(ErrorDatos, match="png por linea"):
        etiquetas.leer_lista_prueba(ruta)


def test_lista_prueba_vacia_falla(tmp_path):
    ruta = tmp_path / "prueba.txt"
    ruta.write_text("\n", encoding="utf-8")
    with pytest.raises(ErrorDatos, match="vacia"):
        etiquetas.leer_lista_prueba(ruta)
