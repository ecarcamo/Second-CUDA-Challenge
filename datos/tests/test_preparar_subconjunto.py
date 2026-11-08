"""prueba de punta a punta con comprimidos, csv y lista falsos, sin red"""
import csv
import json

import pytest

from datos.preparar_subconjunto import main
from datos.tests.utiles import FILAS_CSV, config_base, crear_tar, escribir_config, escribir_csv, png_bytes

# reparte a.png..e.png entre los 5 comprimidos falsos
IMAGENES = ["a.png", "b.png", "c.png", "d.png", "e.png"]


def _preparar_escenario(tmp_path, filas_csv):
    carpeta = tmp_path / "archivos"
    carpeta.mkdir()
    for nombre_tar, imagen in zip(config_base()["archivos"], IMAGENES):
        crear_tar(carpeta / nombre_tar, {f"images/{imagen}": png_bytes(tamano=(8, 8))})
    lista = tmp_path / "prueba.txt"
    lista.write_text("a.png\nd.png\nz.png\n", encoding="utf-8")
    return [
        "--config", str(escribir_config(tmp_path)),
        "--archivos", str(carpeta),
        "--extraido", str(tmp_path / "extraido"),
        "--salida", str(tmp_path / "salidas"),
        "--csv", str(escribir_csv(tmp_path / "e.csv", filas_csv)),
        "--lista-prueba", str(lista),
    ]


def test_corrida_completa_genera_las_tres_salidas(tmp_path):
    filas = FILAS_CSV + [("e.png", "sin_hallazgo"), ("fuera.png", "falso_edema")]
    assert main(_preparar_escenario(tmp_path, filas)) == 0

    salidas = tmp_path / "salidas"
    reporte = json.loads((salidas / "reporte.json").read_text(encoding="utf-8"))
    assert reporte["imagenes"]["total"] == 5
    assert reporte["archivos"]["extraccion"] == {"pngs": 5, "otros_ignorados": 0}
    # todas miden 8x8, asi que todas salen con tamaño distinto a 1024
    assert len(reporte["imagenes"]["tamano_distinto"]) == 5
    assert reporte["etiquetas"]["positivos"]["edema"] == 1
    assert reporte["particion_prueba"] == {
        "total": 2,
        "positivos": {"neumotorax": 0, "edema": 1, "neumonia": 1, "derrame": 1, "consolidacion": 1},
    }
    assert reporte["problemas"] == []

    with (salidas / "etiquetas_subconjunto.csv").open(encoding="utf-8", newline="") as f:
        assert [fila["imagen"] for fila in csv.DictReader(f)] == IMAGENES
    assert (salidas / "particion_prueba.txt").read_text(encoding="utf-8") == "a.png\nd.png\n"


def test_imagen_sin_fila_termina_con_codigo_2(tmp_path, capsys):
    assert main(_preparar_escenario(tmp_path, FILAS_CSV)) == 2
    assert "sin fila" in capsys.readouterr().err


def test_extraido_con_contenido_no_se_pisa(tmp_path, capsys):
    argumentos = _preparar_escenario(tmp_path, FILAS_CSV + [("e.png", "x")])
    (tmp_path / "extraido").mkdir()
    (tmp_path / "extraido" / "previo.png").write_bytes(b"x")
    assert main(argumentos) == 1
    assert "no esta vacia" in capsys.readouterr().err


def test_omitir_extraccion_reusa_lo_que_hay(tmp_path):
    argumentos = _preparar_escenario(tmp_path, FILAS_CSV + [("e.png", "x")])
    assert main(argumentos) == 0
    assert main(argumentos + ["--omitir-extraccion"]) == 0


def test_config_vacio_termina_con_codigo_1(tmp_path, capsys):
    argumentos = _preparar_escenario(tmp_path, FILAS_CSV)
    argumentos[1] = str(tmp_path / "no_existe.json")
    assert main(argumentos) == 1
    assert "no existe el config" in capsys.readouterr().err


def test_ayuda_funciona():
    with pytest.raises(SystemExit) as salida:
        main(["--help"])
    assert salida.value.code == 0
