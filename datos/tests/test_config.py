from pathlib import Path

import pytest

from datos import config
from datos.errores import ErrorDatos
from datos.tests.utiles import escribir_config


def test_config_valida_carga(tmp_path):
    cfg = config.cargar(escribir_config(tmp_path))
    assert len(cfg.archivos) == 5
    assert cfg.hallazgos["edema"] == "falso_edema"


def test_config_del_repo_es_valido():
    # el dataset.json real tiene que pasar la validacion tal cual
    cfg = config.cargar(Path(config.__file__).parent / "dataset.json")
    assert set(cfg.urls) == set(cfg.archivos)


def test_lista_todos_los_campos_que_faltan(tmp_path):
    ruta = escribir_config(tmp_path, columna_imagen="", hallazgos={})
    with pytest.raises(ErrorDatos) as error:
        config.cargar(ruta)
    assert "columna_imagen" in str(error.value)
    assert "hallazgos" in str(error.value)


def test_rechaza_url_sin_https(tmp_path):
    ruta = escribir_config(tmp_path, urls={"falso_001.tar.gz": "http://ejemplo.invalid/a"})
    with pytest.raises(ErrorDatos, match="https"):
        config.cargar(ruta)


def test_rechaza_nombre_con_carpetas(tmp_path):
    nombres = ["../malo.tar.gz"] + [f"falso_{i:03d}.tar.gz" for i in range(2, 6)]
    with pytest.raises(ErrorDatos, match="sin carpetas"):
        config.cargar(escribir_config(tmp_path, archivos=nombres))


def test_rechaza_cantidad_distinta_de_archivos(tmp_path):
    with pytest.raises(ErrorDatos, match="exactamente 5"):
        config.cargar(escribir_config(tmp_path, archivos=["solo_uno.tar.gz"]))


def test_rechaza_algoritmo_de_checksum_desconocido(tmp_path):
    ruta = escribir_config(
        tmp_path, checksum_algoritmo="no_existe", checksums={"falso_001.tar.gz": "abc"}
    )
    with pytest.raises(ErrorDatos, match="checksum_algoritmo"):
        config.cargar(ruta)


def test_columna_por_hallazgo_pide_valor_positivo(tmp_path):
    ruta = escribir_config(tmp_path, formato_hallazgos="columna_por_hallazgo", valor_positivo="")
    with pytest.raises(ErrorDatos, match="valor_positivo"):
        config.cargar(ruta)


def test_config_inexistente(tmp_path):
    with pytest.raises(ErrorDatos, match="no existe"):
        config.cargar(tmp_path / "nada.json")
