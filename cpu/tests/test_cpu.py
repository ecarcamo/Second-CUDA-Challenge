import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from comun import tarea
from cpu.preprocesado import preprocesar, preprocesar_lote


def _referencia_torch(img):
    """recorte de xrv + reduccion por area de pytorch + normalizacion de xrv"""
    f, c, lado = tarea.ventana_recorte(*img.shape)
    x = torch.from_numpy(img[f:f + lado, c:c + lado].astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(x, tarea.TAMANO)[0, 0].numpy() * 2048 / 255 - 1024


@pytest.mark.parametrize("forma", [(1024, 1024), (1000, 1200), (1300, 900), (224, 224)])
def test_preprocesado_igual_a_pytorch(forma):
    img = np.random.default_rng(1).integers(0, 256, forma, dtype=np.uint8)
    out = preprocesar(img)
    assert out.shape == (224, 224) and out.dtype == np.float32
    np.testing.assert_allclose(out, _referencia_torch(img), atol=1e-3)


def test_rango_normalizado():
    negro = np.zeros((1024, 1024), np.uint8)
    blanco = np.full((1024, 1024), 255, np.uint8)
    assert np.allclose(preprocesar(negro), -1024)
    assert np.allclose(preprocesar(blanco), 1024)


def test_cercano_al_pipeline_de_xrv():
    # xrv usa normalize -> center crop -> resize; con cv2 INTER_AREA el resultado es casi igual
    import torchxrayvision as xrv
    img = cv2.GaussianBlur(np.random.default_rng(2).integers(0, 256, (1024, 1024), dtype=np.uint8), (0, 0), 3)
    x = xrv.datasets.normalize(img, 255)[None]
    x = xrv.datasets.XRayCenterCrop()(x)
    x = xrv.datasets.XRayResizer(224, engine="cv2")(x)[0]
    # las cajas de pytorch se traslapan y las de cv2 usan pesos fraccionarios:
    # la diferencia media queda bajo 0.3 % de la escala de 2048
    assert np.abs(preprocesar(img) - x).mean() < 6


def test_lote_con_y_sin_hilos():
    from concurrent.futures import ThreadPoolExecutor
    imgs = [np.random.default_rng(i).integers(0, 256, (512, 512), dtype=np.uint8) for i in range(5)]
    with ThreadPoolExecutor(3) as pool:
        a = preprocesar_lote(imgs, pool)
    assert a.shape == (5, 1, 224, 224)
    np.testing.assert_array_equal(a, preprocesar_lote(imgs))


def test_decodificar_color_y_alfa():
    gris = np.random.default_rng(3).integers(0, 256, (64, 64), dtype=np.uint8)
    rgba = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGRA)
    rgba[..., 3] = 7  # alfa casi transparente: se ignora
    for img in (gris, rgba, gris.astype(np.uint16) * 257):
        ok, png = cv2.imencode(".png", img)
        out = tarea.decodificar(png.ravel())
        assert out.dtype == np.uint8 and out.shape == (64, 64)
        assert np.abs(out.astype(int) - gris).max() <= 1


def test_indices_hallazgos():
    pat = ["Atelectasis", "Consolidation", "Infiltration", "Pneumothorax", "Edema",
           "Emphysema", "Fibrosis", "Effusion", "Pneumonia"]
    assert tarea.indices_hallazgos(pat) == [3, 4, 8, 7, 1]
    with pytest.raises(ValueError):
        tarea.indices_hallazgos(pat[:3])


def test_listar_respeta_csv(tmp_path):
    for sub, n in [("a", "00000002_000.png"), ("b", "00000001_000.png"), ("b", "99999999_000.png")]:
        (tmp_path / sub).mkdir(exist_ok=True)
        (tmp_path / sub / n).write_bytes(b"")
    csv = tmp_path / "e.csv"
    csv.write_text("Image Index,Finding Labels\n00000002_000.png,x\n00000001_000.png,y\n")
    rutas = tarea.listar_imagenes(tmp_path, csv)
    assert [r.name for r in rutas] == ["00000001_000.png", "00000002_000.png"]
    assert len(tarea.listar_imagenes(tmp_path, None)) == 3
