import numpy as np
import pytest
import torch

from cpu.preprocesado import preprocesar

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="sin gpu con cuda")


def _run(images: np.ndarray, method: str, dtype=torch.uint8) -> np.ndarray:
    from gpu.preprocess import Preprocessor
    pre = Preprocessor(*images.shape[1:])
    device = torch.from_numpy(images).to("cuda", dtype)
    out = torch.empty((len(images), 1, 224, 224), dtype=torch.float32, device="cuda")
    pre(device, out, method)
    torch.cuda.synchronize()
    return out.cpu().numpy()[:, 0]


@pytest.mark.parametrize("shape", [(1024, 1024), (1000, 1200), (1300, 900), (224, 224)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_kernel_matches_cpu(shape, dtype):
    images = np.random.default_rng(1).integers(0, 256, (3, *shape), dtype=np.uint8)
    out = _run(images, "kernel", dtype)
    for k in range(len(images)):
        np.testing.assert_allclose(out[k], preprocesar(images[k]), atol=1e-3)


@pytest.mark.parametrize("shape", [(1024, 1024), (1000, 1200)])
def test_kernel_matches_torch(shape):
    images = np.random.default_rng(2).integers(0, 256, (4, *shape), dtype=np.uint8)
    np.testing.assert_allclose(_run(images, "kernel"), _run(images, "torch"), atol=1e-3)


def test_normalized_range():
    images = np.stack([np.zeros((1024, 1024), np.uint8), np.full((1024, 1024), 255, np.uint8)])
    out = _run(images, "kernel")
    assert np.allclose(out[0], -1024) and np.allclose(out[1], 1024)


@pytest.mark.parametrize("extra", [[], ["--transferencia", "float32"], ["--preprocesado", "torch"],
                                   ["--memoria", "paginable"], ["--modo", "solapado"]])
def test_pipeline_matches_cpu(tmp_path, extra):
    import cv2

    from cpu import ejecutar as cpu_run
    from gpu import run as gpu_run

    folder = tmp_path / "png"
    folder.mkdir()
    rng = np.random.default_rng(3)
    for k in range(10):
        shape = (300, 260) if k == 4 else (256, 256)
        img = cv2.GaussianBlur(rng.integers(0, 256, shape, dtype=np.uint8), (0, 0), 4)
        cv2.imwrite(str(folder / f"img_{k:02d}.png"), img)
    common = ["--extraido", str(folder), "--etiquetas", str(tmp_path / "no_existe.csv"), "--lote", "4",
              "-r", "1", "--calentamiento", "0", "--hilos", "4", "--sin-pausas"]
    assert cpu_run.main([*common, "--salida", str(tmp_path / "cpu")]) == 0
    assert gpu_run.main([*common, "--salida", str(tmp_path / "gpu"), *extra]) == 0

    from comun.comparar import comparar, leer
    result = comparar(leer(tmp_path / "cpu" / "cola.csv"), leer(tmp_path / "gpu" / "cola.csv"))
    assert result["imagenes"] == 10
    assert result["dif_max_puntaje"] < 1e-3
