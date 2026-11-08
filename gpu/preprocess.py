from __future__ import annotations

import cupy as cp
import numpy as np
import torch
import torch.nn.functional as F

from comun.tarea import DESPLAZAMIENTO, ESCALA, TAMANO, intervalos_area, ventana_recorte

SOURCE = r"""
template <typename T>
__global__ void crop_area_normalize(const T* __restrict__ src, float* __restrict__ dst,
                                    const int* __restrict__ starts, const int* __restrict__ ends,
                                    int height, int width, int row0, int col0, int out_size,
                                    float scale, float shift)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_size || y >= out_size) return;
    const T* img = src + (size_t)blockIdx.z * height * width;
    int r0 = starts[y], r1 = ends[y], c0 = starts[x], c1 = ends[x];
    float sum = 0.0f;
    for (int r = r0; r < r1; ++r) {
        const T* row = img + (size_t)(row0 + r) * width + col0;
        for (int c = c0; c < c1; ++c) sum += (float)row[c];
    }
    float mean = sum / (float)((r1 - r0) * (c1 - c0));
    dst[(size_t)blockIdx.z * out_size * out_size + y * out_size + x] = __fadd_rn(__fmul_rn(mean, scale), shift);
}
"""
NAMES = ("crop_area_normalize<unsigned char>", "crop_area_normalize<float>")
BLOCK = (16, 16, 1)


def _cupy_stream() -> cp.cuda.ExternalStream:
    return cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)


class Preprocessor:
    def __init__(self, height: int, width: int, size: int = TAMANO):
        self.height, self.width, self.size = height, width, size
        self.row0, self.col0, self.side = ventana_recorte(height, width)
        starts, ends = intervalos_area(self.side, size)
        self.starts = cp.asarray(starts, dtype=cp.int32)
        self.ends = cp.asarray(ends, dtype=cp.int32)
        module = cp.RawModule(code=SOURCE, options=("-std=c++17",), name_expressions=NAMES)
        self.kernels = {torch.uint8: module.get_function(NAMES[0]), torch.float32: module.get_function(NAMES[1])}

    def kernel(self, images: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        n = images.shape[0]
        grid = ((self.size + BLOCK[0] - 1) // BLOCK[0], (self.size + BLOCK[1] - 1) // BLOCK[1], n)
        args = (cp.from_dlpack(images.contiguous()), cp.from_dlpack(out), self.starts, self.ends,
                np.int32(self.height), np.int32(self.width), np.int32(self.row0), np.int32(self.col0),
                np.int32(self.size), np.float32(ESCALA), np.float32(DESPLAZAMIENTO))
        with _cupy_stream():
            self.kernels[images.dtype](grid, BLOCK, args)
        return out

    def torch_reference(self, images: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        crop = images[:, self.row0:self.row0 + self.side, self.col0:self.col0 + self.side]
        x = F.adaptive_avg_pool2d(crop.unsqueeze(1).float(), self.size)
        torch.add(x * ESCALA, DESPLAZAMIENTO, out=out)
        return out

    def __call__(self, images: torch.Tensor, out: torch.Tensor, method: str = "kernel") -> torch.Tensor:
        if method == "kernel":
            return self.kernel(images, out)
        return self.torch_reference(images, out)
