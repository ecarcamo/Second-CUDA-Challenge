from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


def _hwmon(name: str) -> Path | None:
    for path in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (path / "name").read_text().strip() == name:
                return path
        except OSError:
            continue
    return None


def cpu_temperature() -> float | None:
    path = _hwmon("coretemp")
    if path is None:
        return None
    try:
        return int((path / "temp1_input").read_text()) / 1000
    except (OSError, ValueError):
        return None


def gpu_temperature() -> float | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return float(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def cool_down(min_s: float = 120, max_s: float = 600, cpu_max: float = 92, gpu_max: float = 70,
              label: str = "") -> dict:
    start = time.perf_counter()
    time.sleep(min_s)
    cpu, gpu = cpu_temperature(), gpu_temperature()
    while time.perf_counter() - start < max_s:
        if (cpu is None or cpu <= cpu_max) and (gpu is None or gpu <= gpu_max):
            break
        time.sleep(10)
        cpu, gpu = cpu_temperature(), gpu_temperature()
    waited = time.perf_counter() - start
    print(f"  pausa {label}: {waited:.0f} s, cpu {cpu} °C, gpu {gpu} °C", file=sys.stderr, flush=True)
    return {"antes_de": label, "espera_s": round(waited, 1), "cpu_c": cpu, "gpu_c": gpu}
