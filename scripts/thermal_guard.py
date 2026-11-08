from __future__ import annotations

import os
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comun.thermal import cpu_temperature, gpu_temperature

PERIOD_S = 15
HOT_CPU, HOT_GPU = 99.0, 87.0
HOT_FOR_S = 60
RESUME_CPU, RESUME_GPU = 92.0, 75.0
MAX_PAUSE_S = 300


def fan() -> str:
    for path in Path("/sys/class/hwmon").glob("hwmon*/fan1_input"):
        try:
            return path.read_text().strip()
        except OSError:
            pass
    return ""


def main() -> int:
    pgid, log_dir = int(sys.argv[1]), Path(sys.argv[2])
    log_dir.mkdir(parents=True, exist_ok=True)
    temps = (log_dir / "temperaturas.csv").open("a", buffering=1)
    events = (log_dir / "guardian.log").open("a", buffering=1)
    if temps.tell() == 0:
        temps.write("hora,cpu_c,gpu_c,ventilador_rpm,pausado\n")
    hot_since, paused_at = None, None
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return 0
        cpu, gpu = cpu_temperature(), gpu_temperature()
        now = time.monotonic()
        hot = (cpu or 0) >= HOT_CPU or (gpu or 0) >= HOT_GPU
        temps.write(f"{datetime.now():%Y-%m-%d %H:%M:%S},{cpu},{gpu},{fan()},{int(paused_at is not None)}\n")
        if paused_at is None:
            hot_since = (hot_since or now) if hot else None
            if hot_since is not None and now - hot_since >= HOT_FOR_S:
                os.killpg(pgid, signal.SIGSTOP)
                paused_at = now
                events.write(f"{datetime.now():%F %T} pausa: cpu {cpu} °C, gpu {gpu} °C\n")
        else:
            cooled = (cpu or 0) <= RESUME_CPU and (gpu or 0) <= RESUME_GPU
            if cooled or now - paused_at >= MAX_PAUSE_S:
                os.killpg(pgid, signal.SIGCONT)
                events.write(f"{datetime.now():%F %T} sigue tras {now - paused_at:.0f} s: cpu {cpu} °C, gpu {gpu} °C\n")
                paused_at, hot_since = None, None
        time.sleep(PERIOD_S)


if __name__ == "__main__":
    sys.exit(main())
