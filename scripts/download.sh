#!/usr/bin/env bash
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/datos/archivos_nih"
meta="$root/datos/nih_meta"
mkdir -p "$dest" "$meta"
fetch() {
  local url="$1" out="$2"
  [ -f "$out" ] && return 0
  until nice -n 10 curl -fsSL --retry 20 --retry-delay 15 --retry-all-errors --speed-limit 20000 --speed-time 120 -C - -o "$out.parcial" "$url"; do
    echo "$(date +%T) reintento $(basename "$out")"; sleep 20
  done
  mv "$out.parcial" "$out"
  echo "$(date +%T) listo $(basename "$out") $(stat -c %s "$out")"
}
hf=https://huggingface.co/datasets/alkzar90/NIH-Chest-X-ray-dataset/resolve/main/data
for f in Data_Entry_2017_v2020.csv test_list.txt train_val_list.txt; do fetch "$hf/$f" "$meta/$f"; done
python3 - "$root/datos/dataset.json" <<'PY' > "$meta/urls.txt"
import json, sys
for k, v in json.load(open(sys.argv[1]))["urls"].items(): print(k, v)
PY
while read -r name url; do fetch "$url" "$dest/$name" & done < "$meta/urls.txt"
wait
echo "$(date +%T) descarga completa"
ls -l "$dest"
