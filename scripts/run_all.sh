#!/usr/bin/env bash
set -u
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
py="$root/.venv/bin/python"
res="$root/resultados"
logs="$res/logs"
mkdir -p "$logs"
say() { echo "$(date '+%F %T') $*"; }
fail() { say "ERROR: $*"; exit 1; }

setsid "$py" scripts/thermal_guard.py "$$" "$logs" &
guard=$!
trap 'kill $guard 2>/dev/null' EXIT

say "esperando la descarga"
for f in images_001 images_002 images_003 images_004 images_005; do
  until [ -f "datos/archivos_nih/$f.tar.gz" ]; do sleep 60; done
done
say "descarga lista: $(du -sh datos/archivos_nih | cut -f1)"

if [ ! -f datos/salidas/reporte.json ]; then
  say "preparando el subconjunto"
  ionice -c3 nice -n 10 "$py" -m datos.preparar_subconjunto \
    --csv datos/nih_meta/Data_Entry_2017_v2020.csv --lista-prueba datos/nih_meta/test_list.txt \
    > "$logs/preparar.log" 2>&1
  code=$?
  [ $code -eq 1 ] && fail "preparar_subconjunto fallo (ver $logs/preparar.log)"
  say "preparar_subconjunto termino con codigo $code"
fi

say "tests"
"$py" -m pytest -q cpu/tests gpu/tests datos/tests > "$logs/tests.log" 2>&1 || fail "fallan los tests (ver $logs/tests.log)"

run_cpu() {
  local name="$1" prefix="$2"; shift 2
  [ -f "$res/$name/tiempos.json" ] && { say "ya existe $name"; return 0; }
  mkdir -p "$(dirname "$logs/$name.log")"
  say "cpu $name: $prefix $*"
  $prefix "$py" -m cpu.ejecutar --salida "$res/$name" "$@" > "$logs/$name.log" 2>&1 || fail "cpu $name"
}
run_gpu() {
  local name="$1" prefix="$2"; shift 2
  [ -f "$res/$name/tiempos.json" ] && { say "ya existe $name"; return 0; }
  mkdir -p "$(dirname "$logs/$name.log")"
  say "gpu $name: $prefix $*"
  $prefix "$py" -m gpu.run --salida "$res/$name" "$@" > "$logs/$name.log" 2>&1 || fail "gpu $name"
}

run_cpu pruebas/cpu_500 "" --limite 500 -r 1 --calentamiento 0 --hilos 8 --sin-pausas
run_gpu pruebas/gpu_500 "" --limite 500 -r 1 --sin-pausas
"$py" -m comun.comparar "$res/pruebas/cpu_500/cola.csv" "$res/pruebas/gpu_500/cola.csv" > "$res/pruebas/comparacion_500.txt" || fail "comparar 500"

pcores="taskset -c 0-15"
declare -A prefix=([todos]="" [nucleos_p]="$pcores")
for cfg in "todos 32" "todos 24" "todos 16" "todos 8" "nucleos_p 16" "nucleos_p 8"; do
  set -- $cfg
  run_cpu "barrido_cpu/${1}_$2" "${prefix[$1]}" --limite 2000 -r 1 --calentamiento 1 --hilos "$2" --sin-pausas
  sleep 90
done
best=$("$py" - "$res/barrido_cpu" <<'PY'
import json, sys
from pathlib import Path
rows = [(json.loads(p.read_text())["imagenes_por_s"], p.parent.name) for p in Path(sys.argv[1]).glob("*/tiempos.json")]
print(max(rows)[1])
PY
)
best_set=${best%_*}
best_threads=${best##*_}
best_prefix="${prefix[$best_set]}"
say "mejor configuracion de cpu: $best ($best_prefix --hilos $best_threads)"
echo "$best" > "$res/barrido_cpu/mejor.txt"

run_cpu barrido_cpu/pesos_originales "$best_prefix" --limite 2000 -r 1 --calentamiento 1 --hilos "$best_threads" --sin-pausas --pesos-originales
sleep 120

run_cpu cpu "$best_prefix" --hilos "$best_threads"
sleep 300

run_gpu gpu ""
for v in "solapado --modo solapado" "float32 --transferencia float32" "torch --preprocesado torch" \
         "paginable --memoria paginable" "lote_32 --lote 32" "lote_128 --lote 128"; do
  set -- $v; name=$1; shift
  sleep 60
  run_gpu "variantes_gpu/$name" "" "$@"
done
sleep 60
run_gpu variantes_gpu/hilos_cpu "$best_prefix" --hilos "$best_threads"
sleep 60
run_gpu variantes_gpu/pesos_originales "" --limite 2000 -r 1 --sin-pausas --pesos-originales
run_gpu variantes_gpu/pesos_limpios_2000 "" --limite 2000 -r 1 --sin-pausas

say "comparando y evaluando"
"$py" -m comun.comparar "$res/cpu/cola.csv" "$res/gpu/cola.csv" > "$res/comparacion_cpu_gpu.txt" 2>&1 || fail "comparar"
"$py" -m comun.evaluar "$res/cpu/cola.csv" > "$res/auc_cpu.txt" 2>&1 || fail "evaluar cpu"
"$py" -m comun.evaluar "$res/gpu/cola.csv" > "$res/auc_gpu.txt" 2>&1 || fail "evaluar gpu"
say "TODO LISTO"
