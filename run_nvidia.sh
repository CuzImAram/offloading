#!/bin/bash

# Skript bricht ab, falls ein Befehl fehlschlägt
set -e

# Anzahl der Seams aus Argument $1 lesen, sonst Default 500
SEAMS=${1:-500}

# Definition der Compiler-Flags für OpenMP Offloading (ROCm)
OMP_FLAGS="/opt/llvm-with-offloading/bin/clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_86 --libomptarget-nvptx-bc-path=/opt/llvm-with-offloading/lib/nvptx64-nvidia-cuda/ -Wl,-rpath,/opt/llvm-with-offloading/lib/x86_64-unknown-linux-gnu -O3 -g"

echo "--------------------------------------------------"
echo "Starte Kompilierung..."

# 1. main_correct
# echo "[1/4] Kompiliere main_correct..."
# clang $OMP_FLAGS main_correct.c image.c -o main_correct -lm -ljpeg

# 2. main_fast
echo "[2/4] Kompiliere main_fast..."
$OMP_FLAGS main_fast.c image.c -o main_fast -lm -ljpeg

# 3. main_seq
echo "[3/4] Kompiliere main_seq..."
clang -O3 main_seq.c image.c -o main_seq -lm -ljpeg

# 4. main_raw
echo "[4/4] Kompiliere main_raw..."
clang -O3 main_raw.c image.c -o main_raw -lm -ljpeg

echo "Kompilierung erfolgreich abgeschlossen."
echo "--------------------------------------------------"
echo "Starte Benchmarks (Bild: images/tower.jpg, Seams: $SEAMS)"
echo "--------------------------------------------------"

# Ausführung
# echo ">>> Running main_correct"
# ./main_correct images/tower.jpg images/tower_main_correct.jpg $SEAMS

echo ">>> Running main_fast"
./main_fast images/tower.jpg images/tower_main_fast.jpg $SEAMS

echo ">>> Running main_seq"
./main_seq images/tower.jpg images/tower_main_seq.jpg $SEAMS

echo ">>> Running main_raw"
./main_raw images/tower.jpg images/tower_main_raw.jpg $SEAMS

# echo ">>> Running main_correct"
# ./main_correct images/moon.jpg images/moon_main_correct.jpg $SEAMS

echo ">>> Running main_fast"
./main_fast images/moon.jpg images/moon_main_fast.jpg $SEAMS

echo ">>> Running main_seq"
./main_seq images/moon.jpg images/moon_main_seq.jpg $SEAMS

echo ">>> Running main_raw"
./main_raw images/moon.jpg images/moon_main_raw.jpg $SEAMS

echo "--------------------------------------------------"
echo "Fertig."
