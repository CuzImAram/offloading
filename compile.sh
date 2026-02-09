#!/bin/bash

# Skript bricht ab, falls ein Befehl fehlschlägt
set -e

/opt/llvm-with-offloading/bin/clang -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_86 --libomptarget-nvptx-bc-path=/opt/llvm-with-offloading/lib/nvptx64-nvidia-cuda/ -Wl,-rpath,/opt/llvm-with-offloading/lib/x86_64-unknown-linux-gnu -O3 -g enlarge.c image.c -o enlarge -lm -ljpeg

./enlarge images/tower.jpg images/tower_enlarge.jpg 500

clang -O3 enlarge_raw.c image.c -o enlarge_raw -lm -ljpeg

./enlarge_raw images/tower.jpg images/tower_enlarge_raw.jpg 500