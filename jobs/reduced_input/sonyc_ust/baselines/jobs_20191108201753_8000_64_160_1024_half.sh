#!/usr/bin/env bash

# With no hidden layer
sbatch classify_l3_ust.sbatch 20191108201753_8000_64_160_1024_half
sleep 3

# With one hidden layer
sbatch classify_l3_ust.sbatch 20191108201753_8000_64_160_1024_half 1 128 
sleep 3
sbatch classify_l3_ust.sbatch 20191108201753_8000_64_160_1024_half 1 256
sleep 3

# With two hidden layers
sbatch classify_l3_ust.sbatch 20191108201753_8000_64_160_1024_half 2 128
sleep 3
sbatch classify_l3_ust.sbatch 20191108201753_8000_64_160_1024_half 2 256
