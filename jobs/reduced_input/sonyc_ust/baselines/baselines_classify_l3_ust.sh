#!/usr/bin/env bash

# With no hidden layer
sbatch classify_l3_ust.sbatch 20190723135600_16000_128_128_512_half
sleep 1
sbatch classify_l3_ust.sbatch 20190723135620_8000_128_64_256_half_fmax_3500
sleep 1
sbatch classify_l3_ust.sbatch 20190710131207_16000_64_160_1024

# With one hidden layer
sbatch classify_l3_ust.sbatch 20190723135600_16000_128_128_512_half 1 128
sleep 1
sbatch classify_l3_ust.sbatch 20190723135620_8000_128_64_256_half_fmax_3500 1 128
sleep 1
sbatch classify_l3_ust.sbatch 20190710131207_16000_64_160_1024 1 128
sleep 1

sbatch classify_l3_ust.sbatch 20190723135600_16000_128_128_512_half 1 256
sleep 1
sbatch classify_l3_ust.sbatch 20190723135620_8000_128_64_256_half_fmax_3500 1 256
sleep 1
sbatch classify_l3_ust.sbatch 20190710131207_16000_64_160_1024 1 256
sleep 1

# With two hidden layers
sbatch classify_l3_ust.sbatch 20190723135600_16000_128_128_512_half 2 128
sleep 1
sbatch classify_l3_ust.sbatch 20190723135620_8000_128_64_256_half_fmax_3500 2 128
sleep 1
sbatch classify_l3_ust.sbatch 20190710131207_16000_64_160_1024 2 128
sleep 1

sbatch classify_l3_ust.sbatch 20190723135600_16000_128_128_512_half 2 256
sleep 1
sbatch classify_l3_ust.sbatch 20190723135620_8000_128_64_256_half_fmax_3500 2 256
sleep 1
sbatch classify_l3_ust.sbatch 20190710131207_16000_64_160_1024 2 256
sleep 1