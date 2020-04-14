#!/usr/bin/env bash

# With no hidden layer
sbatch classify_l3_ust.sbatch melSpec_20200304183233_48000_256_242_2048
sleep 3

# With one hidden layer
sbatch classify_l3_ust.sbatch melSpec_20200304183233_48000_256_242_2048 1 128 
sleep 3
sbatch classify_l3_ust.sbatch melSpec_20200304183233_48000_256_242_2048 1 256
sleep 3

# With two hidden layers
sbatch classify_l3_ust.sbatch melSpec_20200304183233_48000_256_242_2048 2 128
sleep 3
sbatch classify_l3_ust.sbatch melSpec_20200304183233_48000_256_242_2048 2 256
