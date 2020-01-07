#!/usr/bin/env bash

# With no hidden layer
sbatch classify_mlp_l3_ust.sbatch 20191108201753_8000_64_160_1024_half_for_quantization_flattenOp
sleep 3

# With one hidden layer
sbatch classify_mlp_l3_ust.sbatch 20191108201753_8000_64_160_1024_half_for_quantization_flattenOp 1 128
sleep 3
sbatch classify_mlp_l3_ust.sbatch 20191108201753_8000_64_160_1024_half_for_quantization_flattenOp 1 256
sleep 3

# With two hidden layers
sbatch classify_mlp_l3_ust.sbatch 20191108201753_8000_64_160_1024_half_for_quantization_flattenOp 2 128
sleep 3
sbatch classify_mlp_l3_ust.sbatch 20191108201753_8000_64_160_1024_half_for_quantization_flattenOp 2 256


# Pipeline for Nathan
sbatch pipeline_mlp_l3_ust.sbatch 20191108201753_8000_64_160_1024_half_for_quantization_flattenOp 2 128