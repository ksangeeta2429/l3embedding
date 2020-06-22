#!/usr/bin/env bash

###### Retrain type: finetune ########
# With no hidden layer
sbatch classify_edgel3_mlp_ust.sbatch ft
sleep 3

# With one hidden layer
sbatch classify_edgel3_mlp_ust.sbatch ft 1 128
sleep 3
sbatch classify_edgel3_mlp_ust.sbatch ft 1 256
sleep 3

# With two hidden layers
sbatch classify_edgel3_mlp_ust.sbatch ft 2 128
sleep 3
sbatch classify_edgel3_mlp_ust.sbatch ft 2 256



###### Retrain type: knowledge distillation ########
# With no hidden layer
sbatch classify_edgel3_mlp_ust.sbatch kd
sleep 3

# With one hidden layer
sbatch classify_edgel3_mlp_ust.sbatch kd 1 128
sleep 3
sbatch classify_edgel3_mlp_ust.sbatch kd 1 256
sleep 3

# With two hidden layers
sbatch classify_edgel3_mlp_ust.sbatch kd 2 128
sleep 3
sbatch classify_edgel3_mlp_ust.sbatch kd 2 256
