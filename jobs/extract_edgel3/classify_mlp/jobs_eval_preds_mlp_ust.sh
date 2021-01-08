#!/usr/bin/env bash

###### Retrain type: finetune ########
# With no hidden layer
sbatch evaluate_predictions_mlp_ust.sbatch ft 0 0
sleep 3

# With one hidden layer
sbatch evaluate_predictions_mlp_ust.sbatch ft 1 128
sleep 3
sbatch evaluate_predictions_mlp_ust.sbatch ft 1 256
sleep 3

# With two hidden layers
sbatch evaluate_predictions_mlp_ust.sbatch ft 2 128
sleep 3
sbatch evaluate_predictions_mlp_ust.sbatch ft 2 256
sleep 3


###### Retrain type: knowledge distillation ########
# With no hidden layer
sbatch evaluate_predictions_mlp_ust.sbatch kd 0 0
sleep 3

# With one hidden layer
sbatch evaluate_predictions_mlp_ust.sbatch kd 1 128
sleep 3
sbatch evaluate_predictions_mlp_ust.sbatch kd 1 256
sleep 3

# With two hidden layers
sbatch evaluate_predictions_mlp_ust.sbatch kd 2 128
sleep 3
sbatch evaluate_predictions_mlp_ust.sbatch kd 2 256