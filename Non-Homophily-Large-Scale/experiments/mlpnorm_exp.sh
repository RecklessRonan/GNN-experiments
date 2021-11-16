#!/bin/bash

dataset=$1
sub_dataset=${2:-''}

hidden_channels_lst=(64)
beta_lst=(0.1)
gamma_lst=(0.5)
norm_layers_lst=(2)
dropout_lst=(0.4)
weight_decay_lst=(0.001)

for hidden_channels in "${hidden_channels_lst[@]}"; do
    for beta in "${beta_lst[@]}"; do
        for gamma in "${gamma_lst[@]}"; do
            for norm_layers in "${norm_layers_lst[@]}"; do
                for dropout in "${dropout_lst[@]}"; do
                    for weight_decay in "${weight_decay_lst[@]}"; do
                            if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --hidden_channels $hidden_channels --lr 0.01 --dropout $dropout --weight_decay $weight_decay --alpha 0.1 --beta $beta --gamma $gamma --norm_func_id 2 --norm_layers $norm_layers --orders_func_id 2 --orders 1 --display_step 25 --runs 5 --directed
                            else
                                python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --hidden_channels $hidden_channels --lr 0.01 --dropout $dropout --weight_decay $weight_decay --alpha 0.1 --beta $beta --gamma $gamma --norm_func_id 2 --norm_layers $norm_layers --orders_func_id 2 --orders 1 --display_step 25 --runs 5
                            fi
                    done
                done
            done
        done
    done 
done