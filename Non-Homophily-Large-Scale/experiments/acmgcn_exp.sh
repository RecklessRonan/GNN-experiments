#!/bin/bash

# dataset_lst=(pokec snap-patents arxiv-year genius fb100 twitch-gamer)
# hidden_channel_lst=(80 64 32 16 8)
# weight_decay_lst=(1e-7 1e-2)
# dropout_lst=(0.0 0.7)
# decay_rate_lst=(0.0 1.5)
# num_layers_lst=(3 2 1)

dataset_lst=(arxiv-year)
hidden_channel_lst=(64)
weight_decay_lst=(0.0 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2)
dropout_lst=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
lr_lst=(0.01 0.05 0.1)
num_layers_lst=(2 1)
display_step=25

for dataset in "${dataset_lst[@]}"; do
    for lr in "${lr_lst[@]}"; do
        for num_layers in "${num_layers_lst[@]}"; do
            for weight_decay in "${weight_decay_lst[@]}"; do
                for dropout in "${dropout_lst[@]}"; do
                    for hidden_channel in "${hidden_channel_lst[@]}"; do
                        if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                python main.py --dataset $dataset --sub_dataset None --method acmgcn --lr $lr \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay  \
                                --display_step $display_step --runs 5 --directed
                        else
                            if [ "$dataset" = "fb100" ]; then
                                python main.py --dataset $dataset --sub_dataset Penn94 --method acmgcn --lr $lr \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay  \
                                --display_step $display_step --runs 5
                            else
                                python main.py --dataset $dataset --sub_dataset None --method acmgcn --lr $lr \
                                --num_layers $num_layers --hidden_channels $hidden_channel --dropout $dropout \
                                --weight_decay $weight_decay  \
                                --display_step $display_step --runs 5
                            fi
                        fi
                    done
                done
            done
        done
    done
done