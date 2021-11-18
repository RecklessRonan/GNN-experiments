#!/bin/bash

startTime=`date +%Y%m%d-%H:%M`
startTime_s=`date +%s`

dataset=$1
sub_dataset=${2:-''}

lr_lst=(0.005)
hidden_channels_lst=(256)
dropout_lst=(0 0.1 0.2)
weight_decay_lst=(0 1e-7 5e-6 1e-6 5e-5 1e-5 5e-4 1e-4 5e-3 1e-3)
alpha_lst=(1.0)
beta_lst=(10.0)
gamma_lst=(0.9)
norm_layers_lst=(2)
orders_lst=(2)
epochs=2000
runs=1
norm_func_id=2
order_func_id=2


for lr in "${lr_lst[@]}"; do
    for hidden_channels in "${hidden_channels_lst[@]}"; do
        for beta in "${beta_lst[@]}"; do
            for gamma in "${gamma_lst[@]}"; do
                for norm_layers in "${norm_layers_lst[@]}"; do
                    for dropout in "${dropout_lst[@]}"; do
                        for weight_decay in "${weight_decay_lst[@]}"; do
                            for orders in "${orders_lst[@]}"; do
                                for alpha in "${alpha_lst[@]}"; do
                                    if [ "$dataset" = "snap-patents" ] || [ "$dataset" = "arxiv-year" ]; then
                                        python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --epochs $epochs --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 25 --runs $runs --directed
                                    else
                                        python -u main.py --dataset $dataset --sub_dataset ${sub_dataset:-''} --method mlpnorm --epochs $epochs --hidden_channels $hidden_channels --lr $lr --dropout $dropout --weight_decay $weight_decay --alpha $alpha --beta $beta --gamma $gamma --norm_func_id $norm_func_id --norm_layers $norm_layers --orders_func_id $order_func_id --orders $orders --display_step 25 --runs $runs
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        done
    done 
done


endTime=`date +%Y%m%d-%H:%M`
endTime_s=`date +%s`
sumTime=$[ $endTime_s - $startTime_s ]
echo "$startTime ---> $endTime" "Totl:$sumTime seconds" 