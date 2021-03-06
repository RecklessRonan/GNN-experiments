import pickle
import itertools

dataset = 'wisconsin'

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

for i in range(len(datasets)):
    if datasets[i] == dataset:
        data_id = i

best_config = {
    'chameleon': [0.05, 0.0, 250, 0.0, 1.0, 1000000.0, 0.1, 2, 2, 3, 3],
    'squirrel': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'cora': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'citeseer': [1.0, 10000.0, 0.5, 2, 2, 3, 3],
    'pubmed': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'texas': [10.0, 1.0, 0.5, 2, 2, 3, 3],
    'wisconsin': [0.05, 0.1, 40, 0.0001, 10.0, 0.05, 0.9, 2, 2, 3, 3],
    'cornell': [0.5, 0.1, 0.1, 2, 2, 3, 3],
    'film': [0.001, 0.9, 40, 0.0005, 10.0, 10.0, 0.5, 2, 2, 3, 3]
}

best = best_config[dataset]
run_sh_all = ""
config_list = []

lr = [0.01, 0.05, 0.1]
weight_decay = [0.0, 0.0000001, 0.0000005, 0.000001,
                0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
dropout = [i/10 for i in range(10)]
early_stopping = [40, 100, 200, 250]
alpha = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 1000000.0, 100000000.0]
beta = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 10000.0, 1000000.0, 100000000.0]
gamma = [i/10 for i in range(1, 10)]
orders = [1, 2, 3, 4, 5]
# lr 0, dropout 1, early_stopping 2, weight_decay 3, alpha 4, beta 5, gamma 6, orders 10

alpha = [0.0, 0.01, 0.05, 0.1, 1.0, 10.0]
beta = [0.0, 0.01, 0.05, 0.1, 1.0, 10.0]
dropout = [i/10 for i in range(6)]
weight_decay = [0.00005, 0.0001, 0.0005]
gamma = [i/100 for i in range(85, 95)]

for l, e in itertools.product(lr, early_stopping):
    best[0] = l
    best[2] = e

    # if a+b == 0.0:
    #     continue
    for s in range(10):
        run_sh = "python3 pygcn_raw.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
            " --lr " + str(best[0]) + " --weight_decay " + str(best[3]) + \
            " --early_stopping " + str(best[2]) + \
            " --dropout " + str(best[1]) + " --alpha " + str(best[4]) + \
            " --beta " + str(best[5]) + ' --gamma ' + str(best[6]) + \
            " --norm_layers " + str(best[8]) + " --orders " + str(best[10]) + \
            " --orders_func_id " + str(best[9]) + " --norm_func_id " + str(best[7]) + \
            " --dataset " + dataset + " --split " + str(s)
        run_sh_all += run_sh + '\n'
    config = best * 1
    config.append(data_id)
    config_list.append(config)


with open('config_list11', 'wb') as f:
    pickle.dump(config_list, f)

with open('run11.sh', 'w') as f:
    f.write(run_sh_all)
