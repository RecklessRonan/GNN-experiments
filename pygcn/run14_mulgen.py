import pickle
import itertools

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

dataset = 'squirrel'

for i in range(len(datasets)):
    if datasets[i] == dataset:
        dataset_id = i

best_config = {
    'chameleon': [0.01, 0.0, 200, 0.0, 0.0, 10000000.0, 0.0, 2, 1, 1.0, 2, 2],
    'squirrel': [0.1, 0.0, 200, 0.0, 0.0, 100000000.0, 0.0, 1, 1, 1.0, 2, 2],
    'cora': [0.01, 0.8, 40, 0.00005, 0.0, 800.0, 0.8, 2, 4, 0.9, 2, 2],
    'citeseer': [0.01, 0.7, 40, 0.00001, 0.0, 1000.0, 0.8, 2, 1, 1.0, 2, 2],
    'pubmed': [0.01, 0.6, 40, 0.0001, 0.0, 20000.0, 0.5, 1, 3, 1.0, 2, 2],
    'texas': [0.01, 0.0, 200, 0.00005, 0.0, 0.1, 0.1, 2, 3, 1.0, 2, 2],
    'wisconsin': [0.01, 0.0, 200, 0.00005, 0.0, 1.0, 0.3, 2, 3, 1.0, 2, 2],
    'cornell': [0.01, 0.0, 200, 0.00005, 0.0, 1.0, 0.6, 2, 1, 1.0, 2, 2],
    'film': [0.01, 0.0, 40, 0.001, 0.0, 10000.0, 0.2, 2, 6, 1.0, 2, 2]
}


run_sh_all = ""
config_list = []

lr = [0.01]
dropout = [0.0, 0.5]
beta = [100000000.0, 1.0]
alpha = [0.0]
gamma = [0.0, 0.5, 0.9]
weight_decay = [0.0]
orders = [1, 2, 3]
early_stopping = [200]
norm_layers = [1, 2, 3]
delta = [0.0, 0.5, 1.0]

best = best_config[dataset]
for d, b, g, w, o, e, n, de, a, l in itertools.product(dropout, beta, gamma, weight_decay, orders, early_stopping, norm_layers, delta, alpha, lr):
    best[1] = d
    best[5] = b
    best[6] = g
    best[3] = w
    best[8] = o
    best[2] = e
    best[7] = n
    best[9] = de
    best[4] = a
    best[0] = l
    for s in range(10):
        run_sh = "python3 pygcn_raw.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
            " --lr " + str(best[0]) + " --dropout " + str(best[1]) + " --early_stopping " + str(best[2]) + \
            " --weight_decay " + str(best[3]) + " --alpha " + str(best[4]) + " --beta " + str(best[5]) + \
            " --gamma " + str(best[6]) + " --delta " + str(best[9]) +\
            " --norm_layers " + str(best[7]) + " --orders " + str(best[8]) + \
            " --orders_func_id " + str(best[11]) + " --norm_func_id " + str(best[10]) + \
            " --dataset " + dataset + " --split " + str(s)
        run_sh_all += run_sh + '\n'

    config = best * 1
    config.append(dataset_id)
    config_list.append(config)

with open('config_list14', 'wb') as f:
    pickle.dump(config_list, f)

with open('run14.sh', 'w') as f:
    f.write(run_sh_all)
