import pickle
import itertools

dataset = 'texas'
# dataset = 'chameleon'

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

for i in range(len(datasets)):
    if datasets[i] == dataset:
        data_id = i

best_config = {
    'chameleon': [0.05, 0.3, 250, 0.00005, 1.0, 100000000.0, 0.2, 2, 2, 3, 3],
    'squirrel': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'cora': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'citeseer': [1.0, 10000.0, 0.5, 2, 2, 3, 3],
    'pubmed': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'texas': [0.05, 0.0, 40, 0.0001, 10000.0, 10.0, 0.8, 2, 2, 3, 3],
    'wisconsin': [1.0, 0.1, 0.9, 2, 2, 3, 3],
    'cornell': [0.5, 0.1, 0.1, 2, 2, 3, 3],
    'film': [0.001, 0.9, 40, 0.0005, 10.0, 10.0, 0.5, 2, 2, 3, 3]
}

best = best_config[dataset]
run_sh_all = ""
config_list = []

lr = [0.01, 0.05, 0.1]
weight_decay = [0.00005, 0.0001, 0.0005]
dropout = [0.1, 0.3, 0.5]
early_stopping = [40, 100, 200]
gamma = [0.2, 0.5, 0.8]
orders = [1, 3, 5]


# for a, b in itertools.product(alpha, beta):
#     best[4] = a
#     best[5] = b

#     if a+b == 0.0:
#         continue

for l, d, e, w, g, o in itertools.product(lr, dropout, early_stopping, weight_decay, gamma, orders):
    best[0] = l
    best[1] = d
    best[2] = e
    best[3] = w
    best[6] = g
    best[10] = o

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


# # lr 0, dropout 1, early_stopping 2, weight_decay 3, alpha 4, beta 5, gamma 6, orders 10
# parameter = orders
# pos = 10
# for p in parameter:
#     best[pos] = p
#     for s in range(10):
#         run_sh = "python3 pygcn_raw.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
#             " --lr " + str(best[0]) + " --weight_decay " + str(best[3]) + \
#             " --early_stopping " + str(best[2]) + \
#             " --dropout " + str(best[1]) + " --alpha " + str(best[4]) + \
#             " --beta " + str(best[5]) + ' --gamma ' + str(best[6]) + \
#             " --norm_layers " + str(best[8]) + " --orders " + str(best[10]) + \
#             " --orders_func_id " + str(best[9]) + " --norm_func_id " + str(best[7]) + \
#             " --dataset " + dataset + " --split " + str(s)
#         run_sh_all += run_sh + '\n'
#     config = best * 1
#     config.append(data_id)
#     config_list.append(config)


with open('config_list1', 'wb') as f:
    pickle.dump(config_list, f)

with open('run1.sh', 'w') as f:
    f.write(run_sh_all)
