import pickle
import itertools

# wisconsin needs to be updated from the excel
# dataset = 'wisconsin'
# dataset = 'chameleon'
dataset = 'pubmed'

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

for i in range(len(datasets)):
    if datasets[i] == dataset:
        data_id = i

# best_config = {
#     'chameleon': [0.01, 0.3, 250, 0.00005, 1.0, 100000000.0, 0.1, 1, 2, 3, 3],
#     'squirrel': [0.05, 0.1, 250, 0.00001, 0.0, 1000000.0, 0.1, 1, 2, 3, 3],
#     'cora': [0.01, 0.6, 40, 0.00001, 1000000.0, 10000.0, 0.7, 1, 2, 3, 4],
#     'citeseer': [0.01, 0.6, 40, 0.0000001, 100.0, 100.0, 0.9, 1, 2, 3, 5],
#     'pubmed': [0.01, 0.2, 40, 0.00005, 1000000.0, 1000000.0, 0.5, 1, 2, 3, 1],
#     'texas': [0.1, 0.1, 200, 0.0001, 10000.0, 10.0, 0.8, 1, 2, 3, 3],
#     'wisconsin': [0.01, 0.0, 200, 0.00001, 0.05, 0.05, 0.6, 1, 2, 3, 3],
#     'cornell': [0.05, 0.0, 40, 0.00005, 1.0, 0.1, 0.6, 1, 2, 3, 3],
#     'film': [0.001, 0.0, 40, 0.001, 10.0, 10.0, 0.2, 1, 2, 3, 3]
# }

best_config = {
    'chameleon': [0.01, 0.3, 250, 0.00005, 10.0, 10000000.0, 0.1, 2, 2, 3, 3],
    'squirrel': [0.05, 0.1, 250, 0.00001, 0.0, 1000000.0, 0.1, 2, 2, 3, 3],
    'cora': [0.01, 0.6, 40, 0.00001, 1000000.0, 10000.0, 0.7, 2, 2, 3, 4],
    'citeseer': [0.01, 0.6, 40, 0.0000001, 100.0, 100.0, 0.9, 2, 2, 3, 5],
    'pubmed': [0.01, 0.2, 40, 0.00005, 100000000.0, 100000000.0, 0.5, 2, 2, 3, 1],
    'texas': [0.1, 0.1, 200, 0.0001, 10000.0, 10.0, 0.8, 2, 2, 3, 3],
    'wisconsin': [0.05, 0.1, 40, 0.0001, 10.0, 0.05, 0.9, 2, 2, 3, 3],
    'cornell': [0.05, 0.0, 40, 0.00005, 1.0, 0.1, 0.6, 2, 2, 3, 3],
    'film': [0.001, 0.0, 40, 0.001, 10.0, 10.0, 0.2, 2, 2, 3, 3]
}

best = best_config[dataset]
run_sh_all = ""
config_list = []

lr = [0.001, 0.01, 0.05]
weight_decay = [0.00005]
dropout = [0.2]
early_stopping = [40, 100, 200]
alpha = [100000000.0]
beta = [100000000.0]
gamma = [0.5]
orders = [1, 2, 3, 4, 5]
norm_layers = [1, 2, 3, 4, 5]


for l, e, d, w, a, b, g in itertools.product(lr, early_stopping, dropout, weight_decay, alpha, beta, gamma):
    best[0] = l
    best[1] = d
    best[2] = e
    best[3] = w
    best[6] = g
    best[4] = a
    best[5] = b

# for l, e in itertools.product(lr, early_stopping):
#     best[0] = l
#     best[2] = e
# for a, b in itertools.product(alpha, beta):
#     best[4] = a
#     best[5] = b
#     if a+b <= 0.0:
#         continue
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


# lr 0, dropout 1, early_stopping 2, weight_decay 3, alpha 4, beta 5, gamma 6, norm_layers 8, orders 10
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


with open('config_list11', 'wb') as f:
    pickle.dump(config_list, f)

with open('run11.sh', 'w') as f:
    f.write(run_sh_all)
