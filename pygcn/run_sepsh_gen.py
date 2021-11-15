import pickle
import itertools

# dataset = 'texas'
# dataset = 'chameleon'
# dataset = 'film'
# dataset = 'cora'
# dataset = 'texas'
# dataset = 'film'
# dataset = 'citeseer'
dataset = 'pubmed'
# dataset = 'squirrel'
# dataset = 'wisconsin'

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
#     'wisconsin': [0.05, 0.1, 40, 0.0001, 10.0, 0.05, 0.9, 1, 2, 3, 3],
#     'cornell': [0.05, 0.0, 40, 0.00005, 1.0, 0.1, 0.6, 1, 2, 3, 3],
#     'film': [0.001, 0.0, 40, 0.001, 10.0, 1.0, 10.0, 1, 2, 3, 4]
# }

best_config = {
    'chameleon': [0.01, 0.3, 250, 0.00005, 10.0, 10000000.0, 0.1, 2, 2, 3, 3],
    'squirrel': [0.05, 0.1, 250, 0.00001, 0.0, 1000000.0, 0.1, 2, 2, 3, 3],
    'cora': [0.01, 0.6, 40, 0.00005, 1000000.0, 10000.0, 0.7, 2, 2, 3, 4],
    'citeseer': [0.01, 0.5, 40, 0.0000001, 110.0, 70.0, 0.9, 2, 2, 3, 3],
    'pubmed': [0.01, 0.2, 40, 0.00005, 100000000.0, 100000000.0, 0.5, 2, 2, 3, 1],
    'texas': [0.01, 0.0, 200, 0.0001, 1.0, 0.1, 0.3, 2, 2, 3, 3],
    'wisconsin': [0.01, 0.0, 200, 0.00005, 1.0, 0.1, 0.5, 2, 2, 3, 3],
    'cornell': [0.05, 0.0, 40, 0.00005, 0.03, 0.08, 0.6, 2, 2, 3, 3],
    'film': [0.001, 0.0, 40, 0.001, 0.1, 0.1, 0.2, 2, 2, 3, 2]
}

best = best_config[dataset]
run_sh_all = ""
config_list = []

lr = [0.05]
weight_decay = [0.00001]
dropout = [0.1, 0.0]
early_stopping = [250]
gamma = [0.1, 0.0]
alpha = [0.0, 1000000.0]
beta = [1000000.0, 10000000.0]


# for l, a, b, e, g, w, d in itertools.product(lr, alpha, beta, early_stopping, gamma, weight_decay, dropout):
#     best[0] = l
#     best[1] = d
#     best[4] = a
#     best[5] = b
#     best[6] = g
#     best[2] = e
#     best[3] = w

#     if a+b == 0.0:
#         continue

# for l, d, e, w, g, o in itertools.product(lr, dropout, early_stopping, weight_decay, gamma, orders):
#     best[0] = l
#     best[1] = d
#     best[2] = e
#     best[3] = w
#     best[6] = g
#     best[10] = o

# for s in range(10):
#     run_sh = "python3 pygcn.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
#         " --lr " + str(best[0]) + " --weight_decay " + str(best[3]) + \
#         " --early_stopping " + str(best[2]) + \
#         " --dropout " + str(best[1]) + " --alpha " + str(best[4]) + \
#         " --beta " + str(best[5]) + ' --gamma ' + str(best[6]) + \
#         " --norm_layers " + str(best[8]) + " --orders " + str(best[10]) + \
#         " --orders_func_id " + str(best[9]) + " --norm_func_id " + str(best[7]) + \
#         " --dataset " + dataset + " --split " + str(s)
#     run_sh_all += run_sh + '\n'
# config = best * 1
# config.append(data_id)
# config_list.append(config)


# # lr 0, dropout 1, early_stopping 2, weight_decay 3, alpha 4, beta 5, gamma 6, norm_layers 8, orders 10
# parameter = norm_layers
# pos = 8
# for p in parameter:
#     best[pos] = p
for s in range(10):
    run_sh = "python3 pygcn.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
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


with open('config_list1', 'wb') as f:
    pickle.dump(config_list, f)

with open('run1.sh', 'w') as f:
    f.write(run_sh_all)
