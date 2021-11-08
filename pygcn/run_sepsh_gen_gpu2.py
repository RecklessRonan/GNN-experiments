import pickle

dataset = 'squirrel'
datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']
for i in range(len(datasets)):
    if datasets[i] == dataset:
        data_id = i

best_config = {
    'chameleon': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'squirrel': [0.05, 0.0, 200, 0, 1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'cora': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'citeseer': [1.0, 10000.0, 0.5, 2, 2, 3, 3],
    'pubmed': [0.01, 0.2, 40, 0.00005, 1.0, 10000.0, 0.6, 2, 2, 3, 3],
    'texas': [10.0, 1.0, 0.5, 2, 2, 3, 3],
    'wisconsin': [1.0, 0.1, 0.9, 2, 2, 3, 3],
    'cornell': [0.5, 0.1, 0.1, 2, 2, 3, 3],
    'film': [0.01, 0.9, 40, 0.0005, 10.0, 10.0, 0.5, 2, 2, 3, 3]
}

best = best_config[dataset]
run_sh_all = ""
config_list = []

lr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
gamma = [i/10 for i in range(11)]
# print(gamma)
parameter = lr


for p in parameter:
    best[6] = p
    for s in range(10):
        run_sh = "python3 pygcn_raw.py --model mlp_norm --epochs 2000 --hidden 64" + \
            " --lr " + str(best[0]) + " --weight_decay " + str(best[3]) + \
            " --early_stopping " + str(best[2]) + \
            " --dropout " + str(best[1]) + " --alpha " + str(best[4]) + \
            " --beta " + str(best[5]) + ' --gamma ' + str(best[6]) + \
            " --norm_layers " + str(best[8]) + " --orders " + str(best[10]) + \
            " --orders_func_id " + str(best[9]) + " --norm_func_id " + str(best[7]) + \
            " --dataset " + dataset + " --split " + str(s)
        run_sh_all += run_sh + '\n'
    config = best.append(data_id)
    config_list.append(config)


with open('config_list3', 'wb') as f:
    pickle.dump(config_list, f)

with open('run3.sh', 'w') as f:
    f.write(run_sh_all)