import pickle
import itertools

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']


best_config12 = {
    'chameleon': [0.01, 0.0, 200, 0.00005, 0.0, 1.0, 0.0, 2, 1, 0.0, 1, 2],
    'squirrel': [0.01, 0.8, 200, 0.00005, 0.0, 1.0, 0.0, 2, 1, 1.0, 1, 2],
    'cora': [0.01, 0.8, 40, 0.00001, 77.0, 1000.0, 0.8, 2, 6, 0.9, 1, 2],
    'citeseer': [0.005, 0.9, 40, 0.00005, 0.0, 1000.0, 0.8, 3, 2, 1.0, 1, 2],
    'pubmed': [0.01, 0.5, 40, 0.00005, 0.0, 2000.0, 0.8, 1, 4, 1.0, 1, 2],
    'texas': [0.01, 0.0, 200, 0.00001, 10.0, 0.01, 0.0, 1, 2, 1.0, 1, 2],
    'wisconsin': [0.01, 0.0, 200, 0.00005, 0.5, 0.5, 0.0, 3, 5, 1.0, 1, 2],
    'cornell': [0.01, 0.0, 200, 0.00005, 0.0, 1.0, 0.0, 2, 1, 1.0, 1, 2],
    'film': [0.01, 0.0, 40, 0.001, 0.0, 1000.0, 0.1, 2, 6, 1.0, 1, 2]
}


best_config22 = {
    'chameleon': [0.01, 0.4, 300, 0.0001, 1.0, 1.0, 0.4, 3, 2, 0.0, 2, 2],
    'squirrel': [0.01, 0.8, 200, 0.0, 0.0, 1.0, 0.0, 3, 2, 0.0, 2, 2],
    'cora': [0.01, 0.8, 40, 0.00005, 0.0, 800.0, 0.8, 2, 4, 0.9, 2, 2],
    'citeseer': [0.01, 0.8, 40, 0.00001, 1.0, 1000.0, 0.8, 2, 3, 1.0, 2, 2],
    'pubmed': [0.01, 0.6, 200, 0.0001, 0.0, 20000.0, 0.5, 1, 3, 1.0, 2, 2],
    'texas': [0.01, 0.0, 200, 0.0001, 10.0, 0.1, 0.2, 1, 4, 1.0, 2, 2],
    'wisconsin': [0.01, 0.0, 200, 0.00005, 1.2, 0.05, 0.3, 2, 3, 1.0, 2, 2],
    'cornell': [0.01, 0.0, 200, 0.0001, 1.0, 0.1, 0.7, 2, 2, 1.0, 2, 2],
    'film': [0.01, 0.0, 40, 0.001, 0.0, 15000.0, 0.2, 2, 4, 1.0, 2, 2]
}


run_sh_all = ""
config_list = []


def get_sh(best, dataset, s):
    run_sh = "python3 pygcn_raw.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
        " --lr " + str(best[0]) + " --dropout " + str(best[1]) + " --early_stopping " + str(best[2]) + \
        " --weight_decay " + str(best[3]) + " --alpha " + str(best[4]) + " --beta " + str(best[5]) + \
        " --gamma " + str(best[6]) + " --delta " + str(best[9]) +\
        " --norm_layers " + str(best[7]) + " --orders " + str(best[8]) + \
        " --orders_func_id " + str(best[11]) + " --norm_func_id " + str(best[10]) + \
        " --dataset " + dataset + " --split " + str(s)
    run_sh += '\n'
    return run_sh


for i in range(len(datasets)):
    dataset = datasets[i]
    best = best_config22[dataset]
    for s in range(10):
        run_sh_all += get_sh(best, dataset, s)
    config = best * 1
    config.append(i)
    config_list.append(config)

    best1 = best * 1
    best1[5] = 0.0001
    for s in range(10):
        run_sh_all += get_sh(best1, dataset, s)
    config = best1 * 1
    config.append(i)
    config_list.append(config)

    best2 = best * 1
    if best2[9] != 0.0:
        best2[9] = 0.0
        for s in range(10):
            run_sh_all += get_sh(best2, dataset, s)
        config = best2 * 1
        config.append(i)
        config_list.append(config)

    best3 = best * 1
    if best3[9] != 1.0:
        best3[9] = 1.0
        for s in range(10):
            run_sh_all += get_sh(best3, dataset, s)
        config = best3 * 1
        config.append(i)
        config_list.append(config)


with open('ablation22', 'wb') as f:
    pickle.dump(config_list, f)

with open('run_ablation22.sh', 'w') as f:
    f.write(run_sh_all)
