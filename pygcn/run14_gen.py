import pickle
import itertools

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']


best_config = {
    'chameleon': [0.01, 0.0, 200, 0.0, 0.0, 10000000.0, 0.0, 2, 1, 1.0, 2, 2],
    'squirrel': [0.1, 0.0, 200, 0.0, 0.0, 100000000.0, 0.0, 1, 1, 1.0, 2, 2],
    'cora': [0.001, 0.6, 40, 0.00005, 0.0, 1000.0, 0.8, 2, 4, 1.0, 2, 2],
    'citeseer': [0.01, 0.7, 40, 0.00001, 0.0, 1000.0, 0.8, 2, 1, 1.0, 2, 2],
    'pubmed': [0.01, 0.3, 40, 0.00001, 0.0, 1000.0, 0.4, 1, 4, 1.0, 2, 2],
    'texas': [0.01, 0.0, 200, 0.00005, 0.0, 0.1, 0.1, 2, 3, 1.0, 2, 2],
    'wisconsin': [0.01, 0.0, 200, 0.00005, 0.0, 1.0, 0.3, 2, 3, 1.0, 2, 2],
    'cornell': [0.01, 0.0, 200, 0.00005, 0.0, 1.0, 0.6, 2, 1, 1.0, 2, 2],
    'film': [0.01, 0.0, 40, 0.001, 0.0, 1000.0, 0.2, 2, 6, 1.0, 2, 2]
}


run_sh_all = ""
config_list = []

delta = [i/10 for i in range(0, 11)]

for i in range(len(datasets)):
    dataset = datasets[i]
    best = best_config[dataset]
    for d in delta:
        best[9] = d
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
    config.append(i)
    config_list.append(config)

with open('config_list14', 'wb') as f:
    pickle.dump(config_list, f)

with open('run14.sh', 'w') as f:
    f.write(run_sh_all)
