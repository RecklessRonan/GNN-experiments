import pickle

dataset = 'film'

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

best_config = {
    'chameleon': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'squirrel': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'cora': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'citeseer': [1.0, 10000.0, 0.5, 2, 2, 3, 3],
    'pubmed': [1.0, 10000.0, 0.9, 2, 2, 3, 3],
    'texas': [10.0, 1.0, 0.5, 2, 2, 3, 3],
    'wisconsin': [1.0, 0.1, 0.9, 2, 2, 3, 3],
    'cornell': [0.5, 0.1, 0.1, 2, 2, 3, 3],
    'film': [0.01, 0.9, 40, 0.0005, 10.0, 10.0, 0.5, 2, 2, 3, 3]
}

best = best_config[dataset]
run_sh_all = ""
config_list = []

lr = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


for l in lr:
    best[0] = l
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
    config = best.append(3)
    config_list.append(config)


with open('config_list_film', 'wb') as f:
    pickle.dump(config_list, f)

with open('run_film.sh', 'w') as f:
    f.write(run_sh_all)
