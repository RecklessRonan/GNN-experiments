import itertools
import pickle

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

# alpha, beta, gamma, norm_layers, orders, order_func_id, norm_func_id
best_config = {
    'chameleon': [1.0, 10000.0, 0.9, 2, 3, 3, 2],
    'squirrel': [1.0, 10000.0, 0.9, 2, 3, 3, 2],
    'cora': [1.0, 10000.0, 0.9, 2, 3, 3, 2],
    'citeseer': [1.0, 10000.0, 0.5, 2, 3, 3, 2],
    'pubmed': [1.0, 10000.0, 0.9, 2, 3, 3, 2],
    'texas': [10.0, 1.0, 0.5, 2, 3, 3, 2],
    'wisconsin': [1.0, 0.1, 0.9, 2, 3, 3, 2],
    'cornell': [0.5, 0.1, 0.1, 2, 3, 3, 2],
    'film': [10.0, 10.0, 0.5, 2, 3, 3, 2]
}


weight_decay = [0, 1e-7, 1e-6, 5e-5, 1e-4, 5e-4]
dropout = [0.1, 0.3, 0.5, 0.7, 0.9]
lr = [0.1, 0.05, 0.01]
early_stopping = 200


run_sh_all = ""
config_list = []
for l, d, w in itertools.product(lr, dropout, weight_decay):
    for data in datasets:
        best = best_config[data]
        for s in range(10):
            run_sh = "python3 pygcn_raw.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
                " --lr " + str(l) + " --weight_decay " + str(w) + \
                " --early_stopping " + str(early_stopping) + \
                " --dropout " + str(d) + " --alpha " + str(best[0]) + \
                " --beta " + str(best[1]) + ' --gamma ' + str(best[2]) + \
                " --norm_layers " + str(best[3]) + " --orders " + str(best[4]) + \
                " --orders_func_id " + str(best[5]) + " --norm_func_id " + str(best[6]) + \
                " --dataset " + data + " --split " + str(s)
            run_sh_all += run_sh + '\n'
        config = [l, w, early_stopping, w].extend(best)
        config_list.append(config)

with open('config_list6', 'wb') as f:
    pickle.dump(config_list, f)

with open('run6.sh', 'w') as f:
    f.write(run_sh_all)
