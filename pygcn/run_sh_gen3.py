import itertools
import pickle

dataset1 = ['cora', 'citeseer', 'pubmed']
dataset2 = ['chameleon', 'squirrel']
dataset3 = ['cornell', 'texas', 'wisconsin']
datasets = ['chameleon', 'cornell', 'squirrel',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

# lr, early_stopping, alpha, beta, norm_layers, orders
best1 = [0.01, 200, 1.0, 10000.0, 2, 3]
# lr, early_stopping, alpha, beta, norm_layers, orders
best2 = [0.01, 200, 0.1, 0.1, 2, 3]
# lr, early_stopping, alpha, beta , norm_layers, orders
best3 = [0.05, 200, 1.0, 0.1, 2, 3]


weight_decay = [1e-7, 1e-6, 5e-5, 1e-4]
dropout = [0.2, 0.4, 0.6, 0.8]
gamma = [0.2, 0.4, 0.6, 0.8]


run_sh_all = ""
config_list = []
for w, d, g in itertools.product(weight_decay, dropout, gamma):
    for i in range(len(datasets)):
        if datasets[i] in dataset1:
            best = best1
        elif datasets[i] in dataset2:
            best = best2
        else:
            best = best3
        for s in range(10):
            run_sh = "python3 pygcn.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
                " --lr " + str(best[0]) + " --weight_decay " + str(w) + " --early_stopping " + str(best[1]) + \
                " --dropout " + str(d) + " --alpha " + str(best[2]) +\
                " --beta " + str(best[3]) + ' --gamma ' + str(g) + \
                " --norm_layers " + str(best[4]) + " --orders " + str(best[5]) + \
                " --dataset " + str(datasets[i]) + " --split " + str(s)
            run_sh_all += run_sh + '\n'
        config = [best[0], d, best[1], w, best[2],
                  best[3], g, best[4], best[5], i]
        config_list.append(config)

with open('config_list', 'wb') as f:
    pickle.dump(config_list, f)

with open('run.sh', 'w') as f:
    f.write(run_sh_all)
