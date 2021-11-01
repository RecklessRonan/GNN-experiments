import itertools

lr = [0.05, 0.01, 0.001]
early_stopping = [40, 200]
weight_decay = [0, 1e-5, 5e-4]
dropout = [0.1, 0.5, 0.9]
alpha = [0.1, 0.5, 1.0, 10.0]
beta = [0.1, 0.5, 1.0, 10.0]
gamma = [0.1, 0.5, 0.9]
norm_layers = [1, 2, 3]
orders = [1, 3, 5]
datasets = ['chameleon', 'cornell', 'film', 'squirrel',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']


best_abg = {
    'chameleon': [0.1, 0.1, 0.9],
    'cornell': [0.5, 0.1, 0.1],
    'film': [10.0, 10.0, 0.5],
    'squirrel': [0.1, 0.1, 0.1],
    'texas': [10.0, 1.0, 0.5],
    'wisconsin': [1.0, 0.1, 0.9],
    'pubmed': [10.0, 0.1, 0.9],
    'cora': [0.1, 0.1, 0.5],
    'citeseer': [1.0, 10.0, 0.5]
}


run_sh_all = ""
for l, e, w, d in itertools.product(lr, early_stopping, weight_decay, dropout):
    for data in datasets:
        for s in range(10):
            run_sh = "python3 pygcn.py --no-cuda --model mlp_norm --epochs 2000 --hidden 64" + \
                " --lr " + str(l) + " --weight_decay " + str(w) + " --early_stopping " + str(e) + \
                " --dropout " + str(d) + " --alpha " + str(best_abg[data][0]) +\
                " --beta " + str(best_abg[data][1]) + ' --gamma ' + str(best_abg[data][2]) + \
                " --norm_layers " + str(2) + " --orders " + str(3) + \
                " --dataset " + str(data) + " --split " + str(s)
            run_sh_all += run_sh + '\n'

with open('run.sh', 'w') as f:
    f.write(run_sh_all)
