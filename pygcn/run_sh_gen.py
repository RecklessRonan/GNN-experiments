import itertools

lr = [0.1, 0.05, 0.01, 0.001]
early_stopping = [40, 100, 200]
weight_decay = [0, 5e-6, 1e-5, 5e-5, 5e-4]
dropout = [0, 0.3, 0.5, 0.7, 0.9]
alpha = [0.1, 0.5, 1.0, 10.0]
beta = [0.1, 0.5, 1.0, 10.0]
gamma = [0.1, 0.5, 0.9]
norm_layers = [1, 2, 3]
orders = [1, 3, 5]
datasets = ['chameleon', 'cornell', 'film', 'squirrel',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

run_sh_all = ""
for a, b, g in itertools.product(alpha, beta, gamma):
    for d in datasets:
        for s in range(10):
            run_sh = "python3 pygcn.py --no-cuda --model mlp_norm --epochs 2000 --lr 0.01 " +\
                "--weight_decay 5e-4 --hidden 64 --early_stopping 40 " + \
                " --alpha " + str(a) + " --beta " + str(b) + ' --gamma ' + str(g) + \
                " --norm_layers " + str(2) + " --orders " + str(3) + \
                " --dataset " + str(d) + " --split " + str(s)
            run_sh_all += run_sh + '\n'

with open('run.sh', 'w') as f:
    f.write(run_sh_all)
