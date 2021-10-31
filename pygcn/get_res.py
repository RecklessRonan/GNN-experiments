import os
import itertools

datasets = ['chameleon', 'cornell', 'film', 'squirrel',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

config_str = ['lr', 'do', 'es', 'wd', 'alpha', 'beta', 'gamma', 'nl', 'orders']

config_list = []
results_all = [[] for _ in range(len(config_str) + 2)]

lr = 0.01
do = 0.5
es = 40
wd = 0.0005
alpha = [0.1, 0.5, 1, 10]
beta = [0.1, 0.5, 1, 10]
gamma = [0.1, 0.5, 0.9]
nl = 2
orders = 3

for a, b, g in itertools.product(alpha, beta, gamma):
    config = [lr, do, es, wd, a, b, g, nl, orders]
    config_list.append(config)

for config in config_list:
    for d in datasets:
        acc_res = []
        for i in range(10):
            url_configs = ''
            for j in range(len(config_str)):
                url_configs += '_' + config_str[j] + str(config[j])
            url = 'runs/' + d + url_configs + \
                '_split' + str(i) + '_results.txt'
