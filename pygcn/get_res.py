import pandas as pd
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
alpha = [0.1, 0.5, 1.0, 10.0]
beta = [0.1, 0.5, 1.0, 10.0]
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
            if os.path.exists(url):
                with open(url, 'r') as f:
                    s = f.read()
                    sub1 = '"test_acc": '
                    sub2 = ', "test_duration"'
                    result = s[s.index(sub1) + len(sub1): s.index(sub2)]
                    acc_res.append(float(result))
        if len(acc_res) == 10:
            results_all[0].append(d)
            for i in range(len(config_str)):
                results_all[i+1].append(config[i])
            results_all[-1].append(sum(acc_res)/len(acc_res))
        else:
            print(url)


all_config_str = ['dataset', 'lr', 'dropout', 'early_stopping', 'weight_decay',
                  'alpha', 'beta', 'gamma', 'norm_layers', 'orders', 'accuracy']
d = {}
assert len(all_config_str) == len(results_all)
for i in range(len(all_config_str)):
    d[all_config_str[i]] = results_all[i]

df = pd.DataFrame.from_dict(d)
df = df.sort_values(['dataset', 'accuracy']).reset_index(drop=True)
df.to_csv('results/mlpnorm_result_all.csv')
