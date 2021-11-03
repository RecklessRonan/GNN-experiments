import pandas as pd
import os
import itertools

datasets = ['chameleon', 'cornell', 'film', 'squirrel',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']

config_str = ['lr', 'do', 'es', 'wd', 'alpha', 'beta', 'gamma', 'nl', 'orders']

config_list = []
results_all = [[] for _ in range(len(config_str) + 2)]

# lr = 0.01
# do = 0.5
# es = 40
# wd = 0.0005
# alpha = [0.1, 0.5, 1.0, 10.0]
# beta = [0.1, 0.5, 1.0, 10.0]
# gamma = [0.1, 0.5, 0.9]
# nl = 2
# orders = 3

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


for data in datasets:
    for l, e, w, d in itertools.product(lr, early_stopping, weight_decay, dropout):
        config = [l, d, e, w, best_abg[data][0],
                  best_abg[data][1], best_abg[data][2], 2, 3, data]
        config_list.append(config)

non_para = 0
exist_para = 0

for config in config_list:
    # for d in datasets:
    acc_res = []
    for i in range(10):
        url_configs = ''
        for j in range(len(config_str)):
            url_configs += '_' + config_str[j] + str(config[j])
        url = 'runs/' + config[-1] + url_configs + \
            '_split' + str(i) + '_results.txt'
        if os.path.exists(url):
            with open(url, 'r') as f:
                s = f.read()
                sub1 = '"test_acc": '
                sub2 = ', "test_duration"'
                result = s[s.index(sub1) + len(sub1): s.index(sub2)]
                acc_res.append(float(result))
    # print(len(acc_res))
    if len(acc_res) == 10:
        exist_para += 1
        results_all[0].append(config[-1])
        for i in range(len(config_str)):
            results_all[i+1].append(config[i])
        results_all[-1].append(sum(acc_res)/len(acc_res))
    else:
        non_para += 1
        # print(url)

print('the number of all parameters', len(config_list))
print('the number of non-existing parameters', non_para)
print('the number of existing parameters', exist_para)


all_config_str = ['dataset', 'lr', 'dropout', 'early_stopping', 'weight_decay',
                  'alpha', 'beta', 'gamma', 'norm_layers', 'orders', 'accuracy']
d = {}
assert len(all_config_str) == len(results_all)
for i in range(len(all_config_str)):
    d[all_config_str[i]] = results_all[i]

df = pd.DataFrame.from_dict(d)
df = df.sort_values(['dataset', 'accuracy']).reset_index(drop=True)
df.to_csv('results/mlpnorm_result_all.csv')
