import pickle
import pandas as pd
import os


with open('config_list', 'rb') as f:
    config_list = pickle.load(f)
non_para = 0
exist_para = 0

datasets = ['chameleon', 'cornell', 'squirrel',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']
config_str = ['lr', 'do', 'es', 'wd', 'alpha', 'beta', 'gamma', 'nl', 'orders']
results_all = [[] for _ in range(len(config_str) + 2)]


for config in config_list:
    # print('config', config)
    # for d in datasets:
    acc_res = []
    for i in range(10):
        url_configs = ''
        for j in range(len(config_str)):
            url_configs += '_' + config_str[j] + str(config[j])
        url = 'runs/' + datasets[config[-1]] + url_configs + \
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
        results_all[0].append(datasets[config[-1]])
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
