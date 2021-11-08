import pickle
import pandas as pd
import os


with open('config_list5', 'rb') as f:
    config_list = pickle.load(f)
non_para = 0
exist_para = 0

datasets = ['chameleon', 'cornell', 'squirrel', 'film',
            'texas', 'wisconsin', 'pubmed', 'cora', 'citeseer']
config_str = ['lr', 'do', 'es', 'wd', 'alpha', 'beta',
              'gamma', 'nlid', 'nl', 'ordersid', 'orders']
results_all = [[] for _ in range(len(config_str) + 2)]

zip_iterator = zip([0, 1e-7, 1e-6, 5e-5, 1e-4, 5e-4],
                   ['0.0', '1e-07', '1e-06', '5e-05', '0.0001', '0.0005'])
weight_decay_map = dict(zip_iterator)

print('the number of all parameters', len(config_list))
for config in config_list:
    # print('config', config)
    config[3] = weight_decay_map[config[3]]
    acc_res = []
    for i in range(10):
        url_configs = ''
        for j in range(len(config_str)):
            url_configs += '_' + config_str[j] + str(config[j])
        url = 'runs/' + datasets[config[-1]] + url_configs + \
            '_split' + str(i) + '_results.txt'
        # print(url)
        if os.path.exists(url):
            # print('url exists')
            with open(url, 'r') as f:
                s = f.read()
                sub1 = '"test_acc": '
                sub2 = ', "test_duration"'
                result = s[s.index(sub1) + len(sub1): s.index(sub2)]
                acc_res.append(float(result))
    # print(url)
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

print('the number of non-existing parameters', non_para)
print('the number of existing parameters', exist_para)
# print('the length of results', len(results_all[0]))


all_config_str = ['dataset', 'lr', 'dropout', 'early_stopping', 'weight_decay',
                  'alpha', 'beta', 'gamma', 'norm_func_id', 'norm_layers',
                  'orders_func_id', 'orders', 'accuracy']
d = {}
assert len(all_config_str) == len(results_all)
for i in range(len(all_config_str)):
    d[all_config_str[i]] = results_all[i]

df = pd.DataFrame.from_dict(d)
df = df.sort_values(['dataset', 'accuracy']).reset_index(drop=True)
df.to_csv('results/mlpnorm_result_all5.csv')
