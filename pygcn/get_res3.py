import pickle


with open('config_list', 'rb') as f:
    config_list = pickle.load(f)

print(len(config_list))
