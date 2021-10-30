import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from gcn.utils import load_data

# np.set_printoptions(threshold=np.inf)

x, y, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('citeseer')
print(x.nnz)
print(x.shape)
print(y.shape)

x = x.toarray()
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if x[i][j] != x[j][i]:
            print('hehe')
            break