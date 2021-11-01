from __future__ import division
from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import scipy
import itertools
import sys
from utils import load_data, accuracy, full_load_data, data_split, normalize, train, normalize_adj, sparse_mx_to_torch_sparse_tensor, dataset_edge_balance, random_disassortative_splits, eval_rocauc, load_fixed_splits, evaluate
from models import GCN, snowball, GCNII, SAGE


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--param_tunning', action='store_true', default=False,
                    help='Parameter fine-tunning mode')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5000,
                    help='Number of epochs to train.')
parser.add_argument('--num_splits', type=int,
                    help='number of training/val/test splits ', default=10)
parser.add_argument('--model', type=str,
                    help='name of model (gcn, sgc, graphsage, snowball, gcnII, acmgcn, acmsgc, acmsnowball, mlp)', default='acmgcn')
parser.add_argument('--early_stopping', type=float, default=200,
                    help='early stopping used in GPRGNN')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64,
                    help='Number of hidden units.')
parser.add_argument('--hops', type=int, default=1,
                    help='Number of hops we use, k= 1,2')
parser.add_argument('--layers', type=int, default=1,
                    help='Number of hidden layers, i.e. network depth')
parser.add_argument('--dataset_name', type=str,
                    help='Dataset name.', default='cornell')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.5, help='alpha_l')
parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')
parser.add_argument('--fixed_splits', type=float, default=0,
                    help='0 for random splits in GPRGNN, 1 for fixed splits in GeomGCN')
parser.add_argument('--variant', action='store_true',
                    default=False, help='Indicate ACM, GCNII variant models.')

args = parser.parse_args()
device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

best_result = 0
best_std = 0
best_dropout = None
best_weight_decay = None
best_lr = None
best_hop = None
best_runtime_average = None
best_epoch_average = None
best_layers = None
best_alpha = None
best_lambda = None

# if args.hops == 1:
#     lr = [0.01, 0.05, 0.1]  # [0.002,0.01,0.05]
# else:
#     lr = [0.01, 0.05, 0.1]  # [0.002,0.01,0.05]
# weight_decay = [0, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
lambda_range = [0.5, 1, 1.5]
alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5]
layers_range = [4, 8, 16, 32, 64]
# if args.model == 'sgc':
#     dropout = [0.0]
# else:
#     dropout = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
datasets = ['cornell', 'wisconsin', 'texas', 'film', 'chameleon',
            'squirrel', 'cora', 'citeseer', 'pubmed', 'deezer-europe']


lr = [args.lr, args.lr]
weight_decay = [args.weight_decay, args.weight_decay]
dropout = [args.dropout, args.dropout]


if args.model == 'graphsage':
    adj_low, features, labels = full_load_data(
        args.dataset_name, sage_data=True)
    adj_high = torch.tensor(0)
    nnodes = labels.shape[0]
else:
    adj_low, features, labels = full_load_data(args.dataset_name)
    nnodes = labels.shape[0]
    if (args.model == 'sgc' or args.model == 'acmsgc') and (args.hops > 1):
        A_EXP = adj_low.to_dense()
        for _ in range(args.hops-1):
            A_EXP = torch.mm(A_EXP, adj_low.to_dense())
        adj_low = A_EXP.to_sparse()
        del A_EXP
    adj_high = (torch.eye(nnodes) - adj_low).to_sparse()


if args.cuda & (args.model not in {'graphsage'}):
    features, adj_low, adj_high, labels = features.cuda(
    ), adj_low.cuda(), adj_high.cuda(), labels.cuda()
    torch.cuda.manual_seed(args.seed)


criterion = nn.NLLLoss()
eval_func = accuracy
if args.dataset_name in {'deezer-europe'}:
    args.num_splits = 5
    split_idx_lst = load_fixed_splits(args.dataset_name, '')

# Hyperparameter for other GNNs
for args.lr, args.weight_decay, args.dropout in itertools.product(lr, weight_decay, dropout):
    # for args.layers, args.lamda, args.alpha, args.weight_decay in itertools.product(layers_range, lambda_range, alpha_range, weight_decay): #Hyperparameter for GCNII
    print(args.lr, args.weight_decay, args.dropout)

    def test(eval_func, dataset_name):
        model.eval()
        output = model(features, adj_low, adj_high)
        output = F.log_softmax(output, dim=1)
        acc_test = eval_func(output[idx_test], labels[idx_test])
        return acc_test

    # Train model
    t_total = time.time()
    epoch_total = 0

    result = np.zeros(args.num_splits)
    for idx in range(args.num_splits):
        if args.model == 'snowball':
            model = snowball(nfeat=features.shape[1], nlayers=args.layers, nhid=args.hidden, nclass=labels.max(
            ).item() + 1, dropout=args.dropout).to(device)
        elif args.model == 'gcnII' or args.model == 'acmgcnII':
            model = GCNII(nfeat=features.shape[1], nlayers=args.layers, nhidden=args.hidden, nclass=labels.max().item(
            ) + 1, dropout=args.dropout, lamda=args.lamda, alpha=args.alpha, variant=args.variant, model_type=args.model).to(device)
        elif args.model == 'graphsage':
            model = SAGE(in_feats=features.shape[1], n_hidden=args.hidden, n_classes=labels.max().item(
            ) + 1, n_layers=2, dropout=args.dropout, model_type=args.model, variant=args.variant).to(device)
        else:  # args.model in {'mlp', 'gcn', 'sgc', 'acmsgc', 'acmgcn', 'acmsnowball'}
            model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=labels.max().item(
            ) + 1, dropout=args.dropout,  model_type=args.model, nlayers=args.layers, variant=args.variant).to(device)

        # for layer in model.gcns:
        #     layer.reset_parameters()

        if args.dataset_name in {'deezer-europe'}:
            idx_train, idx_val, idx_test = split_idx_lst[idx]['train'].to(
                device), split_idx_lst[idx]['valid'].to(device), split_idx_lst[idx]['test'].to(device)
            args.epochs = 500
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            if args.fixed_splits == 0:
                idx_train, idx_val, idx_test = random_disassortative_splits(
                    labels, labels.max()+1)
            else:
                idx_train, idx_val, idx_test = data_split(
                    idx, args.dataset_name)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_test = 0
        best_training_loss = None
        best_val_acc = 0
        best_val_loss = float('inf')
        val_loss_history = torch.zeros(args.epochs)
        for epoch in range(args.epochs):
            t = time.time()
            acc_train, loss_train = train(model, optimizer, adj_low, adj_high, features,
                                          labels, idx_train, idx_val, criterion, dataset_name=args.dataset_name)

            model.eval()
            output = model(features, adj_low, adj_high)

            output = F.log_softmax(output, dim=1)
            val_loss, val_acc = criterion(output[idx_val], labels[idx_val]), evaluate(
                output, labels, idx_val, eval_func)

            if args.dataset_name in {'deezer-europe'}:
                if val_acc > best_val_acc:  # :
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_test = evaluate(output, labels, idx_test, eval_func)
                    best_training_loss = loss_train

            else:
                if val_loss < best_val_loss:  # :
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_test = evaluate(output, labels, idx_test, eval_func)
                    best_training_loss = loss_train

                if epoch >= 0:
                    val_loss_history[epoch] = val_loss.detach()
                if args.early_stopping > 0 and epoch > args.early_stopping:
                    tmp = torch.mean(
                        val_loss_history[epoch-args.early_stopping:epoch])
                    if val_loss > tmp:
                        break
        epoch_total = epoch_total + epoch

        if args.param_tunning:
            print("Optimization for %s, %s, weight decay %.5f, dropout %.4f, split %d, Best Test Result: %.4f, Training Loss: %.4f" % (
                args.model, args.dataset_name, args.weight_decay, args.dropout, idx, best_test, best_training_loss))
        # Testing

        result[idx] = best_test
        del model, optimizer
        if args.cuda:
            torch.cuda.empty_cache()
    total_time_elapsed = time.time() - t_total
    runtime_average = total_time_elapsed/args.num_splits
    epoch_average = total_time_elapsed/epoch_total * 1000
    if np.mean(result) > best_result:
        best_result = np.mean(result)
        best_std = np.std(result)
        best_dropout = args.dropout
        best_weight_decay = args.weight_decay
        best_lr = args.lr
        best_hop = args.hops
        best_layers = args.layers
        best_alpha = args.alpha
        best_lambda = args.lamda
        best_runtime_average = runtime_average
        best_epoch_average = epoch_average

    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if args.param_tunning:
        print("%s, Hops: %s, Hidden: %s, learning rate %.4f, weight decay %.6f, dropout %.4f, layers %s, alpha %.2f, lambda %.2f, Test Mean: %.4f, Test Std: %.4f, runtime average time: %4f s, epoch average time: %4f ms" %
              (args.dataset_name, args.hops, args.hidden, args.lr, args.weight_decay, args.dropout, args.layers, args.alpha, args.lamda, np.mean(result), np.std(result), runtime_average, epoch_average))
    else:
        print("%s, Model: %s, variant: %s, Hops: %s, Hidden: %s, learning rate %.4f, weight decay %.6f, dropout %.4f, layers %s, alpha %.2f, lambda %.2f, Test Mean: %.4f, Test Std: %.4f, runtime average time: %.2fs, epoch average time: %.2fms" %
              (args.dataset_name, args.model, args.variant, args.hops, args.hidden, args.lr, args.weight_decay, args.dropout, args.layers, args.alpha, args.lamda, np.mean(result), np.std(result), runtime_average, epoch_average))

print("Best Result of %s, variant: %s, on Dataset: %s, Hops: %s, Hidden: %s, learning rate %.4f, weight decay %.6f, dropout %.4f, layers %s, alpha %.2f, lambda %.2f, Test Mean: %.4f, Test Std: %.4f, epoch average/runtime average time: %.2fms/%.2fs" %
      (args.model, args.variant, args.dataset_name, best_hop, args.hidden, best_lr, best_weight_decay, best_dropout, best_layers, best_alpha, best_lambda, best_result, best_std, best_epoch_average, best_runtime_average))
