import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import sys
import pickle as pkl
import networkx as nx
import json
from networkx.readwrite import json_graph
import pdb
import os
import re
import torch.nn as nn
import torch as th
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score
from numpy.linalg import matrix_power
from os import path
import dgl
from google_drive_downloader import GoogleDriveDownloader as gdd
import scipy
DATAPATH = path.dirname(path.abspath(__file__)) + '/data/'
dataset_drive_url = {
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}

if torch.cuda.is_available():
    from collections import defaultdict
    import scipy.io
    from sklearn.preprocessing import label_binarize
    from ogb.nodeproppred import NodePropPredDataset

    from torch_geometric.datasets import Planetoid
    from torch_geometric.transforms import NormalizeFeatures
    from torch_geometric.utils import add_self_loops, to_dense_adj, contains_self_loops, remove_self_loops

    from torch_sparse import SparseTensor

device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

sys.setrecursionlimit(99999)


def train(model, optimizer, adj_low, adj_high, features, labels, idx_train, idx_val, criterion, dataset_name):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj_low, adj_high)
    output = F.log_softmax(output, dim=1)
    loss_train = criterion(output[idx_train], labels[idx_train])
    acc_train = accuracy(labels[idx_train], output[idx_train])
    loss_train.backward()
    optimizer.step()

    return 100 * acc_train.item(), loss_train.item()


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = (1.0 / rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def load_torch_geometric_data(dataset, name):
    cur = os.getcwd()

    if dataset in {'WikiCS', 'Flickr'}:
        data = eval(dataset + "(root = '" + cur.replace("\\", "/") +
                    "/torch_geometric_data/" + dataset + "')")
    else:
        data = eval(dataset + "(root = '" + cur.replace("\\", "/") +
                    "/torch_geometric_data/" + dataset + "'," + "name = '" + name + "')")
    # e.g. Coauthor(root='...', name = 'CS')

    edge = data[0].edge_index
    if contains_self_loops(edge):
        edge = remove_self_loops(edge)[0]
        print("Original data contains self-loop, it is now removed")

    adj = to_dense_adj(edge)[0].numpy()

    print("Nodes: %d, edges: %d, features: %d, classes: %d. \n" % (
        len(adj[0]), len(edge[0])/2, len(data[0].x[0]), len(np.unique(data[0].y))))

    mask = np.transpose(adj) != adj
    col_sum = adj.sum(axis=0)
    print("Check adjacency matrix is sysmetric: %r" % (mask.sum().item() == 0))
    print("Check the number of isolated nodes: %d" %
          ((col_sum == 0).sum().item()))
    print("Node degree Max: %d, Mean: %.4f, SD: %.4f" %
          (col_sum.max(), col_sum.mean(), col_sum.std()))

    return adj, data[0].x.numpy(), data[0].y.numpy()


def full_load_data(dataset_name, sage_data=False):
    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        adj, features, labels = load_data(dataset_name)
        labels = np.argmax(labels, axis=-1)
        features = features.todense()
        G = nx.DiGraph(adj).to_undirected()
    elif dataset_name in {'CitationFull_dblp', 'Coauthor_CS', 'Coauthor_Physics', 'Amazon_Computers', 'Amazon_Photo'}:
        dataset, name = dataset_name.split("_")
        adj, features, labels = load_torch_geometric_data(dataset, name)

    elif dataset_name in {'Flickr', 'WikiCS'}:
        adj, features, labels = load_torch_geometric_data(dataset_name, None)
    elif dataset_name == 'deezer-europe':
        dataset = load_deezer_dataset()

        row, col = dataset.graph['edge_index']
        N = dataset.graph['num_nodes']

        adj = sp.coo_matrix((np.ones(row.shape[0]), (row, col)), shape=(N, N))
        features, labels = dataset.graph['node_feat'], dataset.label

    else:
        graph_adjacency_list_file_path = os.path.join(
            'new_data', dataset_name, 'out1_graph_edges.txt')
        graph_node_features_and_labels_file_path = os.path.join('new_data', dataset_name,
                                                                'out1_node_feature_label.txt')

        G = nx.DiGraph().to_undirected()
        graph_node_features_dict = {}
        graph_labels_dict = {}

        if dataset_name == 'film':
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    feature_blank = np.zeros(932, dtype=np.uint8)
                    feature_blank[np.array(
                        line[1].split(','), dtype=np.uint16)] = 1
                    graph_node_features_dict[int(line[0])] = feature_blank
                    graph_labels_dict[int(line[0])] = int(line[2])
        else:
            with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
                graph_node_features_and_labels_file.readline()
                for line in graph_node_features_and_labels_file:
                    line = line.rstrip().split('\t')
                    assert (len(line) == 3)
                    assert (int(line[0]) not in graph_node_features_dict and int(
                        line[0]) not in graph_labels_dict)
                    graph_node_features_dict[int(line[0])] = np.array(
                        line[1].split(','), dtype=np.uint8)
                    graph_labels_dict[int(line[0])] = int(line[2])

        with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
            graph_adjacency_list_file.readline()
            for line in graph_adjacency_list_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 2)
                if int(line[0]) not in G:
                    G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                               label=graph_labels_dict[int(line[0])])
                if int(line[1]) not in G:
                    G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                               label=graph_labels_dict[int(line[1])])
                G.add_edge(int(line[0]), int(line[1]))

        adj = nx.adjacency_matrix(G, sorted(G.nodes()))

        features = np.array(
            [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
        labels = np.array(
            [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    features = preprocess_features(features)
    features = th.FloatTensor(features).to(device)
    labels = th.LongTensor(labels).to(device)

    if sage_data == True:
        if dataset_name in {'deezer-europe'}:
            g = dgl.DGLGraph(adj+sp.eye(N))  # .to(device)
        else:
            g = dgl.DGLGraph(adj+sp.eye(adj.shape[0]))  # .to(device)
        # Adapted from https://docs.dgl.ai/tutorials/models/1_gnn/1_gcn.html
        g.ndata['features'] = features
        g.ndata['labels'] = labels
        degs = g.in_degrees().float()
        norm = th.pow(degs, -1).to(device)
        norm[th.isinf(norm)] = 0
        g.ndata['norm'] = norm.unsqueeze(1)
        return g, features, labels

    adj = sparse_mx_to_torch_sparse_tensor(normalize(adj+sp.eye(adj.shape[0])))

    return adj, features, labels


def data_split(idx, dataset_name):
    splits_file_path = 'splits/'+dataset_name+'_split_0.6_0.2_'+str(idx)+'.npz'
    with np.load(splits_file_path) as splits_file:
        train_mask = splits_file['train_mask']
        val_mask = splits_file['val_mask']
        test_mask = splits_file['test_mask']
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)
    test_mask = th.BoolTensor(test_mask)
    return train_mask, val_mask, test_mask


def normalize(mx, eqvar=None):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if eqvar:
        r_inv = np.power(rowsum, -1/eqvar).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    else:
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


def accuracy(labels, output):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sys_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    row_sum = np.array(adj.sum(1))
    row_sum = (row_sum == 0)*1+row_sum
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def dataset_edge_balance(adj, labels):
    """Measure the edge balance of each dataset"""
    num_class = labels.max().item() + 1
    num_nodes = labels.size()[0]
    num_nodes_per_class = np.zeros(num_class)
    edge_balance_per_class = np.zeros([num_class, 2])
    for i in range(num_class):
        idx = np.where(labels.numpy() == i)[0]
        num_nodes_per_class[i] = idx.shape[0]
        edge_balance_per_class[i, 0] = np.sum(adj[idx, :][:, idx])
        edge_balance_per_class[i, 1] = np.sum(
            adj[idx, :][:, np.delete(np.arange(num_nodes), idx)])

    return num_nodes_per_class, edge_balance_per_class


def random_disassortative_splits(labels, num_classes):
    # * 0.6 labels for training
    # * 0.2 labels for validation
    # * 0.2 labels for testing
    labels, num_classes = labels.cpu(), num_classes.cpu().numpy()
    indices = []
    for i in range(num_classes):
        index = torch.nonzero((labels == i)).view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)
    percls_trn = int(round(0.6*(labels.size()[0]/num_classes)))
    val_lb = int(round(0.2*labels.size()[0]))
    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    train_mask = index_to_mask(train_index, size=labels.size()[0])

    val_mask = index_to_mask(rest_index[:val_lb], size=labels.size()[0])
    test_mask = index_to_mask(rest_index[val_lb:], size=labels.size()[0])

    return train_mask.to(device), val_mask.to(device), test_mask.to(device)


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask


def rand_train_test_idx(label, train_prop=.6, valid_prop=.2, ignore_negative=True):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return train_indices, val_indices, test_indices

    train_idx = labeled_nodes[train_indices]
    valid_idx = labeled_nodes[val_indices]
    test_idx = labeled_nodes[test_indices]

    return train_idx, valid_idx, test_idx


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction: 

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form: 
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """
        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None


def load_deezer_dataset():
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{DATAPATH}deezer-europe.mat')
    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def gen_normalized_adjs(dataset):
    """ returns the normalized adjacency matrix
    """
    dataset.graph['edge_index'] = add_self_loops(
        dataset.graph['edge_index'])[0]
    row, col = dataset.graph['edge_index']
    N = dataset.graph['num_nodes']
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    deg = adj.sum(dim=1).to(torch.float)
    D_isqrt = deg.pow(-0.5)
    D_isqrt[D_isqrt == float('inf')] = 0
    DAD = D_isqrt.view(-1, 1) * adj * D_isqrt.view(1, -1)
    DA = D_isqrt.view(-1, 1) * D_isqrt.view(-1, 1) * adj
    AD = adj * D_isqrt.view(1, -1) * D_isqrt.view(1, -1)
    return DAD, DA, AD


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(
            y_pred, dim=-1)[:, 1].unsqueeze(1).detach().cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list)/len(rocauc_list)


@torch.no_grad()
def evaluate(output, labels, split_idx, eval_func):
    acc = eval_func(labels[split_idx], output[split_idx])
    return acc


def load_fixed_splits(dataset, sub_dataset):
    name = dataset
    if sub_dataset:
        name += f'-{sub_dataset}'

    if not os.path.exists(f'./data/splits/{name}-splits.npy'):
        assert dataset in splits_drive_url.keys()
        gdd.download_file_from_google_drive(
            file_id=splits_drive_url[dataset],
            dest_path=f'./data/splits/{name}-splits.npy', showsize=True)

    splits_lst = np.load(f'./data/splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])
    return splits_lst


dataset_drive_url = {
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}

splits_drive_url = {
    'snap-patents': '12xbBRqd8mtG_XkNLH8dRRNZJvVM4Pw-N',
    'pokec': '1ZhpAiyTNc0cE_hhgyiqxnkKREHK7MK-_',
}
