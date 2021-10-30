import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, general_GCN_layer, snowball_layer, GraphConvolutionII
import torch
import numpy as np

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
import time
import argparse
import tqdm
from layers import SAGEConv


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, model_type, nlayers=1, variant=False):
        super(GCN, self).__init__()
        self.gcns, self.mlps = nn.ModuleList(),nn.ModuleList()
        self.model_type, self.nlayers, = model_type, nlayers
        if self.model_type =='mlp':
            self.gcns.append(GraphConvolution(nfeat, nhid, model_type = model_type)) 
            self.gcns.append(GraphConvolution(nhid, nclass, model_type = model_type, output_layer=1)) 
        elif self.model_type =='gcn' or self.model_type =='acmgcn':
            self.gcns.append(GraphConvolution(nfeat, nhid,  model_type = model_type, variant = variant)) 
            self.gcns.append(GraphConvolution(nhid, nclass,  model_type = model_type, output_layer=1, variant = variant)) 
        elif self.model_type =='sgc' or self.model_type =='acmsgc':
            self.gcns.append(GraphConvolution(nfeat, nclass, model_type = model_type))
        elif self.model_type =='acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolution(k * nhid + nfeat, nhid, model_type = model_type, variant = variant))
            self.gcns.append(GraphConvolution(nlayers * nhid + nfeat, nclass, model_type = model_type, variant = variant))
        self.dropout = dropout
    

    def forward(self, x, adj_low, adj_high):
        if self.model_type =='acmgcn' or self.model_type =='acmsgc' or self.model_type =='acmsnowball':
            x = F.dropout(x, self.dropout, training=self.training)
        
        if self.model_type =='acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))
            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea = (self.gcns[0](x, adj_low, adj_high))
        
        if self.model_type =='gcn' or  self.model_type =='mlp' or self.model_type =='acmgcn': 
            fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
            fea = self.gcns[-1](fea, adj_low, adj_high)
        return fea
    

class graph_convolutional_network(nn.Module):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(graph_convolutional_network, self).__init__()
        self.nfeat, self.nlayers, self.nhid, self.nclass = nfeat, nlayers, nhid, nclass
        self.dropout = dropout
        self.hidden = nn.ModuleList()

    def reset_parameters(self):
        for layer in self.hidden:
            layer.reset_parameters()
        self.out.reset_parameters()

class snowball(graph_convolutional_network):
    def __init__(self, nfeat, nlayers, nhid, nclass, dropout):
        super(snowball, self).__init__(nfeat, nlayers, nhid, nclass, dropout)
        for k in range(nlayers):
            self.hidden.append(snowball_layer(k * nhid + nfeat, nhid))
        self.out = snowball_layer(nlayers * nhid + nfeat, nclass)
    
    def forward(self, x, adj, adj_high):
        list_output_blocks = []
        for layer, layer_num in zip(self.hidden, np.arange(self.nlayers)):
            if layer_num == 0:
                list_output_blocks.append(F.dropout(F.relu(layer(x, adj)), self.dropout, training=self.training))
            else:
                list_output_blocks.append(F.dropout(F.relu(layer(torch.cat([x] + list_output_blocks[0: layer_num], 1), adj)), self.dropout, training=self.training))
        output = self.out(torch.cat([x] + list_output_blocks, 1), adj, eye=False)
        return output
    
    

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant, model_type):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()

        for _ in range(nlayers):
            self.convs.append(GraphConvolutionII(nhidden + torch.tensor(variant)*nhidden, nhidden, model_type = model_type, variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.model_type = model_type 

    def forward(self, x, adj, adj_high):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,adj_high,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout, model_type, variant=False):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, model_type, variant))
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, model_type, variant))
        self.layers.append(SAGEConv(n_hidden, n_classes, model_type, variant))
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, graph, adj_high):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h