import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn

import dgl.function as fn
import dgl.utils as utl 

class general_GCN_layer(Module):
    def __init__(self):
        super(general_GCN_layer, self).__init__()

    @staticmethod
    def multiplication(A, B):
        if str(A.layout) == 'torch.sparse_coo':
            return torch.spmm(A, B)
        else:
            return torch.mm(A, B)

class snowball_layer(general_GCN_layer):
    def __init__(self, in_features, out_features):
        super(snowball_layer, self).__init__()
        self.in_features, self.out_features = in_features, out_features
        if torch.cuda.is_available():
            self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features).cuda()), Parameter(torch.FloatTensor(self.out_features).cuda())
        else:
            self.weight, self.bias = Parameter(torch.FloatTensor(self.in_features, self.out_features)), Parameter(torch.FloatTensor(self.out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv_weight, stdv_bias = 1. / math.sqrt(self.weight.size(1)), 1. / math.sqrt(self.bias.size(0))
        torch.nn.init.uniform_(self.weight, -stdv_weight, stdv_weight)
        torch.nn.init.uniform_(self.bias, -stdv_bias, stdv_bias)
    
    def forward(self, input, adj, eye=False):
        XW = torch.mm(input, self.weight)
        if eye:
            return XW + self.bias
        else:
            return self.multiplication(adj, XW) + self.bias

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, model_type, output_layer = 0, variant = False):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type, self.variant = in_features, out_features, output_layer, model_type, variant
        self.att_low, self.att_high, self.att_mlp = 0,0,0
        if torch.cuda.is_available():
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features).cuda()), Parameter(torch.FloatTensor(in_features, out_features).cuda()), Parameter(torch.FloatTensor(in_features, out_features).cuda())           
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1).cuda()), Parameter(torch.FloatTensor(out_features, 1).cuda()), Parameter(torch.FloatTensor(out_features, 1).cuda())
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(1, 1).cuda()), Parameter(torch.FloatTensor(1, 1).cuda()), Parameter(torch.FloatTensor(1, 1).cuda())
            
            self.att_vec = Parameter(torch.FloatTensor(3, 3).cuda())

        else:
            self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))           
            self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1)), Parameter(torch.FloatTensor(out_features, 1)), Parameter(torch.FloatTensor(out_features, 1))
            self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(1, 1)), Parameter(torch.FloatTensor(1, 1)), Parameter(torch.FloatTensor(1, 1))
            
            self.att_vec = Parameter(torch.FloatTensor(3, 3))
        self.reset_parameters()

    def reset_parameters(self): 
        
        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt( self.att_vec_mlp.size(1))
        
        std_att_vec = 1. / math.sqrt( self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        
        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)
 
    def attention(self, output_low, output_high, output_mlp): #
        T = 3
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low) ,torch.mm((output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)  ],1)), self.att_vec)/T,1) #
        return att[:,0][:,None],att[:,1][:,None],att[:,2][:,None]

    def forward(self, input, adj_low, adj_high):
        output = 0
        if self.model_type == 'mlp':
            output_mlp = (torch.mm(input, self.weight_mlp))
            return output_mlp
        elif self.model_type == 'sgc' or self.model_type == 'gcn':
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == 'acmgcn' or self.model_type == 'acmsnowball':
            if self.variant:
                output_low = (torch.spmm(adj_low, F.relu(torch.mm(input, self.weight_low))))
                output_high = (torch.spmm(adj_high, F.relu(torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))          
            else:
                output_low = F.relu(torch.spmm(adj_low, (torch.mm(input, self.weight_low))))
                output_high = F.relu(torch.spmm(adj_high, (torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))
                
            self.att_low, self.att_high, self.att_mlp = self.attention((output_low), (output_high), (output_mlp)) # 
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp) #  3*(output_low + output_high + output_mlp) #
        elif self.model_type == 'acmsgc':
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            output_high = torch.spmm(adj_high,  torch.mm(input, self.weight_high)) #torch.mm(input, self.weight_high) - torch.spmm(self.A_EXP,  torch.mm(input, self.weight_high))
            output_mlp = torch.mm(input, self.weight_mlp)
            
            self.att_low, self.att_high, self.att_mlp = self.attention((output_low), (output_high), (output_mlp)) #   self.attention(F.relu(output_low), F.relu(output_high), F.relu(output_mlp)) 
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp) # 3*(output_low + output_high + output_mlp) #
            
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
               

class GraphConvolutionII(GraphConvolution):
    def __init__(self, in_features, out_features, model_type = 'gcnII', residual=False, variant=False):
        super(GraphConvolutionII, self).__init__(in_features, out_features, model_type)
        self.model_type, self.variant = model_type, variant
        self.out_features = out_features
        self.residual = residual

    def forward(self, input, adj, adj_high, h0, lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi_low = torch.spmm(adj, input)
        if self.model_type == 'gcnII':
            if self.variant:
                support = torch.cat([hi_low,h0],1)
                r = (1-alpha)*hi_low+alpha*h0
            else:
                support = (1-alpha)*hi_low+alpha*h0
                r = support
            output = theta*torch.mm(support, self.weight_low)+(1-theta)*r
            if self.residual:
                output = output+input
            return output
        else:
            hi_high = torch.spmm(adj, input)
            if self.variant:
                support_low = torch.cat([hi_low,h0],1)
                r_low = (1-alpha)*hi_low+alpha*h0
                support_high = torch.cat([hi_high,h0],1)
                r_high = (1-alpha)*hi_high+alpha*h0
                support_mlp = torch.cat([input,h0],1)
                r_mlp = (1-alpha)*input+alpha*h0
            else:
                support_low = (1-alpha)*hi_low+alpha*h0
                r_low = support_low
                support_high = (1-alpha)*hi_high+alpha*h0
                r_high = support_high
                support_mlp = (1-alpha)*input+alpha*h0
                r_mlp = support_mlp
            output_low = F.relu(theta*torch.mm(support_low, self.weight_low)+(1-theta)*r_low)
            output_high = F.relu(theta*torch.mm(support_high, self.weight_high)+(1-theta)*r_high)
            output_mlp = F.relu(theta*torch.mm(support_mlp, self.weight_mlp)+(1-theta)*r_mlp)
            self.att_low, self.att_high, self.att_mlp = self.attention((output_low), (output_high), (output_mlp)) #self.attention(F.relu(output_low), F.relu(output_high), F.relu(output_mlp)) 
            output = 3*(self.att_low*output_low + self.att_high*output_high  + self.att_mlp*output_mlp) # 
            if self.residual:
                output = output+input
            return output
            



class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, model_type, variant, feat_drop=0., bias=True, norm=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = utl.expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.model_type, self.variant = model_type, variant
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)

        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_self_high = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.fc_neigh_high = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.fc_identity = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        
        
        self.att_vec_mlp= nn.Linear(out_feats, 1, bias=False)
        self.att_vec_low = nn.Linear(out_feats, 1, bias=False)
        self.att_vec_high = nn.Linear(out_feats, 1, bias=False)
        self.att_vec = nn.Linear(3, 3, bias=False)
                                          
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh_high.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_identity.weight, gain=gain)
        
     
        nn.init.xavier_uniform_(self.att_vec_mlp.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_low.weight, gain=gain)
        nn.init.xavier_uniform_(self.att_vec_high.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self._in_src_feats)),
             m.new_zeros((1, batch_size, self._in_src_feats)))
        _, (rst, _) = self.lstm(m, h)
        return {'neigh': rst.squeeze(0)}
    
    def _mean_reducer(self, nodes):
        return {'neigh':torch.mean(nodes.mailbox['m'],dim=1)}
    
    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(self.att_vec(torch.sigmoid(torch.cat([self.att_vec_low(output_low) ,self.att_vec_high(output_high) ,self.att_vec_mlp(output_mlp) ],1)))/T,1)
        return att[:,0][:,None],att[:,1][:,None],att[:,2][:,None]
    def mean_aggregator(self, graph, feat_src):
        graph.srcdata['h'] = feat_src * graph.ndata['norm']
        graph.update_all(fn.copy_src('h', 'm'), self._mean_reducer)
        return graph.dstdata['neigh']
    def forward(self, graph, feat):
        with graph.local_scope():
            feat = self.feat_drop(feat)
            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat.shape[0], self._in_src_feats).to(feat)
            if self.model_type == 'acmgraphsage':
                if self.variant:
                    self_low = F.relu(self.fc_self(feat))
                    self_high = F.relu(self.fc_self_high(feat))
                    neigh_low = self.mean_aggregator(graph, F.relu(self.fc_neigh(feat)))
                    neigh_high = self.mean_aggregator(graph, F.relu(self.fc_neigh_high(feat)))
                    low = self_low+neigh_low
                    high = self_high-neigh_high
                else:
                    self_low = (self.fc_self(feat))
                    self_high = (self.fc_self_high(feat))
                    neigh_low = (self.fc_neigh(self.mean_aggregator(graph, feat)))
                    neigh_high = (self.fc_neigh_high(self.mean_aggregator(graph, feat)))
                    low = F.relu(self_low+neigh_low)
                    high = F.relu(self_high-neigh_high)
                identity = F.relu(self.fc_identity(feat))
                att_low, att_high, att_mlp = self.attention(low, high, identity)
                rst = 3*(att_low*low + att_high*high + att_mlp*identity)
            else:
                rst = self.fc_self(feat) + self.fc_neigh(self.mean_aggregator(graph, feat))
            # rst = F.relu(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst
