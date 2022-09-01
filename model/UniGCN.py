import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import time

class UniGCNIILayer(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 vertex_dim,
                 edge_dim,
                 use_norm=False
                 ):
        super(UniGCNIILayer, self).__init__()
        
        self.W = nn.Linear(input_vdim, vertex_dim, bias=False)
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.W.weight, gain=gain)
    
    def message1_func(self, edges):
        return {'Wh': edges.src['h']}
    
    def reduce1_func(self, nodes):
        aggr = torch.mean(nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}
    
    def weight_fn(self, edges):
        weight = edges.src['deg'] * edges.dst['deg']
        return {'weight': weight}
    
    def message2_func(self, edges):
        #pdb.set_trace()
        return {'Wh': edges.src['h'], 'weight': edges.data['weight']}

    def reduce2_func(self, nodes):
        Weight = nodes.mailbox['weight']
        aggr = torch.sum(Weight * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}
    
    def forward(self, g1, g2, vfeat, efeat, degE, degV, alpha, beta, vfeat0):
        with g1.local_scope():
            g1.srcnodes['node'].data['h'] = vfeat
            g1.update_all(self.message1_func, self.reduce1_func, etype='in')
            efeat = g1.dstnodes['edge'].data['h']
        with g2.local_scope():
            g2.srcnodes['edge'].data['h'] = efeat
            g2.srcnodes['edge'].data['deg'] = degE[:g2['con'].num_src_nodes()]
            g2.dstnodes['node'].data['deg'] = degV[:g2['con'].num_dst_nodes()]
            g2.apply_edges(self.weight_fn, etype='con')
            g2.update_all(self.message2_func, self.reduce2_func, etype='con')
            vfeat = g2.dstnodes['node'].data['h']
            
            vi = (1-alpha) * vfeat + alpha * vfeat0[:g2['con'].num_dst_nodes()]
            v = (1-beta) * vi + beta * self.W(vi)
            
        return [v, efeat]
    
class UniGCNII(nn.Module):
    def __init__(self,
                input_vdim,
                input_edim,
                hidden_dim,
                output_vdim,
                output_edim,
                num_layer = 2,
                dropout=0.2,
                device='0'):
        super(UniGCNII, self).__init__()
        
        self.activation = F.relu
        self.dropout = nn.Dropout(dropout) # 0.2 is chosen for GCNII

        l = []
        l.append(torch.nn.Linear(input_vdim, hidden_dim))
        for i in range(num_layer):
            l.append(UniGCNIILayer(hidden_dim, hidden_dim, hidden_dim, hidden_dim))
        self.outlin_v = torch.nn.Linear(hidden_dim, output_vdim)
        self.outlin_e = torch.nn.Linear(hidden_dim, output_edim)
        self.layers = nn.ModuleList(l)
        self.device = device
        self.reg_params = list(self.layers[1:].parameters())
        self.non_reg_params = list(self.layers[0:1].parameters())+list(self.outlin_v.parameters()) + list(self.outlin_e.parameters())
         
    def forward(self, blocks, vfeat, efeat, degE, degV):
        lamda, alpha = 0.5, 0.1
        vfeat = self.dropout(vfeat)
        vfeat = self.activation(self.layers[0](vfeat))
        v0 = vfeat
        for i, layer in enumerate(self.layers[1:]):
            vfeat = self.dropout(vfeat)
            beta = math.log(lamda / (i+1)+1)
            vfeat, efeat = layer(blocks[2*i], blocks[2*i+1], vfeat, efeat, degE, degV, alpha, beta, v0)
            vfeat = F.relu(vfeat)
        vfeat = self.dropout(vfeat)
        efeat = self.dropout(efeat)
        vfeat = self.outlin_v(vfeat)
        efeat = self.outlin_e(efeat)
        return vfeat, efeat
        