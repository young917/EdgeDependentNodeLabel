import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pdb
import time
# source from https://github.com/ksalsw1996DM/HyFER/blob/master/code/models.py

class HGNNLayer(nn.Module):
    def __init__(self, input_dim, vertex_dim, edge_dim, dropout=0.5):
        super(HGNNLayer, self).__init__()
        self.dropout = dropout    
        self.edge_dim = edge_dim
        self.lin_v = torch.nn.Linear(input_dim, vertex_dim)
        if edge_dim > 0:
            self.lin_e = torch.nn.Linear(vertex_dim, edge_dim)
    
    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'weight': edges.src['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        aggr = torch.sum(Weight.unsqueeze(-1) * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}
    
    def weight_fn(self, edges):
        return {'weight_mul': edges.src['weight'] * edges.dst['weight']}

    def message_func2(self, edges):
        return {'h': edges.src['h'], 'weight_mul': edges.data['weight_mul']}

    def reduce_func2(self, nodes):
        weight = nodes.mailbox['weight_mul']
        aggr = torch.sum(weight.unsqueeze(-1) * nodes.mailbox['h'], dim=1)
        return {'h': aggr}


    def forward(self, g1, g2, vfeat, efeat, DV2, invDE):
        with g1.local_scope():
            # vertex to edge gathering
            g1.srcnodes['node'].data['h'] = vfeat
            g1.srcnodes['node'].data['weight'] = DV2[:g1['in'].num_src_nodes()]
            g1.dstnodes['edge'].data['weight'] = invDE[:g1['in'].num_dst_nodes()]
            g1.srcnodes['node'].data['Wh'] = self.lin_v(vfeat)
            g1.apply_edges(self.weight_fn, etype='in')
            g1.update_all(self.message_func, self.reduce_func, etype='in')
            efeat = g1.dstnodes['edge'].data['h']
        with g2.local_scope():
            # edge to vertex gathering
            g2.srcnodes['edge'].data['h'] = efeat
            g2.srcnodes['edge'].data['weight'] = invDE[:g2['con'].num_src_nodes()]
            g2.dstnodes['node'].data['weight'] = DV2[:g2['con'].num_dst_nodes()]
            g2.apply_edges(self.weight_fn, etype='con')
            g2.update_all(self.message_func2, self.reduce_func2, etype='con')
            
            vfeat = g2.dstnodes['node'].data['h']
            vfeat = F.relu(vfeat)            
            vfeat = F.dropout(vfeat, self.dropout)
            
        if self.edge_dim > 0:
            efeat = self.lin_e(efeat)
            
        return [vfeat, efeat]
    
class HGNN(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 hidden_dim, 
                 output_vdim, 
                 output_edim, 
                 num_layer=2, 
                 dropout=0.5, 
                 device='0'):
        super(HGNN, self).__init__()
        
        l = []
        if num_layer == 1:
            l.append(HGNNLayer(input_vdim, output_vdim, output_edim, dropout=0.))
        else:
            for i in range(num_layer):
                if i == 0:
                    l.append(HGNNLayer(input_vdim, hidden_dim, 0, dropout=0.))
                elif i == (num_layer - 1):
                    l.append(HGNNLayer(hidden_dim, output_vdim, output_edim, dropout=dropout))
                else:
                    l.append(HGNNLayer(hidden_dim, hidden_dim, 0, dropout=dropout))
        self.layers = nn.ModuleList(l)
        self.device = device
        
    def forward(self, blocks, vfeat, efeat, DV2, invDE):
        for i, layer in enumerate(self.layers):
            vfeat, efeat = layer(blocks[2*i], blocks[2*i+1], vfeat, efeat, DV2, invDE)
        return vfeat, efeat
    