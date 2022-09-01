import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pdb
import time
# source from https://github.com/ksalsw1996DM/HyFER/blob/master/code/models.py

class HyperAttnLayer(nn.Module):
    # edge attention  version
    def __init__(self, input_vdim, input_edim, query_dim, vertex_dim, edge_dim, weight_dim=0, dropout = 0.5):
        super(HyperAttnLayer, self).__init__()
        self.dropout = dropout
        self.query_dim = query_dim
        self.weight_dim = weight_dim
        
        self.vtx_lin = torch.nn.Linear(input_vdim, vertex_dim)
        
        self.qe_lin = torch.nn.Linear(input_edim, query_dim)
        self.kv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.vv_lin = torch.nn.Linear(vertex_dim, edge_dim)
        
        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.ve_lin = torch.nn.Linear(edge_dim, vertex_dim)
        
        if self.weight_dim > 0:
            self.wt_lin = nn.Linear(weight_dim, query_dim, bias=False)

    def attention(self, edges):
        if self.weight_dim > 0:
            keyvalue = edges.src['k'] + self.wt_lin(edges.data['weight'])
            attn_score = F.leaky_relu((keyvalue * edges.dst['q']).sum(-1))
        else:
            attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        return {'Attn': attn_score/np.sqrt(self.query_dim)}
    
    def message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def reduce_func(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
            
        return {'h': aggr}

    def forward(self, g1, g2, vfeat, efeat):
        with g1.local_scope():
            feat_v = self.vtx_lin(vfeat)
            feat_e = efeat[:g1['in'].num_dst_nodes()]

            # edge attention
            g1.srcnodes['node'].data['h'] = feat_v
            g1.srcnodes['node'].data['k'] = self.kv_lin(feat_v)
            g1.srcnodes['node'].data['v'] = self.vv_lin(feat_v)
            g1.dstnodes['edge'].data['q'] = self.qe_lin(feat_e)
            g1.apply_edges(self.attention, etype='in')
            g1.update_all(self.message_func, self.reduce_func, etype='in')
            feat_e = F.relu(g1.dstnodes['edge'].data['h'])
            
        with g2.local_scope():
            # node attention
            g2.srcnodes['edge'].data['h'] = feat_e
            g2.srcnodes['edge'].data['k'] = self.ke_lin(feat_e)
            g2.srcnodes['edge'].data['v'] = self.ve_lin(feat_e)
            g2.dstnodes['node'].data['q'] = self.qv_lin(feat_v[:g2['con'].num_dst_nodes()])
            g2.apply_edges(self.attention, etype='con')
            g2.update_all(self.message_func, self.reduce_func, etype='con')
            feat_v = F.relu(g2.dstnodes['node'].data['h'])
            feat_v = F.dropout(feat_v, self.dropout)
                        
            return [feat_v, feat_e]
        
class HyperAttn(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 hidden_dim,
                 output_vdim, 
                 output_edim, 
                 query_dim=64,
                 weight_dim=0,
                 num_layer=2, 
                 dropout=0.5, 
                 device='0'):
                    
        super(HyperAttn, self).__init__()
        
        l = []
        if num_layer == 1:
            l.append(HyperAttnLayer(input_vdim, input_edim, query_dim, output_vdim, output_edim, weight_dim, dropout=0.))
        else:
            for i in range(num_layer):
                if i == 0:
                    l.append(HyperAttnLayer(input_vdim, input_edim, query_dim, hidden_dim, hidden_dim, weight_dim, dropout=0.))
                elif i == (num_layer - 1):
                    l.append(HyperAttnLayer(hidden_dim, hidden_dim, query_dim, output_vdim, output_edim, weight_dim, dropout=dropout))
                else:
                    l.append(HyperAttnLayer(hidden_dim, hidden_dim, query_dim, hidden_dim, hidden_dim, weight_dim, dropout=dropout))
        self.layers = nn.ModuleList(l)
        self.device = device
        
    def forward(self, blocks, vfeat, efeat):
        for i, layer in enumerate(self.layers):
            vfeat, efeat = layer(blocks[2*i], blocks[2*i+1], vfeat, efeat)
        
        return vfeat, efeat
    