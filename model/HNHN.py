import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import time
            
class HNHNLayer(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 vertex_dim, 
                 edge_dim, 
                 use_efeat=False,
                 dropout=0.5, 
                 activation=F.relu):
        super(HNHNLayer, self).__init__()
        
        self.use_efeat = use_efeat
        self._edge_dim = edge_dim
        self.ve_lin = torch.nn.Linear(input_vdim, edge_dim)
        self.ev_lin = torch.nn.Linear(edge_dim, vertex_dim)
        if  self.use_efeat:
            self.efeat_lin = torch.nn.Linear(input_edim, edge_dim)
            
        self.dropout = dropout
        self.activation = activation
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.ve_lin.weight, gain=gain)
        nn.init.xavier_normal_(self.ev_lin.weight, gain=gain)
        if hasattr(self,'efeat_lin'):
            nn.init.xavier_normal_(self.efeat_lin.weight, gain=gain)
        
    def weight_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']
        return {'weight': weight}
    
    def message_func(self, edges):
        return {'Wh': edges.src['Wh'], 'weight': edges.data['weight']}

    def reduce_func(self, nodes):
        Weight = nodes.mailbox['weight']
        aggr = torch.sum(Weight * nodes.mailbox['Wh'], dim=1)
        return {'h': aggr}

    def forward(self, g1, g2, vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum):
        with g1.local_scope():
            given_efeat = efeat
            # edge aggregation
            g1.srcnodes['node'].data['h'] = vfeat
            g1.srcnodes['node'].data['Wh'] = vfeat
            g1.srcnodes['node'].data['reg_weight'] = v_reg_weight[:g1['in'].num_src_nodes()]
            g1.srcnodes['node'].data['reg_sum'] = v_reg_sum[:g1['in'].num_src_nodes()]
            g1.dstnodes['edge'].data['reg_weight'] = e_reg_weight[:g1['in'].num_dst_nodes()]
            g1.dstnodes['edge'].data['reg_sum'] = e_reg_sum[:g1['in'].num_dst_nodes()]
            g1.apply_edges(self.weight_fn, etype='in')
            g1.update_all(self.message_func, self.reduce_func, etype='in')
            norm_vfeat = g1.dstnodes['edge'].data['h']
            if self.activation is not None:
                efeat = self.activation(self.ve_lin(norm_vfeat))
            else:
                efeat = self.ve_lin(norm_vfeat)
            if self.use_efeat:
                emb = self.efeat_lin(given_efeat)
                efeat = efeat + emb
        with g2.local_scope():
            # node aggregattion
            g2.srcnodes['edge'].data['Wh'] = efeat
            g2.srcnodes['edge'].data['reg_weight'] = e_reg_weight[:g2['con'].num_src_nodes()]
            g2.srcnodes['edge'].data['reg_sum'] = e_reg_sum[:g2['con'].num_src_nodes()]
            g2.dstnodes['node'].data['reg_weight'] = v_reg_weight[:g2['con'].num_dst_nodes()]
            g2.dstnodes['node'].data['reg_sum'] = v_reg_sum[:g2['con'].num_dst_nodes()]
            g2.apply_edges(self.weight_fn, etype='con')
            g2.update_all(self.message_func, self.reduce_func, etype='con')
            norm_efeat = g2.dstnodes['node'].data['h']
            if self.activation is not None:
                vfeat = self.activation(self.ev_lin(norm_efeat))
            else:
                vfeat = self.ev_lin(norm_efeat)
            vfeat = F.dropout(vfeat, self.dropout, training=self.training)
            
            return [vfeat, efeat]
        
class HNHN(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 hidden_dim, 
                 output_vdim, 
                 output_edim, 
                 use_efeat=False,
                 num_layer=2, 
                 dropout=0.5, 
                 device='0'):
                    
        super(HNHN, self).__init__()
        
        self.activation = F.relu
        l = []
        if num_layer == 1:
            l.append(HNHNLayer(input_vdim, input_edim, output_vdim, output_edim, use_efeat=use_efeat, dropout=0., activation=self.activation))
        else:
            for i in range(num_layer):
                if i == 0:
                    l.append(HNHNLayer(input_vdim, input_edim, hidden_dim, hidden_dim, use_efeat=use_efeat, dropout=0., activation=self.activation))
                elif i == (num_layer - 1):
                    l.append(HNHNLayer(hidden_dim, 0, output_vdim, output_edim, use_efeat=False, dropout=dropout, activation=self.activation))
                else:
                    l.append(HNHNLayer(hidden_dim, 0, hidden_dim, hidden_dim, use_efeat=False, dropout=dropout, activation=self.activation))
        self.layers = nn.ModuleList(l)
        self.device = device
        
    def forward(self, blocks, vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum):
        for i, layer in enumerate(self.layers):
            vfeat, efeat = layer(blocks[2*i], blocks[2*i+1], vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum)
        
        return vfeat, efeat
        