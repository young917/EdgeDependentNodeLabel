import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import os
# Based on SetTransformer
from model.Transformer import MAB, SAB, ISAB, PMA

class TransformerHNHNLayer(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 output_vdim, 
                 output_edim,
                 weight_dim,
                 att_type_v = "OrderPE",
                 agg_type_v = "PrevQ",
                 num_att_layer = 2,
                 dim_hidden = 128,
                 num_heads=4, 
                 num_inds=4,
                 dropout=0.6,
                 ln=False,
                 activation=F.relu):
        super(TransformerHNHNLayer, self).__init__()
        
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.input_vdim = input_vdim
        self.input_edim = input_edim
        self.hidden_dim = dim_hidden
        self.query_dim = dim_hidden
        self.output_vdim = output_vdim
        self.output_edim = output_edim
        self.weight_dim = weight_dim
        self.att_type_v = att_type_v
        self.agg_type_v = agg_type_v
        self.num_att_layer = num_att_layer
        self.activation = activation
        self.dropout = dropout
        self.lnflag = ln
        
        if self.att_type_v == "OrderPE":
            self.pe_v = nn.Linear(weight_dim, input_vdim)
        self.dropoutlayer = nn.Dropout(dropout)
        
        # For Node -> Hyperedge
        #     Attention Part
        dimension = input_vdim
        self.enc_v = nn.ModuleList()
        for _ in range(self.num_att_layer):
            if self.att_type_v != "NoAtt":
                self.enc_v.append(ISAB(dimension, dim_hidden, num_heads, num_inds, ln=ln))
                dimension = dim_hidden
        #     Aggregate Part
        self.dec_v = nn.ModuleList()
        if self.agg_type_v == "PrevQ":
            self.dec_v.append(MAB(input_edim, dimension, dim_hidden, num_heads, ln=ln))
        elif self.agg_type_v == "pure":
            self.dec_v.append(PMA(dimension,  dim_hidden, num_heads, 1, ln=ln, numlayers=1))
        elif self.agg_type_v == "pure2":
            self.dec_v.append(PMA(dimension,  dim_hidden, num_heads, 1, ln=ln, numlayers=2))
        self.dec_v.append(nn.Dropout(self.dropout))
        self.dec_v.append(nn.Linear(dim_hidden, output_edim))
        
        # Use HNHN
        self.ev_lin = torch.nn.Linear(output_edim, output_vdim)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.ev_lin.weight, gain=gain)
        
    def message_func(self, edges):
        if self.weight_dim > 0: # weight represents positional information
            return {'q': edges.dst['feat'], 'v': edges.src['feat'], 'weight': edges.data['weight']}
        else:
            return {'q': edges.dst['feat'], 'v': edges.src['feat']}

    def v_reduce_func(self, nodes):
        Q = nodes.mailbox['q'][:,0:1,:]
        v = nodes.mailbox['v']
        if self.weight_dim > 0:
            W = nodes.mailbox['weight']
             
        # Attention
        if self.att_type_v == "OrderPE":
            v = v + self.pe_v(W)
        for i, layer in enumerate(self.enc_v):
            v = layer(v)
        v = self.dropoutlayer(v)
        # Aggregate
        o = v
        for i, layer in enumerate(self.dec_v):
            if i == 0 and self.agg_type_v == "PrevQ":
                o = layer(Q, o)
            else:
                o = layer(o)
        return {'o': o}
    
    def weight_hnhn_fn(self, edges):
        weight = edges.src['reg_weight']/edges.dst['reg_sum']
        return {'weight_HNHN': weight}
    
    def e_message_hnhn_func(self, edges):
        return {'feat': edges.src['feat'], 'weight_HNHN': edges.data['weight_HNHN']}

    def e_reduce_hnhn_func(self, nodes):
        Weight = nodes.mailbox['weight_HNHN']
        aggr = torch.sum(Weight * nodes.mailbox['feat'], dim=1)
        return {'o': aggr}
    
    def forward(self, g1, g2, vfeat, efeat, e_reg_weight, v_reg_sum):
        with g1.local_scope():
            g1.srcnodes['node'].data['feat'] = vfeat
            g1.dstnodes['edge'].data['feat'] = efeat[:g1['in'].num_dst_nodes()]
            g1.update_all(self.message_func, self.v_reduce_func, etype='in')
            efeat = g1.dstnodes['edge'].data['o']
            efeat = efeat.squeeze(1)
        with g2.local_scope():
            g2.srcnodes['edge'].data['feat'] = efeat
            g2.dstnodes['node'].data['feat'] = vfeat[:g2['con'].num_dst_nodes()]
            g2.srcnodes['edge'].data['reg_weight'] = e_reg_weight[:g2['con'].num_src_nodes()] 
            #g2.dstnodes['node'].data['reg_weight'] = v_reg_weight[:g2['con'].num_dst_nodes()]
            #g2.srcnodes['edge'].data['reg_sum'] = e_reg_sum[:g2['con'].num_src_nodes()] 
            g2.dstnodes['node'].data['reg_sum'] = v_reg_sum[:g2['con'].num_dst_nodes()]
            g2.apply_edges(self.weight_hnhn_fn, etype='con')
            g2.update_all(self.e_message_hnhn_func, self.e_reduce_hnhn_func, etype='con')
            norm_efeat = g2.dstnodes['node'].data['o']
            vfeat = self.ev_lin(norm_efeat)
            if self.activation:
                vfeat = self.activation(vfeat)
            vfeat = F.dropout(vfeat, self.dropout, training=self.training)
        
        return [vfeat, efeat]
    
    
# ============================ Transformer ===============================
class TransformerHNHN(nn.Module): 
    def __init__(self, 
                 model,
                 input_vdim,
                 input_edim,
                 hidden_dim, 
                 vertex_dim,
                 edge_dim,
                 weight_dim=0,
                 num_layers=2,
                 num_heads=4,
                 num_inds=4,
                 att_type_v = "OrderPE",
                 agg_type_v = "PrevQ",
                 num_att_layer = 2,
                 layernorm = False,
                 dropout=0.6,
                 activation=F.relu):
        super(TransformerHNHN, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        
        self.layers = nn.ModuleList()               
        if num_layers == 1:
             self.layers.append(model(input_vdim, input_edim, vertex_dim, edge_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                      att_type_v=att_type_v, agg_type_v=agg_type_v, num_att_layer=num_att_layer,
                                      dropout=dropout, ln=layernorm, activation=None))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(model(input_vdim, input_edim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             att_type_v=att_type_v, agg_type_v=agg_type_v, num_att_layer=num_att_layer,
                                             dropout=dropout, ln=layernorm, activation=self.activation))
                elif i == (num_layers - 1):
                    self.layers.append(model(hidden_dim, hidden_dim, vertex_dim, edge_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             att_type_v=att_type_v, agg_type_v=agg_type_v, num_att_layer=num_att_layer,
                                             dropout=dropout, ln=layernorm, activation=None))
                else:
                    self.layers.append(model(hidden_dim, hidden_dim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             att_type_v=att_type_v, agg_type_v=agg_type_v, num_att_layer=num_att_layer,
                                             dropout=dropout, ln=layernorm, activation=self.activation))
    
    def forward(self, blocks, vfeat, efeat, e_reg_weight, v_reg_sum):
        for l in range(self.num_layers):
            vfeat, efeat = self.layers[l](blocks[2*l], blocks[2*l+1], vfeat, efeat, e_reg_weight, v_reg_sum)
            
        return vfeat, efeat