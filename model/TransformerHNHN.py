import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import os
# Based on SetTransformer
from model.Transformer import MAB, SAB, ISAB, PMA, IMAB

class TransformerHNHNLayer(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 output_vdim, 
                 output_edim,
                 weight_dim,
                 num_imab_layers=1,
                 num_isab_layers=1,
                 num_mab_layers=0,
                 num_pma_layers=0,
                 num_sab_layers=0,
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
        self.output_vdim = output_vdim
        self.output_edim = output_edim
        self.weight_dim = weight_dim
        self.num_imab_layers = num_imab_layers
        self.num_isab_layers = num_isab_layers
        self.num_mab_layers = num_mab_layers
        self.num_pma_layers = num_pma_layers
        self.num_sab_layers = num_sab_layers
        self.activation = activation
        self.dropout = dropout
        self.lnflag = ln
        
        # when ranking input is exist,
        if num_imab_layers > 0:
            assert weight_dim > 0
        
        # For Node -> Hyperedge
        #     encoding part: create new node representation specialized in hyperedge
        self.enc_v = nn.ModuleList()
        for i in range(num_imab_layers):
            if i == 0:
                self.enc_v.append(IMAB(weight_dim, input_vdim, dim_hidden, num_heads, num_inds, ln=ln))
            else:
                self.enc_v.append(IMAB(weight_dim, dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        for i in range(num_isab_layers):
            if num_imab_layers == 0 and i == 0:
                self.enc_v.append(ISAB(input_vdim, dim_hidden, num_heads, num_inds, ln=ln)) # input_vdim, dim_hidden, num_heads, num_inds, ln=ln
            else:
                self.enc_v.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)) 
        if len(self.enc_v) > 0:
            self.enc_v.append(nn.Dropout(dropout))
        #     decoding part: aggregate embeddings
        self.dec_v = nn.ModuleList()
        if num_mab_layers > 0:
            assert num_pma_layers == 0 # use hedge or node's own feature
            if (num_imab_layers + num_isab_layers) == 0:
                self.dec_v.append(MAB(input_edim, input_vdim, dim_hidden, num_heads, ln=ln))
            else:
                self.dec_v.append(MAB(input_edim, dim_hidden, dim_hidden, num_heads, ln=ln)) # input_edim, dim_hidden, dim_hidden, num_heads, ln=ln
        elif num_pma_layers > 0: # use variable
            if (num_imab_layers + num_isab_layers) == 0:
                self.dec_v.append(PMA(input_vdim,  dim_hidden, num_heads, 1, ln=ln, numlayers=num_pma_layers))
            else:
                self.dec_v.append(PMA(dim_hidden, dim_hidden, num_heads, 1, ln=ln, numlayers=num_pma_layers))
        # else: use average
        for i in range(num_sab_layers):
            self.dec_v.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.dec_v.append(nn.Dropout(dropout))
        self.dec_v.append(nn.Linear(dim_hidden, output_edim))
        
        # Use HNHN
        self.ev_lin = torch.nn.Linear(output_edim, output_vdim)
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.ev_lin.weight, gain=gain)
        
    def message_func(self, edges):
        if self.weight_dim > 0: # weight represents ranking information
            return {'q': edges.dst['feat'], 'v': edges.src['feat'], 'weight': edges.data['weight']}
        else:
            return {'q': edges.dst['feat'], 'v': edges.src['feat']}

    def v_reduce_func(self, nodes):
        # nodes.mailbox['v' or 'q'].shape = (num batch, hyperedge size, feature dim)
        Q = nodes.mailbox['q'][:,0:1,:] # <- Because nodes.mailbox['q'][i,j] == nodes.mailbox['q'][i,j+1] 
        v = nodes.mailbox['v']
        if self.weight_dim > 0:
            W = nodes.mailbox['weight']
        else:
            W = None
            
        # Encode
        for i, layer in enumerate(self.enc_v):
            if i < self.num_imab_layers:
                v = layer(W, v)
            else:
                v = layer(v)
        # Decode
        if self.num_mab_layers == 0 and self.num_pma_layers == 0:
            o = torch.mean(v, dim=1, keepdim=True)
        for i, layer in enumerate(self.dec_v):
            if i == 0 and self.num_mab_layers > 0:
                o = layer(Q, v)
            elif i == 0 and self.num_pma_layers > 0:
                o = layer(v)
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
                 num_imab_layers=1,
                 num_isab_layers=1,
                 num_mab_layers=0,
                 num_pma_layers=0,
                 num_sab_layers=0,
                 layernorm = False,
                 dropout=0.6,
                 activation=F.relu):
        super(TransformerHNHN, self).__init__()
        self.num_layers = num_layers
        self.activation = activation
        
        self.layers = nn.ModuleList()               
        if num_layers == 1:
             self.layers.append(model(input_vdim, input_edim, vertex_dim, edge_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                      num_imab_layers=num_imab_layers, num_isab_layers=num_isab_layers, 
                                      num_mab_layers=num_mab_layers, num_pma_layers=num_pma_layers, num_sab_layers=num_sab_layers,
                                      dropout=dropout, ln=layernorm, activation=None))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(model(input_vdim, input_edim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             num_imab_layers=num_imab_layers, num_isab_layers=num_isab_layers, 
                                             num_mab_layers=num_mab_layers, num_pma_layers=num_pma_layers, num_sab_layers=num_sab_layers,
                                             dropout=dropout, ln=layernorm, activation=self.activation))
                elif i == (num_layers - 1):
                    self.layers.append(model(hidden_dim, hidden_dim, vertex_dim, edge_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             num_imab_layers=num_imab_layers, num_isab_layers=num_isab_layers, 
                                             num_mab_layers=num_mab_layers, num_pma_layers=num_pma_layers, num_sab_layers=num_sab_layers, 
                                             dropout=dropout, ln=layernorm, activation=None))
                else:
                    self.layers.append(model(hidden_dim, hidden_dim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             num_imab_layers=num_imab_layers, num_isab_layers=num_isab_layers, 
                                             num_mab_layers=num_mab_layers, num_pma_layers=num_pma_layers, num_sab_layers=num_sab_layers,
                                             dropout=dropout, ln=layernorm, activation=self.activation))
    
    def forward(self, blocks, vfeat, efeat, e_reg_weight, v_reg_sum):
        for l in range(self.num_layers):
            vfeat, efeat = self.layers[l](blocks[2*l], blocks[2*l+1], vfeat, efeat, e_reg_weight, v_reg_sum)
            
        return vfeat, efeat