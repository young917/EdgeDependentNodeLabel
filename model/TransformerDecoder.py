import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import os

from Transformer import MAB, SAB, ISAB, PMA

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 dim_hidden = 128,
                 num_isab_layers=2,
                 num_mab_layers=0,
                 num_pma_layers=0,
                 num_sab_layers=0,
                 num_heads=4, 
                 num_inds=4,
                 dropout=0.6,
                 ln=False,
                 activation=F.relu):
        super(TransformerDecoder, self).__init__()
        
        self.num_heads = num_heads
        self.num_inds = num_inds
        
        self.input_dim = input_dim
        self.hidden_dim = dim_hidden
        self.output_dim = output_dim
        
        self.num_isab_layers = num_isab_layers
        self.num_mab_layers = num_mab_layers
        self.num_pma_layers = num_pma_layers
        self.num_sab_layers = num_sab_layers
        self.dropout = dropout
        self.lnflag = ln
        
        self.layers = nn.ModuleList()
        if num_mab_layers > 0:
            assert num_pma_layers == 0 # use hedge or node's own feature
            self.layers.append(MAB(input_dim, dim_hidden, dim_hidden, num_heads, ln=ln))
        elif num_pma_layers > 0: # use variable
            self.layers.append(PMA(dim_hidden, num_heads, 1, ln=ln))
        for i in range(num_sab_layers):
            self.layers.append(SAB(dim_hidden, dim_hidden, num_heads, ln=ln))
        self.layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Linear(dim_hidden, output_dim))
        
    def message_func(self, edges):
        if self.weight_dim > 0: # weight represents ranking information
            return {'q': edges.dst['feat'], 'v': edges.src['feat'], 'weight': edges.data['weight']}
        else:
            return {'q': edges.dst['feat'], 'v': edges.src['feat']}

    def v_reduce_func(self, nodes):
        # nodes.mailbox['v' or 'q'].shape = (num batch, hyperedge size, feature dim)
        Q = nodes.mailbox['q'][:,0:1,:] # <- Because nodes.mailbox['q'][i,j] == nodes.mailbox['q'][i,j+1] 
        V = nodes.mailbox['v']
        
        # Encode
        if self.weight_dim > 0: # sum ranking vector
            emb_w = self.fc_wv(nodes.mailbox['weight'])
            v = V + emb_w
        else:
            v = V
        for i, layer in enumerate(self.enc_v):
            v = layer(v)
        # Decode
        if self.num_mab_layers == 0 and self.num_pma_layers == 0:
            o = torch.mean(v, dim=1, keepdim=True)
        for i, layer in enumerate(self.dec_v):
            if i == 0:
                if self.num_mab_layers > 0:
                    o = layer(Q, v)
                elif self.num_pma_layers > 0:
                    o = layer(v)
            else:
                o = layer(o)
        # dropout -> FF
        if self.residualflag:
            if self.input_edim != self.output_edim:
                o = self.forward_layer_e(Q) + self.ln_v(o)
            else:
                o = Q + self.ln_e(o)
        
        return {'o': o}
    
    def e_reduce_func(self, nodes):
        Q = nodes.mailbox['q'][:,0:1,:]
        V = nodes.mailbox['v']
        # Encode
        if self.weight_dim > 0: # sum ranking vector
            emb_w = self.fc_we(nodes.mailbox['weight'])
            v = V + emb_w
        else:
            v = V
        for layer in self.enc_e:
            v = layer(v)
        # Decode
        if self.num_mab_layers == 0 and self.num_pma_layers == 0:
            o = torch.mean(v, dim=1, keepdim=True)
        for i, layer in enumerate(self.dec_e):
            if i == 0:
                if self.num_mab_layers > 0:
                    o = layer(Q, v)
                elif self.num_pma_layers > 0:
                    o = layer(v)
            else:
                o = layer(o)
        # dropout -> FF
        if self.residualflag:
            if self.input_vdim != self.output_vdim:
                o = self.forward_layer_v(Q) + self.ln_v(o)
            else:
                o = Q + self.ln_v(o)
                
        return {'o': o}
    
    
    def forward(self, g, vfeat, efeat, flag):
        if flag == "node":
            with g.local_scope():
                g.srcnodes['node'].data['feat'] = vfeat
                g.dstnodes['edge'].data['feat'] = efeat[:g['in'].num_dst_nodes()]
                g.update_all(self.message_func, self.v_reduce_func, etype='in')
                efeat = g1.dstnodes['edge'].data['o']
                efeat = efeat.squeeze(1)
                
        elif flag == "edge":
            with g.local_scope():
                g.srcnodes['edge'].data['feat'] = efeat
                g.dstnodes['node'].data['feat'] = vfeat[:g['con'].num_dst_nodes()]
                g.update_all(self.message_func, self.e_reduce_func, etype='con')
                vfeat = g.dstnodes['node'].data['o']
                vfeat = feat.squeeze(1)
            
        return vfeat, efeat
    