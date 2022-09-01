import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import os
# Based on SetTransformer

import dgl.function as fn
import dgl.ops.edge_softmax as edge_softmax
from model.Transformer import MAB, SAB, ISAB, PMA
    
# ============================ Transformer Layer =================================
class TransformerAttLayer(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 output_vdim, 
                 output_edim,
                 weight_dim,
                 dim_hidden = 128,
                 num_isab_layers=2,
                 num_heads=4, 
                 num_inds=4,
                 dropout=0.6,
                 ln=False,
                 activation=None,
                 partial=False):
        super(TransformerAttLayer, self).__init__()
        
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.partial = partial
        
        self.input_vdim = input_vdim
        self.input_edim = input_edim
        self.hidden_dim = dim_hidden
        self.output_vdim = output_vdim
        self.output_edim = output_edim
        self.weight_dim = weight_dim
        self.num_isab_layers = num_isab_layers
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        # Node -> Edge : Encode
        self.enc_v = nn.ModuleList()
        for i in range(num_isab_layers):
            if i == 0:
                self.enc_v.append(ISAB(input_vdim, dim_hidden, num_heads, num_inds, ln=ln))
            else:
                self.enc_v.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.enc_v.append(nn.Dropout(dropout))
        # Node -> Edge : Decode
        self.fc_ke = nn.Linear(dim_hidden, output_edim * num_heads, bias=False) # vertex
        self.fc_qe = nn.Linear(input_edim, output_edim * num_heads, bias=False) # edge
        self.attn_ke = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_edim)))
        self.attn_qe = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_edim)))
        self.bias_e = nn.Parameter(torch.FloatTensor(size=(num_heads * output_edim,)))
        if weight_dim > 0:
            self.fc_wv = nn.Linear(weight_dim, input_vdim)
            
        # Edge -> Node
        if self.partial:
            # Encode
            self.enc_e = nn.ModuleList()
            for i in range(num_isab_layers):
                if i == 0:
                    self.enc_e.append(ISAB(output_edim, dim_hidden, num_heads, num_inds, ln=ln))
                else:
                    self.enc_e.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))   
            self.enc_e.append(nn.Dropout(dropout))
            # Decode
            self.fc_kv = nn.Linear(dim_hidden, output_vdim * num_heads, bias=False) # edge
            self.fc_qv = nn.Linear(input_vdim, output_vdim * num_heads, bias=False) # vertex
            self.attn_kv = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
            self.attn_qv = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
            self.bias_v = nn.Parameter(torch.FloatTensor(size=(num_heads * output_edim,)))
            if weight_dim > 0:
                self.fc_we = nn.Linear(weight_dim, output_edim)
        else:
            # Use HNHN
            self.ev_lin = torch.nn.Linear(output_edim, output_vdim)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_ke.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_qe.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_kv.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_qv.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_ke, gain=gain)
        nn.init.xavier_normal_(self.attn_qe, gain=gain)
        nn.init.xavier_normal_(self.attn_kv, gain=gain)
        nn.init.xavier_normal_(self.attn_qv, gain=gain)
        nn.init.constant_(self.bias_e, 0.0)
        nn.init.constant_(self.bias_v, 0.0)
        if self.weight_dim > 0:
            nn.init.xavier_normal_(self.fc_wv.weight, gain=gain)
        if self.partial:
            nn.init.xavier_normal_(self.ev_lin.weight, gain=gain)
            if self.weight_dim > 0:
                nn.init.xavier_normal_(self.fc_we.weight, gain=gain)
        
    def message_func(self, edges):
        if self.weight_dim > 0:
            return {'q': edges.dst['feat'], 'v': edges.src['feat'], 'weight': edges.data['weight']}
        else:
            return {'q': edges.dst['feat'], 'v': edges.src['feat']}
    
    def attention(self, edges):
        if self._weight_dim > 0:
            e = edges.src['el'] + edges.dst['er'] + edges.data['ew']
        else:
            e = edges.src['el'] + edges.dst['er']  
        
        return {'e': e}
    
    def v_self_attn(self, nodes):
        Q = nodes.mailbox['q'][:,0:1,:] # from precious layer
        V = nodes.mailbox['v']
        # Encode
        if self.weight_dim > 0:
            emb_w = self.fc_wv(nodes.mailbox['weight'])
            v = V + emb_w
        else:
            v = V
        for i, layer in enumerate(self.enc_v):
            v = layer(v)
        # Decode
        
        return {'o': v}
    
    def e_self_attn(self, nodes):
        Q = nodes.mailbox['q'][:,0:1,:] # from precious layer
        V = nodes.mailbox['v'] 
        if self.weight_dim > 0:
            emb_w = self.fc_we(nodes.mailbox['weight'])
            v = V + emb_w
        else:
            v = V
            
        for layer in self.enc_e:
            v = layer(v)
        
        return {'o': v}
    
    def forward(self, g1, g2, vfeat, efeat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum):
        with g1.local_scope():
            print("First", vfeat.shape)
            vfeat = self.dropout(vfeat)
            efeat = self.dropout(efeat)

            # Encode
            g1.srcnodes['node'].data['feat'] = vfeat
            g1.dstnodes['edge'].data['feat'] = efeat
            g1.update_all(self.message_func, self.v_self_attn, etype='in')
            newvfeat = g.ndata['o']['edge']
            newvfeat = newvfeat.squeeze(1)
            # Decode - Attention
            vfeat_ke = self.fc_ke(newvfeat).view(-1, self._num_heads, self._output_edim)
            efeat_qe = self.fc_qe(efeat).view(-1, self._num_heads, self._output_edim)
            el_e = (vfeat_ke * self.attn_ke).sum(dim=-1).unsqueeze(-1)
            er_e = (efeat_qe * self.attn_qe).sum(dim=-1).unsqueeze(-1)
            g['in'].srcdata.update({'ft': vfeat_ke, 'el': el_e})
            g['in'].dstdata.update({'er': er_e})
            g.apply_edges(self.attention, etype='in')
            e = F.relu(g['in'].edata.pop('e'))
            g['in'].edata['a'] = self.attn_drop(edge_softmax(g['in'], e))
            g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='in')
            efeat = g['in'].dstdata['ft']
            efeat = efeat + self.bias_e.view(1, self._num_heads, self._output_edim)
            if self.activation:
                efeat = self.activation(efeat)
            efeat = efeat.mean(1)

            # Encode
            g.ndata['feat'] = {'node': vfeat, 'edge': efeat}
            g.update_all(self.message_func, self.e_reduce_func, etype='con')
            newefeat = g.ndata['o']['node']
            newefeat = newefeat.squeeze(1)
            # Decode - Attention
            efeat_kv = self.fc_kv(self.feat_drop(newefeat)).view(-1, self._num_heads, self._output_vdim)
            vfeat_qv = self.fc_qv(vfeat).view(-1, self._num_heads, self._output_vdim)
            el_v = (efeat_kv * self.attn_kv).sum(dim=-1).unsqueeze(-1)
            er_v = (vfeat_qv * self.attn_qv).sum(dim=-1).unsqueeze(-1)
            # self.curoutputname = self.outputname_v
            g['con'].srcdata.update({'ft': efeat_kv, 'el': el_v})
            g['con'].dstdata.update({'er': er_v})
            if hasattr(self, 'attn_wv') and hasattr(self, 'fc_wv'):
                wv = g['con'].edata["weight"]
                feat_wv = self.fc_wv(wv).view(-1, self._num_heads, self._output_vdim)
                ew_v = (feat_wv * self.attn_wv).sum(dim=-1).unsqueeze(-1)
                g['con'].edata['ew'] = ew_v
            g.apply_edges(self.attention, etype='con')
            e = F.relu(g['con'].edata.pop('e'))
            g['con'].edata['a'] = self.attn_drop(edge_softmax(g['con'], e))
            g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='con')
            vfeat = g['con'].dstdata['ft']
            vfeat = vfeat + self.bias_v.view(1, self._num_heads, self._output_vdim)
            if self.activation:
                vfeat = self.activation(vfeat)
            vfeat = vfeat.mean(1)

        return [vfeat, efeat]
    
# ============================ Transformer Layer Rank Query ===============================
class TransformerAttLayerRankQ(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 output_vdim, 
                 output_edim,
                 weight_dim,
                 dim_hidden = 128,
                 num_isab_layers=1,
                 num_heads=4, 
                 num_inds=4,
                 dropout=0.6,
                 ln=False,
                 activation=None):
        super(TransformerAttLayerRankQ, self).__init__()
        
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.input_vdim = input_vdim
        self.input_edim = input_edim
        self.output_vdim = output_vdim
        self.output_edim = output_edim
        self.weight_dim = weight_dim
        assert weight_dim > 0
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        
        self.enc_v = nn.ModuleList() # MAB -> ISAB
        self.enc_e = nn.ModuleList() # MAB -> ISAB
        self.enc_v.append(MAB(weight_dim, input_vdim, dim_hidden, num_heads, ln=ln))
        self.enc_e.append(MAB(weight_dim, output_edim, dim_hidden, num_heads, ln=ln))
        for i in range(num_isab_layers):
            self.enc_v.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
            self.enc_e.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)) 
        self.enc_v.append(self.dropout)
        self.enc_e.append(self.dropout)
        
        self.fc_ke = nn.Linear(input_vdim, output_edim * num_heads, bias=False) # vertex
        self.fc_qe = nn.Linear(input_edim, output_edim * num_heads, bias=False) # edge
        self.attn_ke = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_edim)))
        self.attn_qe = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_edim)))
        self.bias_e = nn.Parameter(torch.FloatTensor(size=(num_heads * output_edim,)))
        
        self.fc_kv = nn.Linear(output_edim, output_vdim * num_heads, bias=False) # edge
        self.fc_qv = nn.Linear(input_vdim, output_vdim * num_heads, bias=False) # vertex
        self.attn_kv = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
        self.attn_qv = nn.Parameter(torch.FloatTensor(size=(1, num_heads, output_vdim)))
        self.bias_v = nn.Parameter(torch.FloatTensor(size=(num_heads * output_edim,)))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc_ke.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_qe.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_kv.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_qv.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_ke, gain=gain)
        nn.init.xavier_normal_(self.attn_qe, gain=gain)
        nn.init.xavier_normal_(self.attn_kv, gain=gain)
        nn.init.xavier_normal_(self.attn_qv, gain=gain)
        nn.init.constant_(self.bias_e, 0.0)
        nn.init.constant_(self.bias_v, 0.0)
        
    def attention(self, edges):
        if self._weight_dim > 0:
            e = edges.src['el'] + edges.dst['er'] + edges.data['ew']
        else:
            e = edges.src['el'] + edges.dst['er']  
        
        return {'e': e}
    
    def message_func(self, edges):
        if self.weight_dim > 0:
            return {'q': edges.dst['feat'], 'v': edges.src['feat'], 'weight': edges.data['weight']}
        else:
            return {'q': edges.dst['feat'], 'v': edges.src['feat']}

    def v_reduce_func(self, nodes):
        Q = nodes.mailbox['q'][:,0:1,:]
        v = nodes.mailbox['v']
        W = nodes.mailbox['weight']
        # Encode
        for i, layer in enumerate(self.enc_v):
            if i == 0:
                v = layer(W, v)
            else:
                v = layer(v)

        return {'o': v}
    
    def e_reduce_func(self, nodes):
        Q = nodes.mailbox['q'][:,0:1,:]
        v = nodes.mailbox['v']
        W = nodes.mailbox['weight']
        for i, layer in enumerate(self.enc_e):
            if i == 0:
                v = layer(W,v)
            else:
                v = layer(v)
        
        return {'o': v}
    
    def forward(self, g, vfeat, efeat):
        with g.local_scope():
            # Encode with Rank
            g.ndata['feat'] = {'node': vfeat, 'edge': efeat}
            g.update_all(self.message_func, self.v_reduce_func, etype='in')
            vfeatrank = g.ndata['o']['edge']
            vfeatrank = vfeatrank.squeeze(1)
            # Decode - Attention
            vfeat_ke = self.fc_ke(vfeatrank).view(-1, self._num_heads, self._output_edim)
            efeat_qe = self.fc_qe(efeat).view(-1, self._num_heads, self._output_edim)
            el_e = (vfeat_ke * self.attn_ke).sum(dim=-1).unsqueeze(-1)
            er_e = (efeat_qe * self.attn_qe).sum(dim=-1).unsqueeze(-1)
            g['in'].srcdata.update({'ft': vfeat_ke, 'el': el_e})
            g['in'].dstdata.update({'er': er_e})
            g.apply_edges(self.attention, etype='in')
            e = F.relu(g['in'].edata.pop('e'))
            g['in'].edata['a'] = self.attn_drop(edge_softmax(g['in'], e))
            g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='in')
            efeat = g['in'].dstdata['ft']
            efeat = efeat + self.bias_e.view(1, self._num_heads, self._output_edim)
            if self.activation:
                efeat = self.activation(efeat)
            efeat = efeat.mean(1)

            # Encode with Rank
            g.ndata['feat'] = {'node': vfeat, 'edge': efeat}
            g.update_all(self.message_func, self.e_reduce_func, etype='con')
            efeatrank = g.ndata['o']['node']
            efeatrank = efeatrank.squeeze(1)
            # Decode - Attention
            efeat_kv = self.fc_kv(self.feat_drop(efeatrank)).view(-1, self._num_heads, self._output_vdim)
            vfeat_qv = self.fc_qv(vfeat).view(-1, self._num_heads, self._output_vdim)
            el_v = (efeat_kv * self.attn_kv).sum(dim=-1).unsqueeze(-1)
            er_v = (vfeat_qv * self.attn_qv).sum(dim=-1).unsqueeze(-1)
            # self.curoutputname = self.outputname_v
            g['con'].srcdata.update({'ft': efeat_kv, 'el': el_v})
            g['con'].dstdata.update({'er': er_v})
            g.apply_edges(self.attention, etype='con')
            e = F.relu(g['con'].edata.pop('e'))
            g['con'].edata['a'] = self.attn_drop(edge_softmax(g['con'], e))
            g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'), etype='con')
            vfeat = g['con'].dstdata['ft']
            vfeat = vfeat + self.bias_v.view(1, self._num_heads, self._output_vdim)
            if self.activation:
                vfeat = self.activation(vfeat)
            vfeat = vfeat.mean(1)

        return [vfeat, efeat]
    
class TransformerAtt(nn.Module): 
    def __init__(self, 
                 model,
                 input_vdim, 
                 input_edim, 
                 vertex_dim, 
                 edge_dim,
                 weight_dim,
                 num_layers,
                 hidden_dim = 128,
                 num_isab_layers=1,
                 num_heads=4, 
                 num_inds=4,
                 dropout=0.6):
        super(TransformerAtt, self).__init__()
        
        self.activation = F.relu
        self.num_layers = num_layers
        self.layers = nn.ModuleList()               
        if num_layers == 1:
             self.layers.append(model(input_vdim, input_edim, vertex_dim, edge_dim, weight_dim=weight_dim, num_isab_layers=num_isab_layers,
                                      num_heads=num_heads, num_inds=num_inds, dropout=0, activation=None))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(model(input_vdim, input_edim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_isab_layers=num_isab_layers,
                                      num_heads=num_heads, num_inds=num_inds, dropout=0, activation=self.activation))
                elif i == (num_layers - 1):
                    self.layers.append(model(hidden_dim, hidden_dim, vertex_dim, edge_dim, weight_dim=weight_dim, num_isab_layers=num_isab_layers,
                                      num_heads=num_heads, num_inds=num_inds, dropout=dropout, activation=None))
                else:
                    self.layers.append(model(hidden_dim, hidden_dim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_isab_layers=num_isab_layers,
                                      num_heads=num_heads, num_inds=num_inds, dropout=dropout, activation=self.activation))
    
    def forward(self, g, vfeat, efeat):
        for l in range(self.num_layers - 1):
            vfeat, efeat = self.layers[l](g, vfeat, efeat)
        vfeat, efeat = self.layers[-1](g, vfeat, efeat)
        return vfeat, efeat
    