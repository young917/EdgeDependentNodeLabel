import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pdb
import math
import os
import time

# Based on SetTransformer https://github.com/juho-lee/set_transformer

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False, numlayers=1, activation=F.relu):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.numlayers = numlayers
        self.fc_o = nn.ModuleList()
        for _ in range(numlayers):
            self.fc_o.append(nn.Linear(dim_V, dim_V))
        self.activation = activation
        
    def forward(self, Q, K, Kpos=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(torch.matmul(Q_, torch.transpose(K_, 1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + torch.matmul(A, V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        resO = O
        for i, lin in enumerate(self.fc_o[:-1]):
            O = self.activation(lin(O), inplace=True)
        O = resO + self.activation(self.fc_o[-1](O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    # use for ablation study of positional encodings
    def __init__(self, dim_in, dim_out, num_heads, ln=False, RE="None"):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln, RE=RE)
    def forward(self, X, Xpos=None):
        out = self.mab(X, X, Xpos)
        return out

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False, activation=F.relu):
        super(ISAB, self).__init__()
        # X is dim_in, I is (num_inds) * (dim_out)
        # After mab0, I is represented by X => H = (num_inds) * (dim_out)
        # After mab1, X is represented by H => X' = (X.size[1]) * (dim_in)
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln, activation=activation)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln, activation=activation)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_seeds, ln=False, numlayers=1):
        # (num_seeds, dim) is represented by X
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim_out))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln,  numlayers=numlayers)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

# ============================ Transformer Layer =================================
class WhatsnetLSPELayer(nn.Module):
    def __init__(self, 
                 input_vdim, 
                 input_edim, 
                 output_vdim, 
                 output_edim,
                 weight_dim,
                 att_type_v = "OrderPE",
                 agg_type_v = "PrevQ",
                 att_type_e = "OrderPE",
                 agg_type_e = "PrevQ",
                 num_att_layer = 2,
                 dim_hidden = 128,
                 num_heads=4, 
                 num_inds=4,
                 dropout=0.6,
                 ln=False,
                 activation=F.relu):
        super(WhatsnetLSPELayer, self).__init__()
        
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.input_vdim = input_vdim
        self.input_edim = input_edim
        self.hidden_dim = dim_hidden
        self.output_vdim = output_vdim
        self.output_edim = output_edim
        self.weight_dim = weight_dim
        self.att_type_v = att_type_v
        self.agg_type_v = agg_type_v
        self.att_type_e = att_type_e
        self.agg_type_e = agg_type_e
        self.num_att_layer = num_att_layer
        self.activation = activation
        self.dropout = dropout
        self.lnflag = ln
        
        if self.att_type_v == "OrderPE":
            self.pe_v = nn.Linear(weight_dim, input_vdim)
        if self.att_type_e == "OrderPE":
            self.pe_e = nn.Linear(weight_dim, output_edim)
        self.dropout = nn.Dropout(dropout)
            
        # For Node -> Hyperedge
        #     Attention part: create edge-dependent embedding
        dimension = input_vdim
        self.enc_v = nn.ModuleList()
        self.enc_vp = nn.ModuleList()
        for _ in range(self.num_att_layer):
            if self.att_type_v != "NoAtt":
                # if self.pe_ablation_flag:
                    # self.enc_v.append(SAB(dimension, dim_hidden, num_heads, ln=ln))
                self.enc_v.append(ISAB(dimension, dim_hidden, num_heads, num_inds, ln=ln))
                self.enc_vp.append(ISAB(self.weight_dim, self.weight_dim, num_heads, num_inds, ln=ln, activation=F.tanh))
                dimension = dim_hidden
        #     Aggregate part
        self.dec_v = nn.ModuleList()
        self.dec_vp = nn.ModuleList()
        if self.agg_type_v == "PrevQ":
            self.dec_v.append(MAB(input_edim, dimension, dim_hidden, num_heads, ln=ln))
            self.dec_vp.append(MAB(self.weight_dim, self.weight_dim, self.weight_dim, num_heads, ln=ln, activation=F.tanh))
        # elif self.agg_type_v == "pure":
        #     self.dec_v.append(PMA(dimension,  dim_hidden, num_heads, 1, ln=ln, numlayers=1))
        # elif self.agg_type_v == "pure2":
        #     self.dec_v.append(PMA(dimension,  dim_hidden, num_heads, 1, ln=ln, numlayers=2))
        self.dec_v.append(nn.Dropout(dropout))
        self.dec_vp.append(nn.Dropout(dropout))
        self.dec_v.append(nn.Linear(dim_hidden, output_edim))
        self.dec_vp.append(nn.Linear(self.weight_dim, self.weight_dim))

        # For Hyperedge -> Node
        #     Attention part: create node-dependent embedding
        dimension = output_edim
        self.enc_e = nn.ModuleList()
        self.enc_ep = nn.ModuleList()
        for _ in range(self.num_att_layer):
            if self.att_type_e != "NoAtt":
                self.enc_e.append(ISAB(dimension, dim_hidden, num_heads, num_inds, ln=ln))   
                self.enc_ep.append(ISAB(self.weight_dim, self.weight_dim, num_heads, num_inds, ln=ln, activation=F.tanh))   
                dimension = dim_hidden
        #     Aggregate part
        self.dec_e = nn.ModuleList()
        self.dec_ep = nn.ModuleList()
        if self.agg_type_e == "PrevQ":
            self.dec_e.append(MAB(input_vdim, dimension, dim_hidden, num_heads, ln=ln))
            self.dec_ep.append(MAB(self.weight_dim, self.weight_dim, self.weight_dim, num_heads, ln=ln, activation=F.tanh))
        # elif self.agg_type_e == "pure":
        #     self.dec_e.append(PMA(dimension, dim_hidden, num_heads, 1, ln=ln, numlayers=1))
        # elif self.agg_type_e == "pure2":
        #     self.dec_e.append(PMA(dimension, dim_hidden, num_heads, 1, ln=ln, numlayers=2))
        self.dec_e.append(nn.Dropout(dropout))
        self.dec_ep.append(nn.Dropout(dropout))
        self.dec_e.append(nn.Linear(dim_hidden, output_vdim))
        self.dec_ep.append(nn.Linear(self.weight_dim, self.weight_dim))
        
    def v_message_func(self, edges):
        return {'q': edges.dst['feat'], 'qpos': edges.dst['pos'], 'v': edges.src['feat'], 'pos': edges.src['pos']}
        
    def e_message_func(self, edges):    
        return {'q': edges.dst['feat'], 'qpos': edges.dst['pos'], 'v': edges.src['feat'], 'pos': edges.src['pos']}
        
    def v_reduce_func(self, nodes):
        # nodes.mailbox['v' or 'q'].shape = (num batch, hyperedge size, feature dim)
        Q = nodes.mailbox['q'][:,0:1,:]
        v = nodes.mailbox['v']
        Qp = nodes.mailbox['qpos'][:,0:1,:]
        p = nodes.mailbox['pos']
            
        # Attention
        if self.att_type_v == "OrderPE":
            v = v + self.pe_v(p)
        for i, layer in enumerate(self.enc_v):
            v = layer(v)
        v = self.dropout(v)
        # Aggregate
        o = v
        for i, layer in enumerate(self.dec_v):
            if i == 0 and self.agg_type_v == "PrevQ":
                o = layer(Q, o)
            else:
                o = layer(o)

        # PE
        # Attention
        for i, layer in enumerate(self.enc_vp):
            p = layer(p)
        p = self.dropout(p)
        # Aggregate
        op = p
        for i, layer in enumerate(self.dec_vp):
            if i == 0 and self.agg_type_v == "PrevQ":
                op = layer(Qp, op)
            else:
                op = layer(op)

        return {'e': o, 'p': op}
    
    def e_reduce_func(self, nodes):
        Q = nodes.mailbox['q'][:,0:1,:]
        v = nodes.mailbox['v']
        Qp = nodes.mailbox['qpos'][:,0:1,:]
        p = nodes.mailbox['pos']

        # Attention
        if self.att_type_e == "OrderPE":
            v = v + self.pe_e(p)
        for i, layer in enumerate(self.enc_e):
            v = layer(v)
        v = self.dropout(v)
        # Aggregate
        o = v
        for i, layer in enumerate(self.dec_e):
            if i == 0 and self.agg_type_e == "PrevQ":
                o = layer(Q, o)
            else:
                o = layer(o)

        # PE
        # Attention
        for i, layer in enumerate(self.enc_ep):
            p = layer(p)
        p = self.dropout(p)
        # Aggregate
        op = p
        for i, layer in enumerate(self.dec_ep):
            if i == 0 and self.agg_type_e == "PrevQ":
                op = layer(Qp, op)
            else:
                op = layer(op)
                
        return {'e': o, 'p': op}
    
    def forward(self, g1, g2, vfeat, efeat, vpos, epos):
        with g1.local_scope():
            g1.srcnodes['node'].data['feat'] = vfeat
            g1.srcnodes['node'].data['pos'] = vpos
            g1.dstnodes['edge'].data['feat'] = efeat[:g1['in'].num_dst_nodes()]
            g1.dstnodes['edge'].data['pos'] = epos[:g1['in'].num_dst_nodes()]
            g1.update_all(self.v_message_func, self.v_reduce_func, etype='in')
            efeat = g1.dstnodes['edge'].data['e']
            efeat = efeat.squeeze(1)
            epos = g1.dstnodes['edge'].data['p']
            epos = epos.squeeze(1)
        with g2.local_scope():       
            g2.srcnodes['edge'].data['feat'] = efeat
            g2.srcnodes['edge'].data['pos'] = epos
            g2.dstnodes['node'].data['feat'] = vfeat[:g2['con'].num_dst_nodes()]
            g2.dstnodes['node'].data['pos'] = vpos[:g2['con'].num_dst_nodes()]
            g2.update_all(self.e_message_func, self.e_reduce_func, etype='con')
            vfeat = g2.dstnodes['node'].data['e']
            vfeat = vfeat.squeeze(1)
            vpos = g2.dstnodes['node'].data['p']
            vpos = vpos.squeeze(1)
        
        return [vfeat, efeat, vpos, epos]
    
    
# ============================ Transformer ===============================
class WhatsnetLSPE(nn.Module): 
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
                 att_type_e = "OrderPE",
                 agg_type_e = "PrevQ",
                 num_att_layer = 2,
                 layernorm = False,
                 dropout=0.6,
                 activation=F.relu):
        super(WhatsnetLSPE, self).__init__()
        self.num_layers = num_layers
        self.activation = activation

        self.pe_v = nn.Linear(weight_dim, vertex_dim)
        self.pe_e = nn.Linear(weight_dim, edge_dim)
        
        self.layers = nn.ModuleList()               
        if num_layers == 1:
             self.layers.append(model(input_vdim, input_edim, vertex_dim, edge_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                      att_type_v=att_type_v, agg_type_v=agg_type_v, att_type_e = att_type_e, agg_type_e=agg_type_e, num_att_layer=num_att_layer,
                                      dropout=dropout, ln=layernorm, activation=None))
        else:
            for i in range(num_layers):
                if i == 0:
                    self.layers.append(model(input_vdim, input_edim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             att_type_v=att_type_v, agg_type_v=agg_type_v, att_type_e = att_type_e, agg_type_e=agg_type_e, num_att_layer=num_att_layer,
                                             dropout=dropout, ln=layernorm, activation=self.activation))
                elif i == (num_layers - 1):
                    self.layers.append(model(hidden_dim, hidden_dim, vertex_dim, edge_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             att_type_v=att_type_v, agg_type_v=agg_type_v, att_type_e = att_type_e, agg_type_e=agg_type_e, num_att_layer=num_att_layer,
                                             dropout=dropout, ln=layernorm, activation=None))
                else:
                    self.layers.append(model(hidden_dim, hidden_dim, hidden_dim, hidden_dim, weight_dim=weight_dim, num_heads=num_heads, num_inds=num_inds,
                                             att_type_v=att_type_v, agg_type_v=agg_type_v, att_type_e = att_type_e, agg_type_e=agg_type_e, num_att_layer=num_att_layer,
                                             dropout=dropout, ln=layernorm, activation=self.activation))
    
    def forward(self, blocks, vfeat, efeat, vpos, epos):
        for l in range(self.num_layers):
            vfeat, efeat, vpos, epos = self.layers[l](blocks[2*l], blocks[2*l+1], vfeat, efeat, vpos, epos)
        
        vfeat = vfeat + self.pe_v(vpos)
        efeat = efeat + self.pe_e(epos)
        
        return vfeat, efeat
    
    
    
    