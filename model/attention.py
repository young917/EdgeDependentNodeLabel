import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from tqdm import tqdm, trange
import copy
import math

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    
    def __init__(
            self,
            n_head,
            d_model,
            d_k,
            d_v,
            dropout,
            diag_mask,
            input_dim):
        super().__init__()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        
        self.w_qs = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(input_dim, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(input_dim, n_head * d_v, bias=False)
        
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (d_model + d_v)))
        
        self.attention = ScaledDotProductAttention(
            temperature=np.power(d_k, 0.5))
        
        self.fc1 = FeedForward([n_head * d_v, d_model], use_bias=False)
        self.fc2 = FeedForward([n_head * d_v, d_model], use_bias=False)
        
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        self.layer_norm3 = nn.LayerNorm(input_dim)
        
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        
        self.diag_mask_flag = diag_mask
        self.diag_mask = None
    
    def pass_(self, inputs):
        return inputs
    
    def forward(self, q, k, v, diag_mask, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        
        residual_dynamic = q
        residual_static = v
        
        q = self.layer_norm1(q)
        k = self.layer_norm2(k)
        v = self.layer_norm3(v)
        
        sz_b, len_q, _ = q.shape
        sz_b, len_k, _ = k.shape
        sz_b, len_v, _ = v.shape
        
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous(
        ).view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous(
        ).view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous(
        ).view(-1, len_v, d_v)  # (n*b) x lv x dv
        
        n = sz_b * n_head
        
        if self.diag_mask is not None:
            if (len(self.diag_mask) <= n) or (
                    self.diag_mask.shape[1] != len_v):
                self.diag_mask = torch.ones((len_v, len_v), device=device)
                if self.diag_mask_flag == 'True':
                    self.diag_mask -= torch.eye(len_v, len_v, device=device)
                self.diag_mask = self.diag_mask.repeat(n, 1, 1)
                diag_mask = self.diag_mask
            else:
                diag_mask = self.diag_mask[:n]
        
        else:
            self.diag_mask = (torch.ones((len_v, len_v), device=device))
            if self.diag_mask_flag == 'True':
                self.diag_mask -= torch.eye(len_v, len_v, device=device)
            self.diag_mask = self.diag_mask.repeat(n, 1, 1)
            diag_mask = self.diag_mask
        
        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
        
        dynamic, attn = self.attention(q, k, v, diag_mask, mask=mask)
        
        dynamic = dynamic.view(n_head, sz_b, len_q, d_v)
        dynamic = dynamic.permute(
            1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)
        static = v.view(n_head, sz_b, len_q, d_v)
        static = static.permute(
            1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)
        
        dynamic = self.dropout(self.fc1(dynamic)) if self.dropout is not None else self.fc1(dynamic)
        static = self.dropout(self.fc2(static)) if self.dropout is not None else self.fc2(static)
        
        
        return dynamic, static, attn