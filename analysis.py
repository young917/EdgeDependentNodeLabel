import torch
import torch.nn as nn
import numpy as np
from torch.nn import Parameter
import torch.optim as optim
from sklearn import metrics
import random
import os
import sys
import utils
from tqdm import tqdm
import time
import argparse
import dgl
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
import multiprocessing
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix

from preprocess.data_load import gen_DGLGraph, gen_weighted_DGLGraph
import preprocess.data_load as dl
from preprocess.batch import DataLoader
from initialize.initial_embedder import MultipleEmbedding
from initialize.random_walk_hyper import random_walk_hyper

from model.HNHN import HNHN
from model.HGNN import HGNN
from model.HAT import HyperAttn
from model.UniGCN import UniGCNII
from model.Whatsnet import Whatsnet, WhatsnetLayer
from model.WhatsnetClassifier import WhatsnetClassifier
from model.layer import FC, Wrap_Embedding

import shutil

# Make Output Directory --------------------------------------------------------------------
initialization = "rw"
args = utils.parse_args()
outputdir = "results_test/" + args.dataset_name + "_" + str(args.k)  + "/" + args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"
savedir = "AnalysisResult/" + args.dataset_name + "_" + str(args.k) + "/" + args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"

assert args.embedder == "whatsnet"

print("OutputDir = " + outputdir)
print("SaveDir = " + savedir)

assert os.path.isdir(outputdir)
if os.path.isdir(savedir) is False:
    os.makedirs(savedir)
else:
    shutil.rmtree(savedir)
    os.makedirs(savedir)

# Read Result -----------------------------------------------------------------------
saved_f1_macro, saved_f1_micro = 0,0
assert os.path.isfile(outputdir + "log_test_macro.txt")
assert os.path.isfile(outputdir + "log_test_micro.txt")

num_epoch = 0
with open(outputdir + "log_test_micro.txt", "r") as f:
    for line in f.readlines():
        ep_str = line.rstrip().split(":")[0]
        f1_str = line.rstrip().split(":")[-1]
        epoch = int(ep_str.split(" ")[0])
        f1 = float(f1_str)
        if saved_f1_micro < f1:
            num_epoch = epoch
            saved_f1_micro = f1
        else:
            break
print("Epoch:", num_epoch)
with open(outputdir + "log_test_macro.txt", "r") as f:
    for line in f.readlines():
        ep_str = line.rstrip().split(":")[0]
        f1_str = line.rstrip().split(":")[-1]
        epoch = int(ep_str.split(" ")[0])
        f1 = float(f1_str)
        if epoch == num_epoch:
            saved_f1_macro = f1
            break
print("Saved F1: (Micro) %.4f  (Macro)  %.4f" % (saved_f1_micro, saved_f1_macro))
            
# Initialization --------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset_name #'citeseer' 'cora'

if args.fix_seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    dgl.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dgl.seed(args.seed)

# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args, dataset_name)
target_data = data.get_data(2)

ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
if data.weight_flag:
    g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE, data.hedge2nodepos, data.node2hedge, data.node2hedgePE, device)
else:
    g = gen_DGLGraph(args, data.hedge2node, data.hedge2nodepos, data.node2hedge, device)
try:
    sampler = dgl.dataloading.NeighborSampler(ls)
except:
    sampler = dgl.dataloading.MultiLayerNeighborSampler(ls, False)
dataloader = dgl.dataloading.NodeDataLoader( g, {"edge": target_data}, sampler, batch_size=256, shuffle=True, drop_last=False)
args.input_vdim = data.v_feat.size(1)
args.input_edim = data.e_feat.size(1)

# init embedder
args.input_vdim = 48
if args.orderflag:
    args.input_vdim = 44
savefname = "../%s_%d_wv_%d_%s.npy" % (args.dataset_name, args.k, args.input_vdim, args.walk)
node_list = np.arange(data.numnodes).astype('int')
if os.path.isfile(savefname) is False:
    walk_path = random_walk_hyper(args, node_list, data.hedge2node)
    walks = np.loadtxt(walk_path, delimiter=" ").astype('int')
    print("Start turning path to strs")
    split_num = 20
    pool = ProcessPoolExecutor(max_workers=split_num)
    process_list = []
    walks = np.array_split(walks, split_num)
    result = []
    for walk in walks:
        process_list.append(pool.submit(utils.walkpath2str, walk))
    for p in as_completed(process_list):
        result += p.result()
    pool.shutdown(wait=True)
    walks = result
    # print(walks)
    print("Start Word2vec")
    print("num cpu cores", multiprocessing.cpu_count())
    w2v = Word2Vec( walks, vector_size=args.input_vdim, window=10, min_count=0, sg=1, epochs=1, workers=multiprocessing.cpu_count())
    print(w2v.wv['0'])
    wv = w2v.wv
    A = [wv[str(i)] for i in range(data.numnodes)]
    np.save(savefname, A)
else:
    print("load exist init walks")
    A = np.load(savefname)
A = StandardScaler().fit_transform(A)
A = A.astype('float32')
A = torch.tensor(A).to(device)
initembedder = Wrap_Embedding(data.numnodes, args.input_vdim, scale_grad_by_freq=False, padding_idx=0, sparse=False)
initembedder.weight = nn.Parameter(A)
initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt"))
initembedder = initembedder.to(device)

print("Model:", args.embedder)
# model init
if args.embedder == "hnhn":
    embedder = HNHN(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, args.num_layers, args.dropout).to(device)
elif args.embedder == "hgnn":
    embedder = HGNN(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, args.num_layers, args.dropout).to(device)
elif args.embedder == "hat":
    embedder = HyperAttn(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, weight_dim=0, num_layer=args.num_layers, dropout=args.dropout).to(device)   
elif args.embedder == "unigcnii":
    embedder = UniGCNII(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, num_layer=args.num_layers, dropout=args.dropout).to(device)
elif args.embedder == "whatsnet":    
    input_vdim = args.input_vdim
    pe_ablation_flag = args.pe_ablation
    embedder = Whatsnet(WhatsnetLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, num_inds=args.num_inds,
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v, att_type_e=args.att_type_e, agg_type_e=args.agg_type_e,
                           num_att_layer=args.num_att_layer, dropout=args.dropout, weight_flag=data.weight_flag, pe_ablation_flag=pe_ablation_flag, vis_flag=args.analyze_att).to(device)
    
embedder.load_state_dict(torch.load(outputdir + "embedder.pt"))
embedder = embedder.to(device)

# pick scorer
if args.scorer == "sm":
    scorer = FC(args.dim_vertex + args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)
elif args.scorer == "wc": #whatsnet
    scorer = WhatsnetClassifier(args.dim_vertex, args.output_dim, dim_hidden=args.dim_hidden, num_layer=args.scorer_num_layers).to(device)
    
scorer.load_state_dict(torch.load(outputdir + "scorer.pt"))
scorer = scorer.to(device)

# Test ===========================================================================================================================================================================
initembedder.eval()
embedder.eval()
scorer.eval()
with torch.no_grad():
    total_pred = []
    total_label = []
    
    for batch_index, (input_nodes, output_nodes, blocks) in tqdm(enumerate(dataloader)):
            
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices_in_batch = srcs.to(device)
        nodeindices = blocks[-1].srcdata[dgl.NID]['node'][srcs]
        hedgeindices_in_batch = dsts.to(device)
        hedgeindices = blocks[-1].srcdata[dgl.NID]['edge'][dsts]
        hedgeindices = dsts.to(device)
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)
        
        # Get Embedding
        # Get Embedding
        if args.embedder == "whatsnet":
            if batch_index > 5:
                embedder.off_vis_flag()
            else:
                embedder.set_savename(savedir, batch_index)
                
            if args.att_type_v in ["ITRE", "ShawRE", "RafRE"]:
                vindex = torch.arange(len(input_nodes['node'])).unsqueeze(1).to(device)
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat, vindex)
            else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat)
        else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat)
                
        # Predict Class
        if args.scorer == "sm":
            hembedding = e[hedgeindices_in_batch]
            vembedding = v[nodeindices_in_batch]
            input_embeddings = torch.cat([hembedding,vembedding], dim=1)
            predictions = scorer(input_embeddings)
        elif args.scorer == "wc":
            predictions, nodelabels = scorer(blocks[-1], v, e)
        
#         if batch_index <= 5:
        save_data = {
            "hembedding": hembedding.detach().cpu(),
            "vembedding" : vembedding.detach().cpu(),
            "nodelabel" : nodelabels.detach().cpu(),
            "predictlabel" : predictions.detach().cpu(),
            "nodeindex" : nodeindices,
            "hedgeindex" : hedgeindices
        }
        with open(savedir + "final_emb_{}.pkl".format(batch_index), "wb") as f:
            pickle.dump(save_data, f)

        # Predict Class
        input_embeddings = torch.cat([hembedding,vembedding], dim=1)
        predictions = scorer(input_embeddings)
        total_pred.append(predictions.detach())
        total_label.append(nodelabels.detach())
        
    total_label = torch.cat(total_label, dim=0)
    total_pred = torch.cat(total_pred)
    pred_cls = torch.argmax(total_pred, dim=1)
    eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    print(eval_acc)
    y_test = total_label.cpu().numpy()
    pred = pred_cls.cpu().numpy()
    confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='micro')
    print("MICRO")
    print("Precision:{}/Recall:{}/F1:{}\n".format(precision, recall, f1))
    confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='macro')
    print("MACRO")
    print("Precision:{}/Recall:{}/F1:{}\n".format(precision,recall,f1))
    