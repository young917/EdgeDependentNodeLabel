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
from collections import defaultdict
import time
import argparse
import dgl
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
from model.WhatsnetHAT import WhatsnetHAT, WhatsnetHATLayer
from model.WhatsnetHNHN import WhatsnetHNHN, WhatsnetHNHNLayer
from model.layer import FC, Wrap_Embedding

# Make Output Directory --------------------------------------------------------------------------------------------------------------
initialization = "rw"
args = utils.parse_args()
outputdir = "results_test/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"
if os.path.isdir(outputdir) is False:
    os.makedirs(outputdir)
print("OutputDir = " + outputdir)

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

exp_num = args.exp_num
test_epoch = args.test_epoch
plot_epoch = args.epochs


# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args, dataset_name)
allhedges = torch.LongTensor(np.arange(data.numhedges))
full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
if data.weight_flag:
    g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE, data.hedge2nodepos, data.node2hedge, data.node2hedgePE, device)
else:
    g = gen_DGLGraph(args, data.hedge2node, data.hedge2nodepos, data.node2hedge, device)
try:
    fullsampler = dgl.dataloading.NeighborSampler(full_ls)
except:
    fullsampler = dgl.dataloading.MultiLayerNeighborSampler(full_ls)
if args.use_gpu:
    g = g.to(device)
    hedge_data = allhedges.to(device)
else:
    hedge_data = allhedges
dataloader = dgl.dataloading.NodeDataLoader( g, {"edge": hedge_data}, fullsampler, batch_size=args.bs, shuffle=False, drop_last=False) # , num_workers=4

# init embedder
args.input_vdim = 48
if args.orderflag:
    args.input_vdim = 44
args.input_edim = data.e_feat.size(1)
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

print("Model:", args.embedder)
# model init
if args.embedder == "hnhn":
    embedder = HNHN(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, args.use_efeat, args.num_layers, args.dropout).to(device)
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
                           num_att_layer=args.num_att_layer, dropout=args.dropout, weight_flag=data.weight_flag, pe_ablation_flag=pe_ablation_flag).to(device)
elif args.embedder == "whatsnetHAT":
    input_vdim = args.input_vdim
    embedder = WhatsnetHAT(WhatsnetHATLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, 
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v,
                           num_att_layer=args.num_att_layer, dropout=args.dropout).to(device)
elif args.embedder == "whatsnetHNHN":
    input_vdim = args.input_vdim
    embedder = WhatsnetHNHN(WhatsnetHNHNLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, 
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v,
                           num_att_layer=args.num_att_layer, dropout=args.dropout).to(device)
    
print("Embedder to Device")
print("Scorer = ", args.scorer)
# pick scorer
if args.scorer == "sm":
    scorer = FC(args.dim_vertex + args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)

if args.embedder == "unigcnii":
    optim = torch.optim.Adam([
            dict(params=embedder.reg_params, weight_decay=0.01),
            dict(params=embedder.non_reg_params, weight_decay=5e-4),
            dict(params=list(initembedder.parameters()) + list(scorer.parameters()), weight_decay=0.0)
        ], lr=0.01)
elif args.optimizer == "adam":
    optim = torch.optim.Adam(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr) #, weight_decay=args.weight_decay)
elif args.optimizer == "adamw":
    optim = torch.optim.AdamW(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
elif args.optimizer == "rms":
    optime = torch.optim.RMSprop(list(initembedder.parameters())+list(embedder.parameters())+list(scorer.parameters()), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=args.gamma)
loss_fn = nn.CrossEntropyLoss()

initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt")) # , map_location=device
embedder.load_state_dict(torch.load(outputdir + "embedder.pt")) # , map_location=device
scorer.load_state_dict(torch.load(outputdir + "scorer.pt")) # , map_location=device

initembedder.eval()
embedder.eval()
scorer.eval()

with torch.no_grad():
    allpredictions = defaultdict(dict)
    
    total_pred = []
    total_label = []
    num_data = 0
    
    # Batch ==============================================================
    for input_nodes, output_nodes, blocks in tqdm(dataloader): #, desc="batch"):      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices_in_batch = srcs.to(device)
        hedgeindices_in_batch = dsts.to(device)
        nodeindices = blocks[-1].srcdata[dgl.NID]['node'][srcs]
        hedgeindices = blocks[-1].srcdata[dgl.NID]['edge'][dsts]
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)
        
        # Get Embedding
        if args.embedder == "hnhn":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v_reg_weight = data.v_reg_weight[input_nodes['node']].to(device)
            v_reg_sum = data.v_reg_sum[input_nodes['node']].to(device)
            e_reg_weight = data.e_reg_weight[input_nodes['edge']].to(device)
            e_reg_sum = data.e_reg_sum[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum)
        elif args.embedder == "whatsnetHNHN":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v_reg_weight = data.v_reg_weight[input_nodes['node']].to(device)
            v_reg_sum = data.v_reg_sum[input_nodes['node']].to(device)
            e_reg_weight = data.e_reg_weight[input_nodes['edge']].to(device)
            e_reg_sum = data.e_reg_sum[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, e_reg_weight, v_reg_sum)
        elif args.embedder == "hgnn":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            DV2 = data.DV2[input_nodes['node']].to(device)
            invDE = data.invDE[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, DV2, invDE)
        elif args.embedder == "unigcnii":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            degV = data.degV[input_nodes['node']].to(device)
            degE = data.degE[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, degE, degV)
        elif args.embedder == "whatsnet":
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
        hembedding = e[hedgeindices_in_batch]
        vembedding = v[nodeindices_in_batch]
        input_embeddings = torch.cat([hembedding,vembedding], dim=1)
        predictions = scorer(input_embeddings)
        
        total_pred.append(predictions.detach())
        pred_cls = torch.argmax(predictions, dim=1)
        total_label.append(nodelabels.detach())
        
        for v, h, vpred, vlab in zip(nodeindices.tolist(), hedgeindices.tolist(), pred_cls.detach().cpu().tolist(), nodelabels.detach().cpu().tolist()):
            assert v in data.hedge2node[h]
            for vorder in range(len(data.hedge2node[h])):
                if data.hedge2node[h][vorder] == v:
                    assert vlab == data.hedge2nodepos[h][vorder]
            if args.binning > 0:
                allpredictions[h][v] = data.binindex[int(vpred)]
            else:
                allpredictions[h][v] = int(vpred)
        num_data += predictions.shape[0]
        
    total_label = torch.cat(total_label, dim=0)
    total_pred = torch.cat(total_pred)
    pred_cls = torch.argmax(total_pred, dim=1)
    eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    y_test = total_label.cpu().numpy()
    pred = pred_cls.cpu().numpy()
    confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
    with open(outputdir + "all_micro.txt", "w") as f:
        f.write("Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(accuracy,precision,recall,f1))
    confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
    with open(outputdir + "all_confusion.txt", "w") as f:
        for r in range(args.output_dim):
            for c in range(args.output_dim):
                f.write(str(confusion[r][c]))
                if c == args.output_dim -1 :
                    f.write("\n")
                else:
                    f.write("\t")
    with open(outputdir + "all_macro.txt", "w") as f:               
        f.write("Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(accuracy,precision,recall,f1))
        
    with open(outputdir + "prediction.txt", "w") as f:
        for h in range(data.numhedges):
            line = []
            for v in data.hedge2node[h]:
                line.append(str(allpredictions[h][v]))
            f.write("\t".join(line) + "\n")
