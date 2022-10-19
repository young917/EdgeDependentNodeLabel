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
import shutil
from tqdm import tqdm
from collections import defaultdict
import time
import argparse
import dgl
from scipy.sparse import csr_matrix
from scipy.sparse import vstack as s_vstack
from scipy.stats import entropy
from scipy.spatial import distance
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
from model.Transformer import Transformer, TransformerLayer
from model.layer import FC, Wrap_Embedding
from model.HCHA import HCHA

# Make Output Directory --------------------------------------------------------------------------------------------------------------
initialization = "rw"
args = utils.parse_args()
if args.evaltype == "test":
    assert args.fix_seed
    outputdir = "results_test/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
    outputParamResFname = outputdir + args.model_name + "/param_result.txt"
    outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"
else:
    outputdir = "results_v2/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
    outputParamResFname = outputdir + args.model_name + "/param_result.txt"
    outputdir += args.model_name + "/" + args.param_name +"/"
if os.path.isdir(outputdir) is False:
    os.makedirs(outputdir)
if os.path.isfile(outputParamResFname) is False:
    with open(outputParamResFname, "w") as f:
        f.write("parameter,TrainLoss,TrainAcc,ValidAcc\n")
print("OutputDir = " + outputdir)
print("Output Param Result = " + outputParamResFname)

if os.path.isdir(outputdir) is False:
    os.makedirs(outputdir)
    
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

# Check ----------------------------------------------------------------------------
exist_filelist = ["log_train.txt", "embedder.pt", "initembedder.pt", "scorer.pt", "log_valid_micro.txt", "log_valid_macro.txt", "log_valid_confusion.txt", "log_test_micro.txt", "log_test_macro.txt", "log_test_confusion.txt"]
for fname in exist_filelist:
    if os.path.isfile(outputdir + fname) is False:
        with open("EXCEPTION.txt", "+a") as f:
            f.write(outputdir + "\t" + fname + "\n")
        sys.exit("No " + fname + " in " + outputdir)

# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args, dataset_name)
test_data = data.get_data(2)
full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
if args.orderflag:
    g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE, data.hedge2nodepos, data.node2hedge, data.node2hedgePE, device)
else:
    g = gen_DGLGraph(args, data.hedge2node, data.hedge2nodepos, data.node2hedge, device)
try:
    fullsampler = dgl.dataloading.NeighborSampler(full_ls)
except:
    fullsampler = dgl.dataloading.MultiLayerNeighborSampler(full_ls)

if args.embedder == "hcha":
    test_edata, test_vdata, test_label = [], [], []
    for hedge in test_data:
        for vidx, v in enumerate(data.hedge2node[hedge]):
            test_edata.append(hedge)
            test_vdata.append(v)
            test_label.append(data.hedge2nodepos[hedge][vidx])
    g = g.to(device)
    test_label = torch.LongTensor(test_label).to(device)
else:
    if args.use_gpu:
        g = g.to(device)
        test_data = test_data.to(device)
    testdataloader = dgl.dataloading.NodeDataLoader(g, {"edge": test_data}, fullsampler, batch_size=args.bs, shuffle=False, drop_last=False)

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

print("Model:", args.embedder)
# model init
if args.embedder == "hnhn":
    embedder = HNHN(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, args.use_efeat, args.num_layers, args.dropout).to(device)
elif args.embedder == "hgnn":
    embedder = HGNN(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, args.num_layers, args.dropout).to(device)
elif args.embedder == "hat":
    if args.encode_type == "":
        embedder = HyperAttn(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, weight_dim=0, num_layer=args.num_layers, dropout=args.dropout).to(device)
    else:
        embedder = HyperAttn(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, weight_dim=args.order_dim, num_layer=args.num_layers, dropout=args.dropout).to(device)   
elif args.embedder == "unigcnii":
    embedder = UniGCNII(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, num_layer=args.num_layers, dropout=args.dropout).to(device)
elif args.embedder == "hcha":
    embedder = HCHA(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, num_layers=args.num_layers, num_heads=args.num_heads, feat_drop=args.dropout).to(device)
elif args.embedder == "transformer":    
    input_vdim = args.input_vdim
    pos_dim = 0
    embedder = Transformer(TransformerLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, pos_dim=pos_dim,
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v, att_type_e=args.att_type_e, agg_type_e=args.agg_type_e,
                           num_att_layer=args.num_att_layer, dropout=args.dropout).to(device)

    
print("Embedder to Device")
print("Scorer = ", args.scorer)
# pick scorer
if args.scorer == "sm":
    scorer = FC(args.dim_vertex + args.dim_edge, args.dim_edge, args.output_dim, args.scorer_num_layers, args.dropout).to(device)

print("Test")
initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt")) # , map_location=device
embedder.load_state_dict(torch.load(outputdir + "embedder.pt")) # , map_location=device
scorer.load_state_dict(torch.load(outputdir + "scorer.pt")) # , map_location=device

initembedder.eval()
embedder.eval()
scorer.eval()

total_pred = []
total_label = []
node2predrole = {}
node2ansrole = {}
nodes = torch.LongTensor(range(data.numnodes)).to(device)

with torch.no_grad():
    if args.embedder == "hcha":
        v_feat, recon_loss = initembedder(nodes)
        e_feat = torch.zeros((data.numhedges, args.input_vdim)).to(device)
        DV2 = data.DV2.to(device)
        invDE = data.invDE.to(device)
        v, e = embedder(g, v_feat, e_feat, DV2, invDE)
        hembedding = e[test_edata]
        vembedding = v[test_vdata]
        input_embeddings = torch.cat([hembedding,vembedding], dim=1)
        predictions = scorer(input_embeddings)
        pred_cls = torch.argmax(predictions, dim=1)
        eval_acc = torch.eq(pred_cls, test_label).sum().item() / len(test_label)
        y_test = test_label.detach().cpu().numpy()
        pred = pred_cls.detach().cpu().numpy()
        
        for hedge, node, predlabel, anslabel in zip(test_edata, test_vdata, pred_cls.detach().cpu().numpy().tolist(), test_label.detach().cpu().numpy().tolist()):
            hedge, node, predlabel, anslabel = int(hedge), int(node), int(predlabel), int(anslabel)
            if node not in node2predrole:
                node2predrole[node] = np.zeros(args.output_dim)
                node2ansrole[node] = np.zeros(args.output_dim)

            node2predrole[node][predlabel] += 1
            node2ansrole[node][anslabel] += 1  
        
    else:
        for input_nodes, output_nodes, blocks in tqdm(testdataloader):     
            # Wrap up loader
            blocks = [b.to(device) for b in blocks]
            srcs, dsts = blocks[-1].edges(etype='in')
            nodeindices_in_batch = srcs.to(device)
            nodeindices = blocks[-1].srcdata[dgl.NID]['node'][srcs]
            hedgeindices_in_batch = dsts.to(device)
            hedgeindices = blocks[-1].dstdata[dgl.NID]['edge'][dsts]
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
            total_label.append(nodelabels.detach())

            pred_cls = torch.argmax(predictions, dim=1)
            for v, h, vpred, vlab in zip(nodeindices.tolist(), hedgeindices.tolist(), pred_cls.detach().cpu().tolist(), nodelabels.detach().cpu().tolist()):
                v, h, vpred, vlab = int(v), int(h), int(vpred), int(vlab)
                if v not in data.hedge2node[h]:
                    print(data.hedge2node[h])
                    print(v)
                assert v in data.hedge2node[h]
                assert h in data.node2hedge[v]
                existflag = False
                for vorder, _v in enumerate(data.hedge2node[h]):
                    if _v == v:
                        if data.hedge2nodepos[h][vorder] == vlab:
                            existflag = True
                assert existflag

                if v not in node2predrole:
                    node2predrole[v] = np.zeros(args.output_dim)
                    node2ansrole[v] = np.zeros(args.output_dim)

                node2predrole[v][vpred] += 1
                node2ansrole[v][vlab] += 1  
                
        total_label = torch.cat(total_label, dim=0)
        total_pred = torch.cat(total_pred)
        pred_cls = torch.argmax(total_pred, dim=1)
        eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
        y_test = total_label.cpu().numpy()
        pred = pred_cls.cpu().numpy()
            
# Compare Test F1 -----------------------------------------------------------------------
confusion, accuracy, precision, recall, f1_micro = utils.get_clf_eval(y_test, pred, avg='micro')
with open(outputdir + "log_test_micro.txt", "r") as f: # Compare
    past_f1_micro = float(f.readline().rstrip().split(":")[-1])
assert abs(f1_micro - past_f1_micro) <= 0.0001

confusion, accuracy, precision, recall, f1_macro = utils.get_clf_eval(y_test, pred, avg='macro')
with open(outputdir + "log_test_macro.txt", "r") as f:
    past_f1_macro = float(f.readline().rstrip().split(":")[-1])
assert abs(f1_macro - past_f1_macro) <= 0.0001

# -------------------------------------------------------------------------------------  
for v in node2predrole.keys():
    node2predrole[v] = node2predrole[v] / np.sum(node2predrole[v])
    node2ansrole[v] = node2ansrole[v] / np.sum(node2ansrole[v])
    
# Calculate Entropy
try:
    entropy_pred = []
    entropy_ans = []
    for v in node2predrole.keys():
        e_pred = entropy(node2predrole[v])
        e_ans = entropy(node2ansrole[v])
        entropy_pred.append(e_pred)
        entropy_ans.append(e_ans)
    with open(outputdir + "entropy.txt", "w") as f:
        f.write("Real,Pred\n")
        for e_pred, e_ans in zip(entropy_pred, entropy_ans):
            f.write(str(e_pred) + "," + str(e_ans) + "\n")
    print("Average (Pred, Ans): %.4f    %.4f" % (np.mean(entropy_pred), np.mean(entropy_ans)))
except:
    print("except on entorpy")
    with open("EXCEPTION.txt", "+a") as f:
        f.write(outputdir + " error on entropy\n")
        
# Calculate JSD-Divergence
jsd_div_list = []
try:
    for v in node2predrole.keys():
        jsd_div = distance.jensenshannon(node2ansrole[v], node2predrole[v])
        jsd_div_list.append(jsd_div)
    with open(outputdir + "jsd_div.txt", "w") as f:
        for jsd_div in jsd_div_list:
            f.write(str(jsd_div) + "\n")
    print("Average: %.4f" % (np.mean(jsd_div_list)) )
except:
    print("except on jsd-divergence")
    with open("EXCEPTION.txt", "+a") as f:
        f.write(outputdir + " error on jsd-divergence\n")
