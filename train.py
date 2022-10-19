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
from model.Transformer import Transformer, TransformerLayer
from model.TransformerHAT import TransformerHAT, TransformerHATLayer
from model.TransformerHNHN import TransformerHNHN, TransformerHNHNLayer
from model.layer import FC, Wrap_Embedding

def run_epoch(args, data, dataloader, initembedder, embedder, scorer, optim, scheduler, loss_fn, opt="train"):
    total_pred = []
    total_label = []
    num_data = 0
    num_recon_data = 0
    total_loss = 0
    total_ce_loss = 0
    total_recon_loss = 0
    
    # Batch ==============================================================
    ts = time.time()
    batchcount = 0
    for input_nodes, output_nodes, blocks in dataloader:      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices = srcs.to(device)
        hedgeindices = dsts.to(device)
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)
        
        batchcount += 1
        # Get Embedding
        if args.embedder == "hnhn":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v_reg_weight = data.v_reg_weight[input_nodes['node']].to(device)
            v_reg_sum = data.v_reg_sum[input_nodes['node']].to(device)
            e_reg_weight = data.e_reg_weight[input_nodes['edge']].to(device)
            e_reg_sum = data.e_reg_sum[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum)
        elif args.embedder == "transformerHNHN":
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
        elif args.embedder == "transformer":
            if args.att_type_v in ["ITRE", "ShawRE"]:
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
        hembedding = e[hedgeindices]
        vembedding = v[nodeindices]
        input_embeddings = torch.cat([hembedding,vembedding], dim=1)
        predictions = scorer(input_embeddings)
        total_pred.append(predictions.detach())
        total_label.append(nodelabels.detach())
        
        # Back Propagation
        num_data += predictions.shape[0]
        num_recon_data += input_nodes['node'].shape[0]
        ce_loss = loss_fn(predictions, nodelabels)
        loss = ce_loss + args.rw * recon_loss
        if opt == "train":
            optim.zero_grad()
            loss.backward() 
            optim.step()
        total_loss += (loss.item() * predictions.shape[0])
        total_ce_loss += (ce_loss.item() * predictions.shape[0])
        total_recon_loss += (recon_loss.item() * input_nodes['node'].shape[0]) # this is fixed as zero
        if opt == "train":
            torch.cuda.empty_cache()
    
    print("Time : ", time.time() - ts)
    
    return total_pred, total_label, total_loss / num_data, total_ce_loss / num_data, total_recon_loss / num_recon_data, initembedder, embedder, scorer, optim, scheduler

def run_test_epoch(args, data, testdataloader, initembedder, embedder, scorer, loss_fn):
    total_pred = []
    total_label = []
    num_data = 0
    num_recon_data = 0
    total_loss = 0
    total_ce_loss = 0
    total_recon_loss = 0
    
    # Batch ==============================================================
    ts = time.time()
    batchcount = 0
    for input_nodes, output_nodes, blocks in testdataloader:      
        # Wrap up loader
        blocks = [b.to(device) for b in blocks]
        srcs, dsts = blocks[-1].edges(etype='in')
        nodeindices_in_batch = srcs.to(device)
        nodeindices = blocks[-1].srcdata[dgl.NID]['node'][srcs]
        hedgeindices_in_batch = dsts.to(device)
        hedgeindices = blocks[-1].srcdata[dgl.NID]['edge'][dsts]
        hedgeindices = dsts.to(device)
        nodelabels = blocks[-1].edges[('node','in','edge')].data['label'].long().to(device)
        
        batchcount += 1
        # Get Embedding
        if args.embedder == "hnhn":
            v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
            e_feat = data.e_feat[input_nodes['edge']].to(device)
            v_reg_weight = data.v_reg_weight[input_nodes['node']].to(device)
            v_reg_sum = data.v_reg_sum[input_nodes['node']].to(device)
            e_reg_weight = data.e_reg_weight[input_nodes['edge']].to(device)
            e_reg_sum = data.e_reg_sum[input_nodes['edge']].to(device)
            v, e = embedder(blocks, v_feat, e_feat, v_reg_weight, v_reg_sum, e_reg_weight, e_reg_sum)
        elif args.embedder == "transformerHNHN":
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
        elif args.embedder == "transformer":
            if args.att_type_v in ["ITRE", "ShawRE", "RafRE"]:
                vindex = torch.arange(len(input_nodes['node'])).unsqueeze(1).to(device)
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat, vindex)
            else:
                v_feat, recon_loss = initembedder(input_nodes['node'].to(device))
                e_feat = data.e_feat[input_nodes['edge']].to(device)
                v, e = embedder(blocks, v_feat, e_feat, vindex)
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
        
        num_data += predictions.shape[0]
        num_recon_data += input_nodes['node'].shape[0]
        ce_loss = loss_fn(predictions, nodelabels)
        loss = ce_loss + args.rw * recon_loss

        total_loss += (loss.item() * predictions.shape[0])
        total_ce_loss += (ce_loss.item() * predictions.shape[0])
        total_recon_loss += (recon_loss.item() * input_nodes['node'].shape[0]) # This is fixed as zero
        
    return total_pred, total_label, total_loss / num_data, total_ce_loss / num_data, total_recon_loss / num_recon_data, initembedder, embedder, scorer

# Make Output Directory --------------------------------------------------------------------------------------------------------------
initialization = "rw"
args = utils.parse_args()
if args.evaltype == "test":
    assert args.fix_seed
    outputdir = "results_test/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
    outputParamResFname = outputdir + args.model_name + "/param_result.txt"
    outputdir += args.model_name + "/" + args.param_name +"/" + str(args.seed) + "/"
else:
    outputdir = "results/" + args.dataset_name + "_" + str(args.k) + "/" + initialization + "/"
    outputdir += args.model_name + "/" + args.param_name +"/"
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

if os.path.isfile(outputdir + "checkpoint.pt") and args.recalculate is False:
    print("Start from checkpoint")
else:
    if os.path.isfile(outputdir + "log_train.txt"):
        os.remove(outputdir + "log_train.txt")
    if os.path.isfile(outputdir + "log_valid_micro.txt"):
        os.remove(outputdir + "log_valid_micro.txt")
    if os.path.isfile(outputdir + "log_valid_confusion.txt"):
        os.remove(outputdir + "log_valid_confusion.txt")
    if os.path.isfile(outputdir + "log_valid_macro.txt"):
        os.remove(outputdir + "log_valid_macro.txt")
    if os.path.isfile(outputdir + "log_test_micro.txt"):
        os.remove(outputdir + "log_test_micro.txt")
    if os.path.isfile(outputdir + "log_test_confusion.txt"):
        os.remove(outputdir + "log_test_confusion.txt")
    if os.path.isfile(outputdir + "log_test_macro.txt"):
        os.remove(outputdir + "log_test_macro.txt")
        
    if os.path.isfile(outputdir + "initembedder.pt"):
        os.remove(outputdir + "initembedder.pt")
    if os.path.isfile(outputdir + "embedder.pt"):
        os.remove(outputdir + "embedder.pt")
    if os.path.isfile(outputdir + "scorer.pt"):
        os.remove(outputdir + "scorer.pt")
    if os.path.isfile(outputdir + "evaluation.txt"):
        os.remove(outputdir + "evaluation.txt")
            
# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args, dataset_name)
train_data = data.get_data(0)
valid_data = data.get_data(1)
if args.evaltype == "test":
    test_data = data.get_data(2)
ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): args.sampling}] * (args.num_layers * 2 + 1)
full_ls = [{('node', 'in', 'edge'): -1, ('edge', 'con', 'node'): -1}] * (args.num_layers * 2 + 1)
if data.weight_flag:
    g = gen_weighted_DGLGraph(args, data.hedge2node, data.hedge2nodePE, data.hedge2nodepos, data.node2hedge, data.node2hedgePE, device)
else:
    g = gen_DGLGraph(args, data.hedge2node, data.hedge2nodepos, data.node2hedge, device)
try:
    sampler = dgl.dataloading.NeighborSampler(ls)
    fullsampler = dgl.dataloading.NeighborSampler(full_ls)
except:
    sampler = dgl.dataloading.MultiLayerNeighborSampler(ls, False)
    fullsampler = dgl.dataloading.MultiLayerNeighborSampler(full_ls)
if args.use_gpu:
    g = g.to(device)
    train_data = train_data.to(device)
    valid_data = valid_data.to(device)
    if args.evaltype == "test":
        test_data = test_data.to(device)
dataloader = dgl.dataloading.NodeDataLoader( g, {"edge": train_data}, sampler, batch_size=args.bs, shuffle=True, drop_last=False) # , num_workers=4
validdataloader = dgl.dataloading.NodeDataLoader( g, {"edge": valid_data}, sampler, batch_size=args.bs, shuffle=True, drop_last=False)
if args.evaltype == "test":
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
    embedder = HNHN(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, args.num_layers, args.dropout).to(device)
elif args.embedder == "hgnn":
    embedder = HGNN(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, args.num_layers, args.dropout).to(device)
elif args.embedder == "hat":
    embedder = HyperAttn(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, weight_dim=0, num_layer=args.num_layers, dropout=args.dropout).to(device)   
elif args.embedder == "unigcnii":
    embedder = UniGCNII(args.input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, num_layer=args.num_layers, dropout=args.dropout).to(device)
elif args.embedder == "transformer":    
    input_vdim = args.input_vdim
    pe_ablation_flag = args.pe_ablation
    embedder = Transformer(TransformerLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers,
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v, att_type_e=args.att_type_e, agg_type_e=args.agg_type_e,
                           num_att_layer=args.num_att_layer, dropout=args.dropout, weight_flag=data.weight_flag, pe_ablation_flag=pe_ablation_flag).to(device)
elif args.embedder == "transformerHAT":
    input_vdim = args.input_vdim
    embedder = TransformerHAT(TransformerHATLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
                           weight_dim=args.order_dim, num_heads=args.num_heads, num_layers=args.num_layers, 
                           att_type_v=args.att_type_v, agg_type_v=args.agg_type_v,
                           num_att_layer=args.num_att_layer, dropout=args.dropout).to(device)
elif args.embedder == "transformerHNHN":
    input_vdim = args.input_vdim
    embedder = TransformerHNHN(TransformerHNHNLayer, input_vdim, args.input_edim, args.dim_hidden, args.dim_vertex, args.dim_edge, 
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

# Train =================================================================================================================================================================================
train_acc=0
patience = 0
best_eval_acc = 0
epoch_start = 1
if os.path.isfile(outputdir + "checkpoint.pt") and args.recalculate is False:
    checkpoint = torch.load(outputdir + "checkpoint.pt") #, map_location=device)
    epoch_start = checkpoint['epoch'] + 1
    initembedder.load_state_dict(checkpoint['initembedder'])
    embedder.load_state_dict(checkpoint['embedder'])
    scorer.load_state_dict(checkpoint['scorer'])
    optim.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    best_eval_acc = checkpoint['best_eval_acc']
    patience = checkpoint['patience']    
    
    print("Load {} epoch trainer".format(epoch_start))
    print("best_eval_acc = {}\tpatience = {}".format(best_eval_acc, patience))

    if args.save_epochs > 0:
        print("Model Save")
        modelsavename = outputdir + "embedder.pt"
        torch.save(embedder.state_dict(), modelsavename)
        scorersavename = outputdir + "scorer.pt"
        torch.save(scorer.state_dict(), scorersavename)
        initembeddersavename = outputdir + "initembedder.pt"
        torch.save(initembedder.state_dict(),initembeddersavename)
    
for epoch in tqdm(range(epoch_start, args.epochs + 1), desc='Epoch'): # tqdm
    print("Training")
    
    # Training stage
    initembedder.train()
    embedder.train()
    scorer.train()
    
    # Calculate Accuracy & Epoch Loss
    total_pred, total_label, train_loss, train_ce_loss, train_recon_loss, initembedder, embedder, scorer, optim, scheduler = run_epoch(args, data, dataloader, initembedder, embedder, scorer, optim, scheduler, loss_fn, opt="train")
    total_pred = torch.cat(total_pred)
    total_label = torch.cat(total_label, dim=0)
    pred_cls = torch.argmax(total_pred, dim=1)
    train_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    scheduler.step()
    print("%d epoch: Training loss : %.4f (%.4f, %.4f) / Training acc : %.4f\n" % (epoch, train_loss, train_ce_loss, train_recon_loss, train_acc))
    with open(outputdir + "log_train.txt", "+a") as f:
        f.write("%d epoch: Training loss : %.4f (%.4f, %.4f) / Training acc : %.4f\n" % (epoch, train_loss, train_ce_loss, train_recon_loss, train_acc))
        
    # Test ===========================================================================================================================================================================
    if epoch % test_epoch == 0:
        initembedder.eval()
        embedder.eval()
        scorer.eval()
        
        with torch.no_grad():
            total_pred, total_label, eval_loss, eval_ce_loss, eval_recon_loss, initembedder, embedder, scorer, optim, scheduler = run_epoch(args, data, validdataloader, initembedder, embedder, scorer, optim, scheduler, loss_fn, opt="valid")
        # Calculate Accuracy & Epoch Loss
        total_label = torch.cat(total_label, dim=0)
        total_pred = torch.cat(total_pred)
        pred_cls = torch.argmax(total_pred, dim=1)
        eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
        y_test = total_label.cpu().numpy()
        pred = pred_cls.cpu().numpy()
        confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
        with open(outputdir + "log_valid_micro.txt", "+a") as f:
            f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, eval_loss, eval_ce_loss, eval_recon_loss, accuracy,precision,recall,f1))
        confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
        with open(outputdir + "log_valid_confusion.txt", "+a") as f:
            for r in range(args.output_dim):
                for c in range(args.output_dim):
                    f.write(str(confusion[r][c]))
                    if c == args.output_dim -1 :
                        f.write("\n")
                    else:
                        f.write("\t")
        with open(outputdir + "log_valid_macro.txt", "+a") as f:               
            f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, eval_loss, eval_ce_loss, eval_recon_loss, accuracy,precision,recall,f1))

        if best_eval_acc < eval_acc:
            print(best_eval_acc)
            best_eval_acc = eval_acc
            patience = 0
            if args.evaltype == "test" or args.save_epochs > 0:
                print("Model Save")
                modelsavename = outputdir + "embedder.pt"
                torch.save(embedder.state_dict(), modelsavename)
                scorersavename = outputdir + "scorer.pt"
                torch.save(scorer.state_dict(), scorersavename)
                initembeddersavename = outputdir + "initembedder.pt"
                torch.save(initembedder.state_dict(),initembeddersavename)
        else:
            patience += 1

        if patience > args.patience:
            break
        
        torch.save({
            'epoch': epoch,
            'embedder': embedder.state_dict(),
            'scorer' : scorer.state_dict(),
            'initembedder' : initembedder.state_dict(),
            'scheduler' : scheduler.state_dict(),
            'optimizer': optim.state_dict(),
            'best_eval_acc' : best_eval_acc,
            'patience' : patience
            }, outputdir + "checkpoint.pt")

if args.evaltype == "test":
    print("Test")
    
    initembedder.load_state_dict(torch.load(outputdir + "initembedder.pt")) # , map_location=device
    embedder.load_state_dict(torch.load(outputdir + "embedder.pt")) # , map_location=device
    scorer.load_state_dict(torch.load(outputdir + "scorer.pt")) # , map_location=device
    
    initembedder.eval()
    embedder.eval()
    scorer.eval()

    with torch.no_grad():
        total_pred, total_label, test_loss, test_ce_loss, test_recon_loss, initembedder, embedder, scorer = run_test_epoch(args, data, testdataloader, initembedder, embedder, scorer, loss_fn)
    # Calculate Accuracy & Epoch Loss
    total_label = torch.cat(total_label, dim=0)
    total_pred = torch.cat(total_pred)
    pred_cls = torch.argmax(total_pred, dim=1)
    eval_acc = torch.eq(pred_cls, total_label).sum().item() / len(total_label)
    y_test = total_label.cpu().numpy()
    pred = pred_cls.cpu().numpy()
    confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='micro', outputdim=args.output_dim)
    with open(outputdir + "log_test_micro.txt", "+a") as f:
        f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, test_loss, test_ce_loss, test_recon_loss, accuracy,precision,recall,f1))
    confusion, accuracy, precision, recall, f1 = utils.get_clf_eval(y_test, pred, avg='macro', outputdim=args.output_dim)
    with open(outputdir + "log_test_confusion.txt", "+a") as f:
        for r in range(args.output_dim):
            for c in range(args.output_dim):
                f.write(str(confusion[r][c]))
                if c == args.output_dim -1 :
                    f.write("\n")
                else:
                    f.write("\t")
    with open(outputdir + "log_test_macro.txt", "+a") as f:               
        f.write("{} epoch:Test Loss:{} ({}, {})/Accuracy:{}/Precision:{}/Recall:{}/F1:{}\n".format(epoch, test_loss, test_ce_loss, test_recon_loss, accuracy,precision,recall,f1))

if os.path.isfile(outputdir + "checkpoint.pt"):
    os.remove(outputdir + "checkpoint.pt")

