import torch 
import numpy as np
import random
from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import argparse

def reorder_name(name):    
    if len(name) == 0:
        print(name)
        return name
    _name = name.split(",")
    _name = sorted(_name)
    
    flag = 0
    idx = 0
    for s in ["degree", "eigenvec", "kcore", "pagerank"]:
        if s in _name[idx]:
            aggflag = False
            for agg in ["sum", "avg", "min", "max"]:
                if agg in _name[idx]:
                    _name[idx] = s + "_" + agg
                    aggflag = True
                    break
            if aggflag is False:
                _name[idx] = s
            flag += 1
            idx += 1
    name = ",".join(_name)
    
    if flag == 4:
        for s in ["sum", "avg", "min", "max"]:
            if s in name:
                return "all" + s
        else:
            return "all"
    else:
        return name

def make_fname(args):
    fname = "vf_" + reorder_name(args.vfeat_input) # + "_" + args.binnings_v
    if len(args.efeat_input) > 0:
        fname += "_ef_" + reorder_name(args.efeat_input) # + "_" + args.binnings_e
    if len(args.use_vweight_input) > 0:
        fname += "_wt_" + reorder_name(args.use_vweight_input) + "_" + reorder_name(args.use_eweight_input)
    if len(args.vrank_input) > 0:
        fname += "_vr_" + args.vrank_input
        # fname += "_vr_" + reorder_name(args.vrank_input)
    if len(args.erank_input) > 0:
        fname += "_er_" + reorder_name(args.erank_input)
        
    return fname 

def parse_args():
    parser = argparse.ArgumentParser(description='Argparse Tutorial')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run_only_test', action='store_true')
    parser.add_argument('--fix_seed', action='store_true')
    parser.add_argument('--do_svd', action='store_true')
    parser.add_argument('--kfold', default=1, type=int)
    parser.add_argument('--log', action='store_true') # log time
    parser.add_argument('--analyze_att', action='store_true') # analyze attention
    parser.add_argument('--recalculate', action='store_true')
    parser.add_argument('--n_trials', default=50, type=int)

    # training parameter
    parser.add_argument('--bs', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--optimizer', type=str, default="adam")
    parser.add_argument('--scheduler', default='multi', type=str)
    parser.add_argument('--lr', default=0.0005, type=float)
    #parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--dropout', default=0.7, type=float)
    parser.add_argument('--test_epoch', default=10, type=int)
    parser.add_argument('--save_epochs', default=-1, type=int)
    parser.add_argument('--rw', type=float, default=0.01,
                        help='The weight of reconstruction of adjacency matrix loss. Default is ')

    # data parameter
    parser.add_argument('--dataset_name', default='DBLP2', type=str)
    parser.add_argument('--inputdir', default='dataset/', type=str)
    parser.add_argument('--exist_hedgename', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--k', default=10000, type=int)
    parser.add_argument('--use_sample_wt', action='store_true')
    parser.add_argument('--use_exp_wt', action='store_true')
    parser.add_argument('--output_dim', default=3, type=int)
    parser.add_argument('--evaltype', default='valid', type=str)
    parser.add_argument('--vsampling', default=-1, type=int) # node -> hedge
    parser.add_argument('--sampling', default=-1, type=int) # hedge -> node
    parser.add_argument('--valid_inputname', default='valid_hindex', type=str)
    parser.add_argument('--test_inputname', default='test_hindex', type=str)
    #     feature
    parser.add_argument('--vfeat_input', default='', type=str)
    parser.add_argument('--vfeat_usecols', default='', type=str)
    parser.add_argument('--binning', default=0.1, type=float) # clustering data
    parser.add_argument('--binnings_v', default='1', type=str)
    parser.add_argument('--efeat_input', default='', type=str)
    parser.add_argument('--efeat_usecols', default='', type=str)
    parser.add_argument('--binnings_e', default='1', type=str)
    #     weight
    parser.add_argument('--use_vweight_input', default='', type=str)
    parser.add_argument('--use_eweight_input', default='', type=str)
    parser.add_argument('--exp_num', default=1, type=int)
    parser.add_argument('--exp_wt', action='store_true')
    parser.add_argument('--alpha_e', default=0, type=float)
    parser.add_argument('--alpha_v', default=0, type=float)
    #     pe
    parser.add_argument('--pe', default='', type=str, help="positional encoding option for ITRE, ShawRE, RafRE, DEAdd; KD, KPRW, DESPD, DERW")
    parser.add_argument('--vrank_input', default='', type=str, help="positional encoding input for RankQ or RankAdd")
    parser.add_argument('--erank_input', default='', type=str, help="deprecated")
    parser.add_argument('--whole_ranking', action='store_true')
    
    # model parameter
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--num_inds', default=4, type=int)
    parser.add_argument('--embedder', default='hnhn', type=str)
    parser.add_argument('--scorer', default='sm', type=str)
    parser.add_argument('--scorer_num_layers', default=1, type=int)
    parser.add_argument('--att_type_v', default='', type=str, help="RankQ, RankAdd, ITRE, ShawRE, RafRE, DEAdd, pure, NoAtt")
    parser.add_argument('--agg_type_v', default='', type=str, help="PrevQ, pure, pure2, AvgAgg")
    parser.add_argument('--att_type_e', default='', type=str, help="RankQ, RankAdd, pure, NoAtt")
    parser.add_argument('--agg_type_e', default='', type=str, help="PrevQ, pure, pure2, AvgAgg")
    parser.add_argument('--num_att_layer', default=1, type=int, help="Set the number of Self-Attention layers")

#     parser.add_argument('--encode_type', default='', type=str)
#     parser.add_argument('--decode_type', default='', type=str)
#     parser.add_argument('--num_isab_layers', default=2, type=int)
#     parser.add_argument('--num_mab_layers', default=0, type=int)
#     parser.add_argument('--num_pma_layers', default=0, type=int)
#     parser.add_argument('--num_sab_layers', default=0, type=int)
#     parser.add_argument('--partial', action='store_true')

    parser.add_argument('--dim_vertex', default=128, type=int) # 400
    parser.add_argument('--dim_edge', default=128, type=int) # 400
    parser.add_argument('--dim_hidden', default=64, type=int)
    
    # random walk
    parser.add_argument('-l', '--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 40.')
    parser.add_argument('-r', '--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('-k', '--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')
    parser.add_argument('--p', type=float, default=2,
                        help='Return hyperparameter. Default is 1.')
    parser.add_argument('--q', type=float, default=0.25,
                        help='Inout hyperparameter. Default is 1.')
    parser.add_argument('-w', '--walk', type=str, default='hyper',
                        help='The walk type, empty stands for rw on hypergraph')
    
    parser.add_argument('--label_percent', default=-1, type=float)
    parser.add_argument('--val_ratio', default=0.25, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--splits', action='store_true')

    args = parser.parse_args()

    args.num_layers = int(os.environ.get('num_layers', 1))
    args.num_att_layer = int(os.environ.get('num_att_layer', 1))
    args.bs = str(os.environ.get('bs', 64))
    args.lr = float(os.environ.get('lr', 0.001))

    args.fname = make_fname(args)
    
    # vfeat
    if len(args.vfeat_input) == 0:
        args.vfeat_input = []
    else:
        args.vfeat_input = args.vfeat_input.split(",")
    # efeat
    if len(args.efeat_input) == 0:
        args.efeat_input = []
        args.use_efeat = False
    else:
        args.efeat_input = args.efeat_input.split(",")
        args.use_efeat = True
    # vrank
    if len(args.vrank_input) == 0:
        args.vrank_input = []
        args.rankflag = False
    else:
        args.vrank_input = args.vrank_input.split(",")
        args.rankflag = True
    # erank
    if len(args.erank_input) == 0:
        args.erank_input = []
    else:
        args.erank_input = args.erank_input.split(",")
        assert len(args.vrank_input) == len(args.erank_input)
    args.rank_dim = len(args.vrank_input)
    # binning
    args.binnings_v = args.binnings_v.split(",")
    for i in range(len(args.binnings_v)):
        if args.binnings_v[i] == '1':
            args.binnings_v[i] = True
        else:
            args.binnings_v[i] = False
    args.binnings_e = args.binnings_e.split(",")
    for i in range(len(args.binnings_e)):
        if args.binnings_e[i] == '1':
            args.binnings_e[i] = True
        else:
            args.binnings_e[i] = False
    # assert len(args.vfeat_input) == len(args.binnings)
    
    args.vfeat_usecols = args.vfeat_usecols.split(",")
    args.efeat_usecols = args.efeat_usecols.split(",")
    
    # Setting File Save Name -----------------------------------------------------------------------------
    args.embedder_name = args.embedder
    if len(args.att_type_v) > 0 and len(args.agg_type_v) > 0:
        args.embedder_name += "-{}-{}".format(args.att_type_v, args.agg_type_v)
    if len(args.att_type_e) > 0 and len(args.agg_type_e) > 0:
        args.embedder_name += "-{}-{}".format(args.att_type_e, args.agg_type_e)
    if len(args.att_type_v) > 0 and args.att_type_v != "NoAtt":
        args.embedder_name += "_atnl{}".format(args.num_att_layer)
    elif len(args.att_type_e) > 0 and args.att_type_e != "NoAtt":
        args.embedder_name += "_atnl{}".format(args.num_att_layer)
    args.embedder_name += "_nl{}".format(args.num_layers)
    
    args.scorer_name = "{}_snl{}".format(args.scorer, args.scorer_num_layers)
    args.model_name = args.embedder_name + "_" + args.scorer_name
    
    if args.embedder == "hcha":
        args.param_name = "hd_{}_od_{}_do_{}_lr_{}_ni_{}_sp_{}".format(args.dim_hidden, args.dim_edge, args.dropout, args.lr, args.num_inds, args.sampling)
    else:
        args.param_name = "hd_{}_od_{}_bs_{}_lr_{}_ni_{}_sp_{}".format(args.dim_hidden, args.dim_edge, args.bs, args.lr, args.num_inds, args.sampling)
    if len(args.vrank_input) == 1:
        args.param_name += "_vr_{}".format(args.vrank_input[0].split("_")[0])
    if len(args.pe) > 0:
        args.param_name += "_pe_{}".format(args.pe)
    # ---------------------------------------------------------------------------------------------------
    return args

def get_clf_eval(y_test, pred, avg='micro'):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=avg)
    recall = recall_score(y_test, pred, average=avg)
    f1 = f1_score(y_test, pred, average=avg)
    print('Confusion Matrix')
    print(confusion)
    print('Accuracy:{}, Precision:{}, Recall:{}, F1:{}'.format(accuracy, precision, recall, f1))
    return confusion, accuracy, precision, recall, f1

def walkpath2str(walk):
	return [list(map(str, w)) for w in walk]

class Word2Vec_Skipgram_Data_Empty(object):
	"""Word2Vec model (Skipgram)."""
	
	def __init__(self):
		return
	
	def next_batch(self):
		"""Train the model."""
		
		return 0, 0, 0, 0, 0

import subprocess
import json 
DEFAULT_ATTRIBUTES = ( 
    'index', 'uuid', 'name', 'timestamp', 'memory.total', 
    'memory.free', 'memory.used', 'utilization.gpu', 'utilization.memory' ) 
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]
    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def move_to_device(device, A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG ):
    A = A.to(device)
    V = V.to(device)
    E = E.to(device)
    D_V = D_V.to(device)
    D_V_INV = D_V_INV.to(device)
    D_E = D_E.to(device)
    D_E_INV = D_E_INV.to(device)
    V_H = V_H.to(device)
    E_H = E_H.to(device)
    TG = TG.to(device)

    return A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG 

def train_batch(batch_index, data, embedder, scorer, embedder_optimizer, scorer_optimizer, criterion, device):
    A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG = data.get_batch(0, batch_index)
    A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG = move_to_device(device, A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG)

    embedder.train()
    scorer.train()
    embedder_optimizer.zero_grad()
    scorer_optimizer.zero_grad()

    node_embeddings, hedge_embeddings = embedder(V, E, A, D_V, D_E, D_V_INV, D_E_INV)
    node_embeddings = torch.matmul(V_H, node_embeddings)
    hedge_embeddings = torch.matmul(E_H, hedge_embeddings)
    
    pred = scorer(torch.cat((node_embeddings, hedge_embeddings), 1))
    loss = criterion(pred, TG)
    loss.backward()
    embedder_optimizer.step()
    scorer_optimizer.step()

    # for name, param in embedder.named_parameters():
    #     print(name)
    #     print(param.requires_grad)
    #     print(param.grad.data)
    # print()
    # print()
    # for name, param in scorer.named_parameters():
    #     print(name)
    #     print(param.requires_grad)
    #     print(param.grad.data)

    return loss.item() * TG.shape[0], TG.shape[0]

def evaluation(data, embedder, scorer, criterion, type, device, bfsdepth=None):
    
    batch_num = data.generate_batch(type, bfsdepth=bfsdepth)
    eval_loss = 0
    eval_acc = 0
    eval_prec = 0
    eval_f1 = 0
    eval_recall = 0
    eval_total = 0
    # print(batch_num)
    # print(len(data.batches))
    for batch_index in range(batch_num):
        embedder.eval()
        scorer.eval()

        A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG = data.get_batch(type, batch_index)
        A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG = move_to_device(device, A, V, E, D_V, D_V_INV, D_E, D_E_INV, V_H, E_H, TG)
        node_embeddings, hedge_embeddings = embedder(V, E, A, D_V, D_E, D_V_INV, D_E_INV)

        node_embeddings = torch.matmul(V_H, node_embeddings)
        hedge_embeddings = torch.matmul(E_H, hedge_embeddings)

        pred = scorer(torch.cat((node_embeddings, hedge_embeddings), 1))
        loss = criterion(pred, TG)
        eval_loss += loss.item() * TG.shape[0] 
        eval_total += TG.shape[0]

        pred_label = torch.argmax(pred, dim=1).data.to('cpu')
        TG = TG.data.to('cpu')
        eval_acc += accuracy_score(TG, pred_label) * TG.shape[0]
        eval_prec += precision_score(TG, pred_label, average='macro') * TG.shape[0] #'micro
        eval_recall += recall_score(TG, pred_label, average='macro') * TG.shape[0]
        eval_f1 += f1_score(TG, pred_label, average='macro') * TG.shape[0]

    eval_loss /= eval_total
    eval_acc /= eval_total
    eval_prec /= eval_total
    eval_recall /= eval_total
    eval_f1 /= eval_total

    return eval_loss, eval_acc, eval_prec, eval_recall, eval_f1

def competitor_evaluation(data, criterion, device, output_postfix):    
    types = ["valid", "test"]
    fully_random_result = {}
    random_result = {}
    split_num = 1

    with open("results/baseline{}.txt".format(output_postfix), "w") as f:
        f.write("type,data,acc,precision,recall,f1score\n")

    for i, type in enumerate(types):
        _, _, _, _, _, D_E, _, _, _, TG = data.get_data(i)
        split = (D_E.shape[0] // split_num) + 1
        total_eval1 = defaultdict(float)
        total_eval2 = defaultdict(float)
        total_length = 0
        prev_idx = 0

        for split_idx in range(split_num):
            pred1 = []
            pred2 = []
            end_idx = min(D_E.shape[0], split * (split_idx + 1))
            for hidx in range(split * split_idx, end_idx):
                hsize = int(D_E[hidx, hidx].item())
                if hsize > 1:
                    two, zero = random.sample(range(hsize), 2)
                else:
                    two = -1
                    zero = 0
                for j in range(hsize):
                    sample_label = random.randint(0,2)
                    pred1.append(sample_label)
                    
                    if two == j:
                        pred2.append(2)
                    elif zero == j:
                        pred2.append(0)
                    else:
                        pred2.append(1)

            _TG = TG[prev_idx:prev_idx + len(pred1)]
            prev_idx += len(pred1)
            pred1 = torch.tensor(np.array(pred1), dtype=torch.float, requires_grad=False)
            pred2 = torch.tensor(np.array(pred2), dtype=torch.float, requires_grad=False)

            acc_1 = accuracy_score(_TG, pred1)
            prec_1 = precision_score(_TG, pred1, average='macro')
            recall_1 = recall_score(_TG, pred1, average='macro')
            f1_1 = f1_score(_TG, pred1, average='macro')
            total_eval1["acc"] += acc_1
            total_eval1["prec"] += prec_1
            total_eval1["recall"] += recall_1
            total_eval1["f1"] += f1_1

            acc_2 = accuracy_score(_TG, pred2)
            prec_2 = precision_score(_TG, pred2, average='macro')
            recall_2 = recall_score(_TG, pred2, average='macro')
            f1_2 = f1_score(_TG, pred2, average='macro')
            total_eval2["acc"] += acc_2
            total_eval2["prec"] += prec_2
            total_eval2["recall"] += recall_2
            total_eval2["f1"] += f1_2
            
            total_length += (end_idx - (split * split_idx))
        
        
        with open("results/baseline{}.txt".format(output_postfix), "+a") as f:
            # f.write("type,data,acc,precision,recall,f1score\n")
            f.write(",".join(["fully_random", type, str(total_eval1["acc"]), str(total_eval1["prec"]), str(total_eval1["recall"]), str(total_eval1["f1"])]) + "\n")
            f.write(",".join(["random", type, str(total_eval2["acc"]), str(total_eval2["prec"]), str(total_eval2["recall"]), str(total_eval2["f1"])]) + "\n")
            
        # print("[Comp] " + type + " loss : %.4f    %.4f" % (fully_random_result[type], random_result[type]))

    # with open("results/result{}.txt".format(output_postfix), "+a") as f:
    #     for type in types:
    #         f.write("Fully Random " + type + " loss = " + str(fully_random_result[type]) + "\n")
    #     for type in types:
    #         f.write("Random " + type + " loss = " + str(random_result[type]) + "\n")

