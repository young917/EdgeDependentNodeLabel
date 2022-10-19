import torch 
import numpy as np
import random
from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix
import argparse

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
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--test_epoch', default=10, type=int)
    parser.add_argument('--save_epochs', default=-1, type=int)
    parser.add_argument('--rw', type=float, default=0.01, help='The weight of reconstruction loss. But not used')

    # data parameter
    parser.add_argument('--inputdir', default='dataset/', type=str)
    parser.add_argument('--dataset_name', default='DBLP', type=str)
    parser.add_argument('--exist_hedgename', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--k', default=10000, type=int)
    parser.add_argument('--use_exp_wt', action='store_true')
    parser.add_argument('--output_dim', default=3, type=int)
    parser.add_argument('--evaltype', default='valid', type=str)
    parser.add_argument('--sampling', default=-1, type=int) # hedge -> node
    parser.add_argument('--valid_inputname', default='valid_hindex', type=str)
    parser.add_argument('--test_inputname', default='test_hindex', type=str)
    #     feature
    parser.add_argument('--binning', default=0, type=int) # downstream data
    # random walk from Hyper-SAGNN
    parser.add_argument('-l', '--walk-length', type=int, default=40, help='Length of walk per source')
    parser.add_argument('-r', '--num-walks', type=int, default=10, help='Number of walks per source')
    parser.add_argument('-k', '--window-size', type=int, default=10, help='Context size for optimization')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--p', type=float, default=2, help='Return hyperparameter')
    parser.add_argument('--q', type=float, default=0.25, help='Inout hyperparameter')
    parser.add_argument('-w', '--walk', type=str, default='hyper', help='The walk type, empty stands for rw on hypergraph')
    #     weight for HNHN
    parser.add_argument('--exp_num', default=1, type=int)
    parser.add_argument('--exp_wt', action='store_true')
    parser.add_argument('--alpha_e', default=0, type=float)
    parser.add_argument('--alpha_v', default=0, type=float)
    #     pe
    parser.add_argument('--pe', default='', type=str, help="positional encoding option for ITRE, ShawRE; KD, KPRW")
    parser.add_argument('--vorder_input', default='', type=str, help="positional encoding input for OrderPE")
    parser.add_argument('--whole_order', action='store_true')
    
    # model parameter
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--num_inds', default=4, type=int)
    parser.add_argument('--embedder', default='hnhn', type=str)
    parser.add_argument('--scorer', default='sm', type=str)
    parser.add_argument('--scorer_num_layers', default=1, type=int)
    parser.add_argument('--att_type_v', default='', type=str, help="OrderPE, ITRE, ShawRE, pure, NoAtt")
    parser.add_argument('--agg_type_v', default='', type=str, help="PrevQ, pure, pure2")
    parser.add_argument('--att_type_e', default='', type=str, help="OrderPE, pure, NoAtt")
    parser.add_argument('--agg_type_e', default='', type=str, help="PrevQ, pure, pure2")
    parser.add_argument('--num_att_layer', default=1, type=int, help="Set the number of Self-Attention layers")
    parser.add_argument('--pe_ablation', action='store_true')

    parser.add_argument('--dim_vertex', default=128, type=int)
    parser.add_argument('--dim_edge', default=128, type=int)
    parser.add_argument('--dim_hidden', default=256, type=int)
    
    parser.add_argument('--val_ratio', default=0.25, type=float)
    parser.add_argument('--test_ratio', default=0.2, type=float)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--splits', action='store_true')

    args = parser.parse_args()
    
    # vorder
    if len(args.vorder_input) == 0:
        args.vorder_input = []
        args.orderflag = False
    else:
        args.vorder_input = args.vorder_input.split(",")
        args.orderflag = True
    args.order_dim = len(args.vorder_input)
    
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
        
    if len(args.pe) > 0:
        args.param_name += "_pe_{}".format(args.pe)
    elif args.whole_order:
        args.param_name += "_pe_whole"
    elif args.pe_ablation and args.att_type_v == "OrderPE":
        args.param_name += "_pe_sab"
    # ---------------------------------------------------------------------------------------------------
    return args

def get_clf_eval(y_test, pred, avg='micro', outputdim=None):
    if outputdim is not None:
        confusion = confusion_matrix(y_test, pred, labels=np.arange(outputdim))
    else:
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
