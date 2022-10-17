import pandas as pd
import numpy as np
import os
import math
import argparse
import random
import sys
import utils
from matplotlib import pyplot as plt
from collections import defaultdict

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score
from scipy.spatial import distance

import preprocess.data_load as dl

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

def get_result(Y_test, pred, dirname, name):
    confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, pred, avg='micro')
    confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, pred, avg='macro')
    result_dict = {
               "precision_micro" : precision_micro,
               "precision_macro" : precision_macro,
               "recall_micro" : recall_micro,
               "recall_macro" : recall_macro,
               "f1_micro" : f1_micro,
               "f1_macro" : f1_macro
    }
    
    base_output = dirname + "{}_confusion.txt".format(name)
    output_dim = confusion.shape[1]
    with open(base_output, "w") as f:
        for r in range(output_dim):
            for c in range(output_dim):
                f.write(str(confusion[r][c]))
                if c == (output_dim - 1):
                    f.write("\n")
                else:
                    f.write("\t")
    return result_dict

# Main ===========================================================================================
args = utils.parse_args()
outputdir = "./nongraph_results/" + args.dataset_name + "_" + str(args.k) + "/"
if args.evaltype == "test":
    assert args.fix_seed
    outputdir += "test/" + str(args.seed) + "/"
else:
    outputdir += "valid/"
if os.path.isdir(outputdir) is False:
    os.makedirs(outputdir)

if args.fix_seed:
    random.seed(args.seed)
    np.random.seed(args.seed)
    
# Data -----------------------------------------------------------------------------
data = dl.Hypergraph(args, args.dataset_name)
data.split_data(args.val_ratio, args.test_ratio)
train_data = data.get_data(0)
if args.evaltype == "test":
    target_data = data.get_data(2)
else:
    target_data = data.get_data(1)
    
### Baseline 0
y_answer = []
y_pred = []
node2ansrole = {}
node2predrole = {}
for h in target_data:
    for v, vpos in zip(data.hedge2node[h], data.hedge2nodepos[h]):
        y_answer.append(vpos)
        vpos_pred = np.random.randint(args.output_dim)
        y_pred.append(vpos_pred)
        if v not in node2predrole:
            node2predrole[v] = np.zeros(args.output_dim)
            node2ansrole[v] = np.zeros(args.output_dim)
        node2predrole[v][vpos_pred] += 1
        node2ansrole[v][vpos] += 1
baseline0_result = get_result(y_answer, y_pred, outputdir, "baselineU")
# Calculate JSD-Divergence
jsd_div_list = []
try:
    for v in node2predrole.keys():
        jsd_div = distance.jensenshannon(node2ansrole[v], node2predrole[v])
        jsd_div_list.append(jsd_div)
    with open(outputdir + "baselineU_jsd_div.txt", "w") as f:
        for jsd_div in jsd_div_list:
            f.write(str(jsd_div) + "\n")
    print("Average: %.4f" % (np.mean(jsd_div_list)) )
except:
    print("except on jsd-divergence")
    with open("EXCEPTION.txt", "+a") as f:
        f.write(outputdir + " error on jsd-divergence\n")
        
### Baseline 1
label_dist = defaultdict(int)
y_pred = []
node2predrole = {}
total = 0
for h in train_data:
    for vpos in data.hedge2nodepos[h]:
        label_dist[int(vpos)] += 1.0
        total += 1
for lab in label_dist.keys():
    label_dist[lab] = label_dist[lab] / total
for h in target_data:
    for v, vpos in zip(data.hedge2node[h], data.hedge2nodepos[h]):
        vpos_pred = np.random.choice(3, 1, p=[label_dist[0], label_dist[1], label_dist[2]])[0]
        y_pred.append(vpos_pred)
        if v not in node2predrole:
            node2predrole[v] = np.zeros(args.output_dim)
        node2predrole[v][vpos_pred] += 1
baseline1_result = get_result(y_answer, y_pred, outputdir, "baselineP")
# Calculate JSD-Divergence
jsd_div_list = []
try:
    for v in node2predrole.keys():
        jsd_div = distance.jensenshannon(node2ansrole[v], node2predrole[v])
        jsd_div_list.append(jsd_div)
    with open(outputdir + "baselineP_jsd_div.txt", "w") as f:
        for jsd_div in jsd_div_list:
            f.write(str(jsd_div) + "\n")
    print("Average: %.4f" % (np.mean(jsd_div_list)) )
except:
    print("except on jsd-divergence")
    with open("EXCEPTION.txt", "+a") as f:
        f.write(outputdir + " error on jsd-divergence\n")
        