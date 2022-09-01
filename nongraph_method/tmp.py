from matplotlib import pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import f1_score
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import argparse

# Evaluation -------------------------------------------------------------------------------------------------------
def get_clf_eval(y_test, pred, avg='micro'):
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=avg)
    recall = recall_score(y_test, pred, average=avg)
    f1 = f1_score(y_test, pred, average=avg)
    print('Confusion Matrix')
    print(confusion)
    print('Accuracy:{}, Precision:{}, Recall:{}, F1:{}'.format(accuracy, precision, recall, f1))
    
    return accuracy, precision, recall, f1

def update_result(propertyname, result_dict, property2eval):
    for k,v in result_dict.items():
        property2eval[propertyname][k] = v
    return property2eval

# Main --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--input_dir', default='DBLP2', type=str)
parser.add_argument('--k', default=10000, type=int)
parser.add_argument('--input_name', default='whole_data', type=str)
# parser.add_argument('--base', default='allwgiven', type=str, help="allwgiven / allwogiven / allworank / allworankgiven")
args = parser.parse_args()

property2eval = defaultdict(dict) # "all" or "degree" ... -> "acc" or "precision_micro" ... -> "model name" 
baseline2eval = defaultdict(dict) # baseline_0 or baseline_1  -> "acc" or "precision_micro" ...

identity = ['node reindex']
nongiven_features = ['degree', 'eigenvec', 'kcore', 'pagerank']
hedge_degree_features = ['degree_avg', 'degree_max', 'degree_min', 'degree_sum']
hedge_eigenvec_features = ['eigenvec_avg', 'eigenvec_max', 'eigenvec_min', 'eigenvec_sum']
hedge_kcore_features = ['kcore_avg', 'kcore_max', 'kcore_min', 'kcore_sum']
hedge_pagerank_features = ['pagerank_avg', 'pagerank_max', 'pagerank_min', 'pagerank_sum']

hedge_avg_features = ['degree_avg', 'eigenvec_avg', 'kcore_avg', 'pagerank_avg'] 
hedge_max_features = ['degree_max', 'eigenvec_max', 'kcore_max', 'pagerank_max']
hedge_min_features = ['degree_min', 'eigenvec_min', 'kcore_min', 'pagerank_min']
hedge_sum_features = ['degree_sum', 'eigenvec_sum', 'kcore_sum', 'pagerank_sum']

hedge_features = ['degree_avg', 'degree_max', 'degree_min', 'degree_sum', 
                  'eigenvec_avg', 'eigenvec_max', 'eigenvec_min', 'eigenvec_sum',
                  'kcore_avg', 'kcore_max', 'kcore_min', 'kcore_sum',
                  'pagerank_avg', 'pagerank_max', 'pagerank_min', 'pagerank_sum']
# past = ['degree_avg', 'degree_max', 'degree_min']
rank_features = ['degree_rank', 'eigenvec_rank', 'kcore_rank', 'pagerank_rank']
given_features = ['affiliation', 'year', 'conf', 'cs', 'de', 'se', 'th']

all_cols = identity + nongiven_features + rank_features + hedge_features

# Use All Columns -----------------------------------------------------------------------------------------------------------------------------
usecols = ['pos', 'hedgename'] + all_cols
data = pd.read_csv("../dataset/" + args.input_dir + "/" + args.input_name + "_" + str(args.k) + ".txt", usecols=usecols)

## Encoding
## Label encoding location and salary
le = LabelEncoder()
if "affiliation" in usecols:
    data['affiliation'] = le.fit_transform(data['affiliation'])
if "conf" in usecols:
    data['conf'] = le.fit_transform(data['conf'])

## Split
# if os.path.isfile("../dataset/" + args.input_dir + "/test_hindex_" + str(args.k) + ".txt"):
#     test_hedgename = []
#     with open("../dataset/" + args.input_dir + "/test_hindex_" + str(args.k) + ".txt", "r") as f:
#         for line in f.readlines():
#             test_hedgename.append(line.rstrip())
#     test_index = []
#     for hedgename in test_hedgename:
#         test_index += data.index[data['hedgename'] == hedgename].tolist()
#     train_index = list(set(range(len(data))) - set(test_index))
    
# else:
hedgename_list = data['hedgename'].unique()
numhedges = list(range(len(hedgename_list)))
train_hindex, test_hindex = train_test_split(numhedges, test_size = 0.2, random_state = 21)
train_hedgename = hedgename_list[train_hindex]
test_hedgename = hedgename_list[test_hindex]
train_index = []
test_index = []
for hedgename in train_hedgename:
    train_index += data.index[data['hedgename'] == hedgename].tolist()
for hedgename in test_hedgename:
    test_index += data.index[data['hedgename'] == hedgename].tolist()
assert len(test_index) > 0
with open("../dataset/" + args.input_dir + "/test_hindex_" + str(args.k) + ".txt", "w") as f:
    for hedgename in test_hedgename:
        f.write(str(hedgename) + "\n")
        