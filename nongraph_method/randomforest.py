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
    
    return confusion, accuracy, precision, recall, f1

def get_according_to_label(y_test, pred):
    confusion = confusion_matrix(y_test, pred)
    rec_bar = []
    for i in range(confusion.shape[0]):
        row_sum = np.sum(confusion[i])
        ans = confusion[i,i]
        if row_sum == 0:
            rec_bar.append(0)
        else:
            rec_bar.append(ans / row_sum)
    recall_0 = rec_bar[0]
    recall_1 = rec_bar[1]
    recall_2 = rec_bar[2]
    
    prec_bar = []
    for i in range(confusion.shape[0]):
        col_sum = np.sum(confusion[:,i])
        ans = confusion[i,i]
        if col_sum == 0:
            prec_bar.append(0)
        else:
            prec_bar.append(ans / col_sum)
    precision_0  = prec_bar[0]
    precision_1 = prec_bar[1]
    precision_2 = prec_bar[2]
    
    return recall_0, precision_0, recall_1, precision_1, recall_2, precision_2

def update_result(propertyname, result_dict, property2eval):
    for k,v in result_dict.items():
        property2eval[propertyname][k] = v
    return property2eval

# Main --------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--input_dir', default='DBLP2', type=str)
parser.add_argument('--exist_hedgename', action='store_true')
parser.add_argument('--k', default=10000, type=int)
parser.add_argument('--input_name', default='whole_data', type=str)
parser.add_argument('--repeat_idx', default=0, type=int)
# parser.add_argument('--base', default='allwgiven', type=str, help="allwgiven / allwogiven / allworank / allworankgiven")
args = parser.parse_args()

property2eval = defaultdict(dict) # "all" or "degree" ... -> "acc" or "precision_micro" ... -> "model name" 
baseline2eval = defaultdict(dict) # baseline_0 or baseline_1  -> "acc" or "precision_micro" ...

identity = ['node reindex']
nongiven_features = ['degree', 'eigenvec', 'kcore', 'pagerank']

vrank_features = ['degree_vrank', 'eigenvec_vrank', 'kcore_vrank', 'pagerank_vrank']
erank_features = ['degree_erank', 'eigenvec_erank', 'kcore_erank', 'pagerank_erank']

# hedge_degree_features = ['degree_avg', 'degree_max', 'degree_min', 'degree_sum']
# hedge_eigenvec_features = ['eigenvec_avg', 'eigenvec_max', 'eigenvec_min', 'eigenvec_sum']
# hedge_kcore_features = ['kcore_avg', 'kcore_max', 'kcore_min', 'kcore_sum']
# hedge_pagerank_features = ['pagerank_avg', 'pagerank_max', 'pagerank_min', 'pagerank_sum']

# hedge_avg_features = ['degree_avg', 'eigenvec_avg', 'kcore_avg', 'pagerank_avg'] 
# hedge_max_features = ['degree_max', 'eigenvec_max', 'kcore_max', 'pagerank_max']
# hedge_min_features = ['degree_min', 'eigenvec_min', 'kcore_min', 'pagerank_min']
# hedge_sum_features = ['degree_sum', 'eigenvec_sum', 'kcore_sum', 'pagerank_sum']

# hedge_features = ['degree_avg', 'degree_max', 'degree_min', 'degree_sum', 
#                   'eigenvec_avg', 'eigenvec_max', 'eigenvec_min', 'eigenvec_sum',
#                   'kcore_avg', 'kcore_max', 'kcore_min', 'kcore_sum',
#                   'pagerank_avg', 'pagerank_max', 'pagerank_min', 'pagerank_sum']
# past = ['degree_avg', 'degree_max', 'degree_min']
given_features = ['affiliation', 'year', 'conf', 'cs', 'de', 'se', 'th']

all_cols = nongiven_features + vrank_features + erank_features # identity +  + hedge_features

# Use All Columns -----------------------------------------------------------------------------------------------------------------------------
usecols = ['pos', 'hedgename'] + all_cols
if args.repeat_idx > 0:
    inputname = "../dataset/{}/{}_{}_{}.txt".format(args.input_dir, args.input_name, args.k, args.repeat_idx)
else:
    inputname = "../dataset/{}/{}_{}.txt".format(args.input_dir, args.input_name, args.k)
data = pd.read_csv(inputname, usecols=usecols)

## Encoding
## Label encoding location and salary
le = LabelEncoder()
if "affiliation" in usecols:
    data['affiliation'] = le.fit_transform(data['affiliation'])
if "conf" in usecols:
    data['conf'] = le.fit_transform(data['conf'])

## Split
need_split_flag = False
if args.repeat_idx > 0:
    input_test_name = "../dataset/{}/test_hindex_{}_{}.txt".format(args.input_dir, args.k, args.repeat_idx)
    input_valid_name = "../dataset/{}/valid_hindex_{}_{}.txt".format(args.input_dir, args.k, args.repeat_idx)
else:
    input_test_name = "../dataset/{}/test_hindex_{}.txt".format(args.input_dir, args.k)
    input_valid_name = "../dataset/{}/valid_hindex_{}.txt".format(args.input_dir, args.k)
if os.path.isfile(input_test_name) is False:
    need_split_flag = True
if os.path.isfile(input_valid_name) is False:
    need_split_flag = True
if need_split_flag:
    hedgename_list = data['hedgename'].unique()
    numhedges = list(range(len(hedgename_list)))
    # train_hindex, test_hindex = train_test_split(numhedges, test_size = 0.2, random_state = 21)
    _train_hindex, test_hindex = train_test_split(numhedges, test_size=0.2, random_state=21)
    train_hindex, valid_hindex = train_test_split(_train_hindex, test_size=0.25, random_state=21)

    train_hedgename = hedgename_list[train_hindex]
    valid_hedgename = hedgename_list[valid_hindex]
    test_hedgename = hedgename_list[test_hindex]

    train_index = []
    valid_index = []
    test_index = []
    for hedgename in train_hedgename:
        train_index += data.index[data['hedgename'] == hedgename].tolist()
    for hedgename in valid_hedgename:
        valid_index += data.index[data['hedgename'] == hedgename].tolist()
    for hedgename in test_hedgename:
        test_index += data.index[data['hedgename'] == hedgename].tolist()
    assert len(valid_index) > 0
    assert len(test_index) > 0
    with open(input_valid_name, "w") as f:
        for hedgename in valid_hedgename:
            f.write(str(hedgename) + "\n")
    with open(input_test_name, "w") as f:
        for hedgename in test_hedgename:
            f.write(str(hedgename) + "\n")
else:
    valid_hedgename = []
    with open(input_valid_name, "r") as f:
        for line in f.readlines():
            if args.exist_hedgename:
                valid_hedgename.append(line.rstrip())
            else:
                valid_hedgename.append(int(line.rstrip()))
    test_hedgename = []
    with open(input_test_name, "r") as f:
        for line in f.readlines():
            if args.exist_hedgename:
                test_hedgename.append(line.rstrip())
            else:
                test_hedgename.append(int(line.rstrip()))
    valid_index = []
    test_index = []
    for hedgename in valid_hedgename:
        test_index += data.index[data['hedgename'] == hedgename].tolist()
    for hedgename in test_hedgename:
        test_index += data.index[data['hedgename'] == hedgename].tolist()
    train_index = list(set(range(len(data))) - set(test_index) - set(valid_index))
    assert len(test_index) > 0 or len(valid_index) > 0

assert len(set(train_index).intersection(set(test_index))) == 0
assert len(set(train_index).intersection(set(valid_index))) == 0
assert len(set(test_index).intersection(set(valid_index))) == 0

print(data.head(2))
column_indexes = [idx for (idx,col) in enumerate(data.columns) if col != 'hedgename' and col != 'pos']
X_train = data.iloc[train_index, column_indexes].values
X_test = data.iloc[test_index, column_indexes].values
Y_train = data.iloc[train_index, -1].values
Y_test = data.iloc[test_index, -1].values

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Baseline
basedirname = "../nongraph_results/" + args.input_dir + "_" + str(args.k) + "/"
if os.path.isdir(basedirname) is False:
    os.makedirs(basedirname)
    
baseline0_pred = [np.random.randint(3) for _ in range(len(Y_test))]
confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, baseline0_pred, avg='micro')
confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, baseline0_pred, avg='macro')
recall_0, precision_0, recall_1, precision_1, recall_2, precision_2 = get_according_to_label(Y_test, baseline0_pred)
result_dict = { "accuracy" : accuracy,
           "precision_micro" : precision_micro,
           "precision_macro" : precision_macro,
           "recall_micro" : recall_micro,
           "recall_macro" : recall_macro,
           "f1_micro" : f1_micro,
           "f1_macro" : f1_macro,
           "recall_0" : recall_0,
           "precision_0" : precision_0,
           "recall_1" : recall_1,
           "precision_1" : precision_1,
           "recall_2" : recall_2,
           "precision_2" : precision_2
}
if args.repeat_idx > 0:
    base_output = basedirname + "baseline0_confusion_{}.txt".format(args.repeat_idx)
else:
    base_output = basedirname + "baseline0_confusion.txt"
with open(base_output, "w") as f:
    for r in range(3):
        for c in range(3):
            f.write(str(confusion[r][c]))
            if c == 2:
                f.write("\n")
            else:
                f.write("\t")
for evalname in result_dict:
    baseline2eval["baseline0"][evalname] = result_dict[evalname]

if args.input_dir == "DBLP2":
    baseline1_pred = [1 for _ in range(len(Y_test))]
elif args.input_dir == "emailEnron":
    baseline1_pred = [0 for _ in range(len(Y_test))]
confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, baseline1_pred, avg='micro')
confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, baseline1_pred, avg='macro')
recall_0, precision_0, recall_1, precision_1, recall_2, precision_2 = get_according_to_label(Y_test, baseline1_pred)
result_dict = { "accuracy" : accuracy,
           "precision_micro" : precision_micro,
           "precision_macro" : precision_macro,
           "recall_micro" : recall_micro,
           "recall_macro" : recall_macro,
           "f1_micro" : f1_micro,
           "f1_macro" : f1_macro, 
           "recall_0" : recall_0,
           "precision_0" : precision_0,
           "recall_1" : recall_1,
           "precision_1" : precision_1,
           "recall_2" : recall_2,
           "precision_2" : precision_2  
}

if args.repeat_idx > 0:
    base_output = basedirname + "baseline1_confusion_{}.txt".format(args.repeat_idx)
else:
    base_output = basedirname + "baseline1_confusion.txt"
with open(base_output, "w") as f:
    for r in range(3):
        for c in range(3):
            f.write(str(confusion[r][c]))
            if c == 2:
                f.write("\n")
            else:
                f.write("\t")
for evalname in result_dict:
    baseline2eval["baseline1"][evalname] = result_dict[evalname]

for baselinename in baseline2eval.keys():
    if args.repeat_idx > 0:
        base_output = basedirname + baselinename + "_{}.txt".format(args.repeat_idx)
    else:
        base_output = basedirname + baselinename + ".txt"
    with open(base_output, "w") as f:
        for evalname in result_dict:
            f.write(evalname + "\t" + str(baseline2eval[baselinename][evalname]) + "\n")

dirname = "../nongraph_results/" + args.input_dir + "_" + str(args.k) + "/RandomForest/"
if os.path.isdir(dirname) is False:
    os.makedirs(dirname)
figdirname = "../figures/" + args.input_dir + "_" + str(args.k) + "/RandomForest/"
if os.path.isdir(figdirname) is False:
    os.makedirs(figdirname)
# Delete each column --------------------------------------------------------------------------------------------------------------------------
search = {"All" : all_cols,
         "vfeat" : nongiven_features,
         "vrank" : vrank_features,
         "erank" : erank_features,
#          "efeat" : hedge_features,
#          "noidentity": nongiven_features + rank_features + hedge_features,
         "vfeat_vrank": nongiven_features + vrank_features,
         "vfeat_erank": nongiven_features + erank_features,
         "vrank_erank": vrank_features + erank_features,
#          "vfeat_efeat": nongiven_features + hedge_features,
#          "vrank_efeat": rank_features + hedge_features,
#          "degree_efeat": hedge_degree_features,
#          "eigenvec_efeat": hedge_eigenvec_features,
#          "kcore_efeat": hedge_kcore_features,
#          "pagerank_efeat": hedge_pagerank_features,
#          "avg_efeat": hedge_avg_features,
#          "max_efeat": hedge_max_features,
#          "min_efeat": hedge_min_features,
#          "sum_efeat": hedge_sum_features
         }
for vrank_name in vrank_features:
    search[vrank_name] = nongiven_features + [vrank_name]
    
for name, col in search.items():
    print()
    print("*" * 20)
    print(name)
    print()
    
    usecols = ['pos'] + col
    # "../dataset/" + args.input_dir + "/" + args.input_name + "_" + str(args.k) + ".txt"
    data = pd.read_csv(inputname, usecols=usecols)
    print("Current Columns = ", data.columns)
    
    # Encoding
    # Label encoding location and salary
    le = LabelEncoder()
    if "affiliation" in usecols:
        data['affiliation'] = le.fit_transform(data['affiliation'])
    if "conf" in usecols:
        data['conf'] = le.fit_transform(data['conf'])
    
    # Split
    X_train = data.iloc[train_index, :-1].values
    X_test = data.iloc[test_index, :-1].values
    Y_train = data.iloc[train_index, -1].values
    Y_test = data.iloc[test_index, -1].values

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    # fig_confusion_mat(Y_test, Y_pred, target_col, "RandomForest")
    confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, Y_pred, avg='micro')
    confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, Y_pred, avg='macro')
    recall_0, precision_0, recall_1, precision_1, recall_2, precision_2 = get_according_to_label(Y_test, Y_pred)
    result_dict = { "accuracy" : accuracy,
               "precision_micro" : precision_micro,
               "precision_macro" : precision_macro,
               "recall_micro" : recall_micro,
               "recall_macro" : recall_macro,
               "f1_micro" : f1_micro,
               "f1_macro" : f1_macro,
               "recall_0" : recall_0,
               "precision_0" : precision_0,
               "recall_1" : recall_1,
               "precision_1" : precision_1,
               "recall_2" : recall_2,
               "precision_2" : precision_2  
    }
    if args.repeat_idx > 0:
        confmat_output = dirname + name + "_confusion_{}.txt".format(args.repeat_idx)
        feat_output = figdirname + "feature_ranking_{}_{}.txt".format(name, args.repeat_idx)
        featfig = figdirname + "feature_ranking_{}_{}.jpg".format(name, args.repeat_idx)
    else:
        confmat_output = dirname + name + "_confusion.txt"
        feat_output = figdirname + "feature_ranking_{}.txt".name(name)
        featfig = figdirname + "feature_ranking_{}.jpg".format(name)
    with open(confmat_output, "w") as f:
        for r in range(3):
            for c in range(3):
                f.write(str(confusion[r][c]))
                if c == 2:
                    f.write("\n")
                else:
                    f.write("\t")
    property2eval = update_result(name, result_dict, property2eval)
    
    # feature importance
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    with open(feat_output, "w") as f:
        for fidx in range(X_train.shape[1]):
            f.write("{}. feature {} ({:.3f})".format(fidx + 1, data.columns[indices][fidx], importances[indices][fidx]))

    plt.figure(dpi=100, figsize=(7,7))
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], color="r", yerr=std[indices], align='center')
    plt.xticks(range(X_train.shape[1]), data.columns[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.savefig(featfig, bbox_inches='tight')
    plt.close()

# Save Results
xs = list(property2eval["All"].keys())
for prop in property2eval.keys():
    if args.repeat_idx > 0:
        outputname = dirname + prop + "_{}.txt".format(args.repeat_idx)
    else:
        outputname = dirname + prop + ".txt"
    with open(outputname, "w") as f:
        for x in xs:
            f.write(x + "\t" + str(property2eval[prop][x]) + "\n")
            
# property legend
# plt.figure(dpi=100, figsize=(4,3))
# for baselinename in baseline2eval.keys():
#     plt.plot(xs, [baseline2eval[baselinename][x] for x in xs], linewidth=2, linestyle='dashed', label=baselinename)
# for property_name in ["All", "vfeat",  "vrank", "vfeat_vrank"]:
#     if property_name == "All":
#         linewidth = 5
#     else:
#         linewidth = 2
#     plt.plot(xs, [property2eval[property_name][x] for x in xs], linewidth=linewidth, label=property_name)
# plt.legend(bbox_to_anchor=(1,1))
# plt.xlabel("Evaluation Metric")
# plt.title("Evaluation")
# plt.xticks(rotation=90)
# plt.savefig(figdirname + "evaluation1.jpg", bbox_inches='tight')
# plt.close()

# property legend
# plt.figure(dpi=100, figsize=(4,3))
# for baselinename in baseline2eval.keys():
#     plt.plot(xs, [baseline2eval[baselinename][x] for x in xs], linewidth=2, linestyle='dashed', label=baselinename)
# for property_name in ["All", "noidentity", "vfeat_vrank", "vrank_efeat", "vfeat_efeat"]:
#     if property_name == "All":
#         linewidth = 5
#     else:
#         linewidth = 2
#     plt.plot(xs, [property2eval[property_name][x] for x in xs], linewidth=linewidth, label=property_name)
# plt.legend(bbox_to_anchor=(1,1))
# plt.xlabel("Evaluation Metric")
# plt.title("Evaluation")
# plt.xticks(rotation=90)
# plt.savefig(figdirname + "evaluation2.jpg", bbox_inches='tight')
# plt.close()

# # property legend
# plt.figure(dpi=100, figsize=(4,3))
# for baselinename in baseline2eval.keys():
#     plt.plot(xs, [baseline2eval[baselinename][x] for x in xs], linewidth=2, linestyle='dashed', label=baselinename)
# for property_name in ["All", "efeat", "degree_efeat", "eigenvec_efeat", "kcore_efeat", "pagerank_efeat"]:
#     if property_name == "All":
#         linewidth = 5
#     else:
#         linewidth = 2
#     plt.plot(xs, [property2eval[property_name][x] for x in xs], linewidth=linewidth, label=property_name)
# plt.legend(bbox_to_anchor=(1,1))
# plt.xlabel("Evaluation Metric")
# plt.title("Evaluation")
# plt.xticks(rotation=90)
# plt.savefig(figdirname + "evaluation3.jpg", bbox_inches='tight')
# plt.close()

# # property legend
# plt.figure(dpi=100, figsize=(4,3))
# for baselinename in baseline2eval.keys():
#     plt.plot(xs, [baseline2eval[baselinename][x] for x in xs], linewidth=2, linestyle='dashed', label=baselinename)
# for property_name in ["All", "efeat", "avg_efeat", "max_efeat", "min_efeat", "sum_efeat"]:
#     if property_name == "All":
#         linewidth = 5
#     else:
#         linewidth = 2
#     plt.plot(xs, [property2eval[property_name][x] for x in xs], linewidth=linewidth, label=property_name)
# plt.legend(bbox_to_anchor=(1,1))
# plt.xlabel("Evaluation Metric")
# plt.title("Evaluation")
# plt.xticks(rotation=90)
# plt.savefig(figdirname + "evaluation4.jpg", bbox_inches='tight')
# plt.close()

# # x-axis is property
# for x in xs:
#     plt.figure(dpi=100, figsize=(4,3))
#     val_all = property2eval["All"][x]
#     for property_name in property2eval.keys():
#         if property_name == "All":
#             continue
#         plt.bar(property_name, val_all - property2eval[property_name][x], color = '#7E6ECD')
#     xmin, xmax = plt.xlim() 
#     #plt.hlines(property2eval["All"][x], xmin, xmax, label="All", linewidth = 5)
#     #for baselinename in baseline2eval.keys():
#     #    plt.hlines(baseline2eval[baselinename][x], xmin, xmax, label=baselinename, linewidth = 5, linestyle='dashed')
#     plt.title(x + " Metric Diff (%.2f)" % (property2eval["All"][x]))
#     plt.ylabel("Importance")
#     plt.xticks(rotation=90)
#     if os.path.isdir("figures/" + args.input_dir + "/RandomForest/") is False:
#         os.makedirs("figures/" + args.input_dir + "/RandomForest/")
#     plt.savefig("figures/" + args.input_dir + "/RandomForest/" + x + "_" + args.base + "_" + str(args.k) + ".jpg", bbox_inches='tight')
#     plt.close()
    