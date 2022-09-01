from matplotlib import pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import argparse
import os

# Evaluation ------------------------------------------------------------------------------------------------------
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

def update_result(propertyname, result_dict, property2eval):
    for k,v in result_dict.items():
        property2eval[propertyname][k] = v
    return property2eval

# Main ----------------------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Argparse Tutorial')

parser.add_argument('--input_dir', default='DBLP2', type=str)
parser.add_argument('--k', default=10000, type=int)
parser.add_argument('--input_name', default='whole_data_onehot', type=str)
parser.add_argument('--base', default='allwogiven', type=str, help="allwogiven / allworankgiven")
args = parser.parse_args()

property2eval = defaultdict(dict)
baseline2eval = defaultdict(dict)

# temp
print("K =", args.k, " BASE =", args.base)
print("Read Data")
data = pd.read_csv("../dataset/" + args.input_dir + "/" + args.input_name + "_" + str(args.k) + ".txt")
columns = data.columns
# Data
base = ['node index', 'hedgename']

identity = [c for c in columns if "identity_" in c]
degree = [c for c in columns if ("degree_" in c) and ("degree_rank" not in c)]
eigenvec = [c for c in columns if ("eigenvec_" in c) and ("eigenvec_rank" not in c)]
kcore = [c for c in columns if ("kcore_" in c) and ("kcore_rank" not in c)]
pagerank = [c for c in columns if ("pagerank_" in c) and ("pagerank" not in c)]

degree_rank = [c for c in columns if "degree_rank_" in c]
eigenvec_rank = [c for c in columns if "eigenvec_rank_" in c]
kcore_rank = [c for c in columns if "kcore_rank_" in c]
pagerank_rank = [c for c in columns if "pagerank_rank_" in c]

deg_avg = [c for c in columns if "deg_avg_" in c]
deg_max = [c for c in columns if "deg_max_" in c]
deg_min = [c for c in columns if "deg_min_" in c]

pos = ['pos']

column_dict = {
    "identity" : identity,
    "degree" : degree,
    "eigenvec" : eigenvec,
    "kcore" : kcore,
    "pagerank" : pagerank,
    "degree_rank" : degree_rank,
    "eigenvec_rank" : eigenvec_rank,
    "kcore_rank" : kcore_rank,
    "pagerank_rank" : pagerank_rank,
    "deg_avg" : deg_avg,
    "deg_max" : deg_max,
    "deg_min" : deg_min,
}

target_cols = []
if args.base == "allwogiven":
    target_cols = [name for name in column_dict.keys()]
elif args.base == 'allworankgiven':
    target_cols = [name for name in column_dict.keys() if "rank" not in name]

# Use All Columns -----------------------------------------------------------------------------------------------------------------------------
print("Find All")
usecols = ['pos', 'hedgename']
for c in target_cols:
    usecols += column_dict[c]
data = pd.read_csv("../dataset/" + args.input_dir + "/" + args.input_name + "_" + str(args.k) + ".txt", usecols=usecols)

## Split
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
    
with open("../dataset/" + args.input_dir + "/test_hindex_" + str(args.k) + ".txt", "w") as f:
    for idx in test_hindex:
        f.write(str(idx) + "\n")
print(data.head(2))
column_indexes = [idx for (idx,col) in enumerate(data.columns) if col != 'hedgename' and col != 'pos']
X_train = data.iloc[train_index, column_indexes].values
X_test = data.iloc[test_index, column_indexes].values
Y_train = data.iloc[train_index, -1].values
Y_test = data.iloc[test_index, -1].values

# Scaling
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

## Baseline
baseline0_pred = [np.random.randint(3) for _ in range(len(Y_test))]
confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, baseline0_pred, avg='micro')
confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, baseline0_pred, avg='macro')
result_dict = { "accuracy" : accuracy,
           "precision_micro" : precision_micro,
           "precision_macro" : precision_macro,
           "recall_micro" : recall_micro,
           "recall_macro" : recall_macro,
           "f1_micro" : f1_micro,
           "f1_macro" : f1_macro    
}
for evalname in result_dict:
    baseline2eval["baseline0"][evalname] = result_dict[evalname]

baseline1_pred = [1 for _ in range(len(Y_test))]
confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, baseline1_pred, avg='micro')
confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, baseline1_pred, avg='macro')
result_dict = { "accuracy" : accuracy,
           "precision_micro" : precision_micro,
           "precision_macro" : precision_macro,
           "recall_micro" : recall_micro,
           "recall_macro" : recall_macro,
           "f1_micro" : f1_micro,
           "f1_macro" : f1_macro    
}
for evalname in result_dict:
    baseline2eval["baseline1"][evalname] = result_dict[evalname]

## Train MLP
print("Train MLP for all")
classifier = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=300, activation='relu', solver='adam', learning_rate_init=0.001, random_state=1, early_stopping=True, n_iter_no_change=20)
#Fitting the training data to the network
classifier.fit(X_train, Y_train)
plt.plot(classifier.loss_curve_)
#     plt.show()
if os.path.isdir("../figures/" + args.input_dir + "/MLP_onehot/") is False:
    os.mkdir("../figures/" + args.input_dir + "/MLP_onehot/")
plt.savefig("../figures/" + args.input_dir + "/MLP_onehot/loss_all_" + args.base + "_" + str(args.k) + ".jpg", bbox_inches='tight')
plt.close()
y_pred = classifier.predict(X_test)
# fig_confusion_mat(Y_test, y_pred, property_name)
confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, y_pred, avg='micro')
confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, y_pred, avg='macro')
result_dict = { "accuracy" : accuracy,
           "precision_micro" : precision_micro,
           "precision_macro" : precision_macro,
           "recall_micro" : recall_micro,
           "recall_macro" : recall_macro,
           "f1_micro" : f1_micro,
           "f1_macro" : f1_macro    
}
with open(dirname + "All_confusion.txt", "w") as f:
    for r in range(3):
        for c in range(3):
            f.write(str(confusion[r][c]))
            if c == 2:
                f.write("\n")
            else:
                f.write("\t")
property2eval = update_result("All", result_dict, property2eval)
print("F1 Score = %.2f , %.2f" % (f1_micro, f1_macro))

# Remove each feature -------------------------------------------------------------------------------------------------    
for target_col in target_cols:
    usecols = ['pos']
    for c in target_cols:
        if c != target_col:
            usecols += column_dict[c]
    print("Target Col =", target_col)
    data = pd.read_csv("dataset/" + args.input_dir + "/" + args.input_name + "_" + str(args.k) + ".txt", usecols=usecols)
    print("Use Cols")
    print(data.columns)

    # Split
    X_train = data.iloc[train_index, :-1].values
    X_test = data.iloc[test_index, :-1].values
    Y_train = data.iloc[train_index, -1].values
    Y_test = data.iloc[test_index, -1].values

    # Scaling
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)
    
    # Model
    classifier = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=300, activation='relu', solver='adam', learning_rate_init=0.001, random_state=1, early_stopping=True, n_iter_no_change=20)
    # Fitting the training data to the network
    classifier.fit(X_train, Y_train)
    plt.plot(classifier.loss_curve_)
    # plt.show()
    if os.path.isdir("figures/" + args.input_dir + "/MLP_onehot/") is False:
        os.mkdir("figures/" + args.input_dir + "/MLP_onehot/")
    plt.savefig("figures/" + args.input_dir + "/MLP_onehot/loss_" + target_col + "_" + args.base + "_" + str(args.k) + ".jpg", bbox_inches='tight')
    plt.close()
    y_pred = classifier.predict(X_test)
    # fig_confusion_mat(Y_test, y_pred, property_name)
    confusion, accuracy, precision_micro, recall_micro, f1_micro = get_clf_eval(Y_test, y_pred, avg='micro')
    confusion, accuracy, precision_macro, recall_macro, f1_macro = get_clf_eval(Y_test, y_pred, avg='macro')
    result_dict = { "accuracy" : accuracy,
               "precision_micro" : precision_micro,
               "precision_macro" : precision_macro,
               "recall_micro" : recall_micro,
               "recall_macro" : recall_macro,
               "f1_micro" : f1_micro,
               "f1_macro" : f1_macro    
    }
    with open(dirname + target_col + "_confusion.txt", "w") as f:
        for r in range(3):
            for c in range(3):
                f.write(str(confusion[r][c]))
                if c == 2:
                    f.write("\n")
                else:
                    f.write("\t")
    property2eval = update_result(target_col, result_dict, property2eval)
    print("F1 Score = %.2f , %.2f" % (f1_micro, f1_macro))
    
# Save Results
xs = list(property2eval["All"].keys())
for prop in property2eval.keys():
    dirname = "nongraph_results/" + args.input_dir + "/MLP_onehot/"
    with open(dirname + prop + "_" + args.base + "_" + str(args.k) + ".txt", "w") as f:
        for x in xs:
            f.write(x + "\t" + str(property2eval[prop][x]) + "\n")

# property legend
plt.figure(dpi=100, figsize=(4,3))
for baselinename in baseline2eval.keys():
    plt.plot(xs, [baseline2eval[baselinename][x] for x in xs], linewidth=2, linestyle='dashed', label=baselinename)
for property_name in property2eval.keys():
    if property_name == "All":
        linewidth = 5
    else:
        linewidth = 2
    plt.plot(xs, [property2eval[property_name][x] for x in xs], linewidth=linewidth, label=property_name)
plt.legend(bbox_to_anchor=(1,1))
plt.xlabel("Evaluation Metric")
plt.title("Evaluation")
plt.xticks(rotation=90)
if os.path.isdir("figures/" + args.input_dir + "/MLP_onehot/") is False:
    os.makedirs("figures/" + args.input_dir + "/MLP_onehot/")
plt.savefig("figures/" + args.input_dir + "/MLP_onehot/evaluation_" + args.base + "_" + str(args.k) + ".jpg", bbox_inches='tight')
plt.close()

# x-axis is property
for x in xs:
    plt.figure(dpi=100, figsize=(4,3))
    val_all = property2eval["All"][x]
    for property_name in property2eval.keys():
        if property_name == "All":
            continue
        plt.bar(property_name, val_all - property2eval[property_name][x], color = '#7E6ECD')
    xmin, xmax = plt.xlim() 
    #plt.hlines(property2eval["All"][x], xmin, xmax, label="All", linewidth = 5)
    #for baselinename in baseline2eval.keys():
    #    plt.hlines(baseline2eval[baselinename][x], xmin, xmax, label=baselinename, linewidth = 5, linestyle='dashed')
    plt.title(x + " Metric Diff (%.2f)" % (property2eval["All"][x]))
    plt.ylabel("Importance")
    plt.xticks(rotation=90)
    if os.path.isdir("figures/" + args.input_dir + "/MLP_onehot/") is False:
        os.makedirs("figures/" + args.input_dir + "/MLP_onehot/")
    plt.savefig("figures/" + args.input_dir + "/MLP_onehot/" + x + "_" + args.base + "_" + str(args.k) + ".jpg", bbox_inches='tight')
    plt.close()
    