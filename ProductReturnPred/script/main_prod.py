import pickle
import numpy as np
import pandas as pd
import argparse
import os
# from scoop import futures

from src.pre_process_cls import SplitTrainValidateTest
from src.hg_prediction_2_step_cls import HypergraphPredictor
from src.hg_prediction_2_step_cls_unif import HypergraphPredictor as HypergraphPredictorUnif
from itertools import compress

parser = argparse.ArgumentParser(description='Argparse Tutorial')
parser.add_argument('--model', default="our_0")
parser.add_argument('--unif_flag', action='store_true', help="Run Hypergraph w/o Labels")
args = parser.parse_args()

config = "GroundTruth"
if args.unif_flag:
    config = "NoLabel"
    assert len(args.model) == 0
elif len(args.model) > 0:
    config = args.model
    assert args.unif_flag is False
input_path = '../data/'
output_path = '../data/split'
result_path = '../rst/' + config
if os.path.isdir(result_path + "/") is False:
    os.makedirs(result_path + "/")
    
seeds = range(3) # [1,2,0] #
step = 2 # step 1, conditional probability
         # step 2, marginal probability
         # for basket level prediction use 2
    
split = SplitTrainValidateTest(0.6, 0.2, unif_flag=args.unif_flag, model=args.model)
split.read_data(input_path)

for sd in seeds:
    print("Start of seed %d" % sd)
    # random split =======================================
    output_path_sd = output_path+str(sd)+str("/") + config + str("/")
    split.random_split(sd)
    split.export_data(output_path_sd)

    # Load data ==========================================
    with open(output_path_sd + "order_no_train.pkl", 'rb') as f:
        order_no_train = pickle.load(f)
    with open(output_path_sd + "khk_ean_train.pkl", 'rb') as f:
        khk_ean_train = pickle.load(f)
    bsk_label_train = pd.read_pickle(output_path_sd + "bsk_label_train.pkl")
    return_rate_train = pd.read_pickle(output_path_sd + "return_rate_train.pkl")
    with open(output_path_sd + "h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open(output_path_sd + "r_train.pkl", 'rb') as f:
        r_train = pickle.load(f)

    with open(output_path_sd + "h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open(output_path_sd + "r_validate.pkl", 'rb') as f:
        r_validate = pickle.load(f)
    with open(output_path_sd + "order_no_validate.pkl", 'rb') as f:
        order_no_validate = pickle.load(f)
    bsk_label_validate = pd.read_pickle(output_path_sd + "bsk_label_validate.pkl")


    with open(output_path_sd + "h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)
    with open(output_path_sd + "r_test.pkl", 'rb') as f:
        r_test = pickle.load(f)
    with open(output_path_sd + "order_no_test.pkl", 'rb') as f:
        order_no_test = pickle.load(f)
    bsk_label_test = pd.read_pickle(output_path_sd + "bsk_label_test.pkl")

    # Ground Truth ============================================================
    if step == 1:
        idx = np.where(bsk_label_validate['RET_Items'].values)[0]
        h_validate = h_validate[idx,:]
        order_no_validate = [order_no_validate[i] for i in idx]
        r_validate = r_validate[idx, :]
        bsk_label_validate = bsk_label_validate.iloc[idx,:]

        idx = np.where(bsk_label_test['RET_Items'].values)[0]
        h_test = h_test[idx,:]
        order_no_test = [order_no_test[i] for i in idx]
        r_test = r_test[idx, :]
        bsk_label_test = bsk_label_test.iloc[idx,:]

    # hyper graph ====================================
    if args.unif_flag:
        p = HypergraphPredictorUnif(max_num_test=1000, parallel="Single", n_cpu=15, chunk_size=1)
    else:
        p = HypergraphPredictor(max_num_test=1000, parallel="Single", n_cpu=15, chunk_size=1) # "Multi"
    p.fit(h_train, bsk_label_train, order_no_train, khk_ean_train,
          return_rate_train, r_train, ratio=None, step=step)
    #prec, _, _, _, _, _, _, _, _ = \
    prec, rec, f, auc, _, _, _, _, _ = \
        p.pred_test_based_on_valid_prod(h_validate, bsk_label_validate, order_no_validate, r_validate,
                                   h_test, bsk_label_test, order_no_test, r_test,
                                   [6, 7, 8, 9, 10],
                                   [0.8, 0.6, 0.4])
    with open(result_path + "/hypergraph.csv", 'a+') as ff:
        ff.write("%f, %f, %f, %f, %f\n" % (sd, prec, rec, f, auc))
    
    if os.path.isdir("../../train_results/Etail/res/") is False:
        os.makedirs("../../train_results/Etail/res/")
    outputname = args.model.split("_")[0]
    seed = args.model.split("_")[1]
    with open("../../train_results/Etail/res/{}.txt".format(outputname), "+a") as ff:
        ff.write(",".join([seed, str(sd), str(prec), str(rec), str(f), str(auc)]) + "\n")
        
#     with open(result_path + "/hypergraph.csv", 'a+') as ff:
#         ff.write("%f, %f, %f, %f, %f\n" % (sd, prec, rec, f, auc))


