import pickle
import numpy as np
import pandas as pd
import os

class SplitTrainValidateTest:
    def __init__(self, train_rate, validate_rate, unif_flag=False, model_flag=False):
        self.train_rate = train_rate
        self.validate_rate = validate_rate
        self.test_rate = 1-train_rate-validate_rate
        self.unif_flag = unif_flag
        self.model_flag = model_flag

    def read_data(self, src_folder='data/'):
        
        src_folder = os.path.normpath(src_folder) + os.sep

        with open(src_folder + "order_no.pkl", 'rb') as f:
            self.order_no = pickle.load(f)
        with open(src_folder + "style_color.pkl", 'rb') as f:
            self.khk_ean = pickle.load(f)
        
        if self.unif_flag:
            with open(src_folder + "h_unif_mat.pkl", 'rb') as f:
                self.h = pickle.load(f)
            with open(src_folder + "r_unif_mat.pkl", 'rb') as f:
                self.r = pickle.load(f)
        elif self.model_flag:
            with open(src_folder + "h_WHATsNet_mat.pkl", 'rb') as f:
                self.h = pickle.load(f)

            with open(src_folder + "r_mat.pkl", 'rb') as f:
                self.r = pickle.load(f)
        else:
            with open(src_folder + "h_mat.pkl", 'rb') as f:
                self.h = pickle.load(f)

            with open(src_folder + "r_mat.pkl", 'rb') as f:
                self.r = pickle.load(f)
                
        self.bsk_label = pd.DataFrame(self.r.sum(axis=1)>0, index=self.order_no, columns=['RET_Items'])
        if self.unif_flag:
            with open(src_folder + "h_mat.pkl", 'rb') as f:
                h_nunif = pickle.load(f)
            with open(src_folder + "r_mat.pkl", 'rb') as f:
                r_nunif = pickle.load(f)
            self.return_rate = pd.DataFrame(((r_nunif.sum(axis=0)+1)/(h_nunif.sum(axis=0)+1)).T, index=self.khk_ean, columns=['RET_Items'])
        else:
            self.return_rate = pd.DataFrame(((self.r.sum(axis=0)+1)/(self.h.sum(axis=0)+1)).T, index=self.khk_ean, columns=['RET_Items'])
            
    def random_split(self, seed = 1):
        return_rate = self.return_rate.loc[self.khk_ean, :]
        bsk_label = self.bsk_label.loc[self.order_no, :]

        # split train test sets ===============================
        m = len(self.order_no)
        np.random.seed(seed)
        rnd_idx = np.random.permutation(m)
        split_pt = int(m*self.train_rate)
        train_idx = rnd_idx[:split_pt]
        test_idx = rnd_idx[split_pt:]

        # train
        self.order_no_train = [self.order_no[i] for i in train_idx]
        h_train = self.h[train_idx, :]
        r_train = self.r[train_idx, :]
        self.bsk_label_train = bsk_label.iloc[train_idx, :]

        # remove the product that did not appear in train set
        zero_edges = h_train.sum(0).A1 == 0
        non_zero_edges = np.where(np.logical_not(zero_edges))[0]
        self.h_train = h_train[:, non_zero_edges]
        self.r_train = r_train[:, non_zero_edges]
        self.khk_ean_train = [self.khk_ean[i] for i in non_zero_edges]
        self.return_rate_train = return_rate.iloc[non_zero_edges, :]

        # validate & test

        h_test_tmp = self.h[test_idx, :][:, zero_edges]
        idx = h_test_tmp.sum(1).A1 == 0
        test_idx = test_idx[idx]  # only the baskets has all the products that appeared in train
        # removed 11% of the test data

        h_test = self.h[test_idx, :][:, non_zero_edges]
        r_test = self.r[test_idx, :][:, non_zero_edges]
        order_no_test = [self.order_no[i] for i in test_idx]
        bsk_label_test = bsk_label.iloc[test_idx, :]

        # test validate split
        split_pt = int(len(order_no_test) * (self.validate_rate/(self.validate_rate + self.test_rate)))
        self.h_validate = h_test[:split_pt, :]
        self.h_test = h_test[split_pt:, :]

        self.r_validate = r_test[:split_pt, :]
        self.r_test = r_test[split_pt:, :]

        self.order_no_validate = order_no_test[:split_pt]
        self.order_no_test = order_no_test[split_pt:]

        self.bsk_label_validate = bsk_label_test[:split_pt]
        self.bsk_label_test = bsk_label_test[split_pt:]


    def export_data(self, target_folder='data/clean'):
        target_folder = os.path.normpath(target_folder) + os.sep
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # write out the files
        with open(target_folder + "order_no_train.pkl", 'wb') as f:
            pickle.dump(self.order_no_train, f)
        with open(target_folder + "khk_ean_train.pkl", 'wb') as f:
            pickle.dump(self.khk_ean_train, f)
        with open(target_folder + "bsk_label_train.pkl", 'wb') as f:
            pickle.dump(self.bsk_label_train, f)
            
        with open(target_folder + "order_no_validate.pkl", 'wb') as f:
            pickle.dump(self.order_no_validate, f)
        with open(target_folder + "bsk_label_validate.pkl", 'wb') as f:
            pickle.dump(self.bsk_label_validate, f)
        with open(target_folder + "order_no_test.pkl", 'wb') as f:
            pickle.dump(self.order_no_test, f)
        with open(target_folder + "bsk_label_test.pkl", 'wb') as f:
            pickle.dump(self.bsk_label_test, f)

        with open(target_folder + "h_train.pkl", 'wb') as f:
            pickle.dump(self.h_train, f)
        with open(target_folder + "r_train.pkl", 'wb') as f:
            pickle.dump(self.r_train, f)
        with open(target_folder + "h_validate.pkl", 'wb') as f:
            pickle.dump(self.h_validate, f)
        with open(target_folder + "r_validate.pkl", 'wb') as f:
            pickle.dump(self.r_validate, f)
        with open(target_folder + "h_test.pkl", 'wb') as f:
            pickle.dump(self.h_test, f)
        with open(target_folder + "r_test.pkl", 'wb') as f:
            pickle.dump(self.r_test, f)
        
        with open(target_folder + "return_rate_train.pkl", 'wb') as f:
                pickle.dump(self.return_rate_train, f)
            

if __name__ == "__main__":
    split = SplitTrainValidateTest(0.6, 0.2)
    split.read_data('../data')
    seed = 0
    split.random_split(seed)
    split.export_data('../data/splitted_'+str(seed))