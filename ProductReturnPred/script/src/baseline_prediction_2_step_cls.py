import pandas as pd
import pickle
import numpy as np
from sklearn import metrics
# import matplotlib.pylab as plt
from src.functions import f_point_5


class BaseLinePredictor:
    """
    A baseline prediction class
    """
    def __init__(self, method='Jaccard', type='Unnormalized'):
        """
        The constructor of the baseline predictor
        :param method: only Jaccard method is supported
        :param type: one of the two options ('Unnormalized', 'Normalized')
        """
        self.method = method
        self.type = type
        self.h_train = None
        self.bsk_label_train = None
        self.__union_train = None
        self.ratio = None
        self.step = 1

    def fit(self, h_train, r_train, bsk_label_train,
            multi_item_bsk = None, ratio = None, step = 2):
        """
        The train method of class
        :param h_train: a csr_matrix
        :param bsk_label_train: a single column data frame
        :return: None
        """
        multi_item_bsk = (h_train>1).nonzero()[0]
        multi_idx = np.zeros(h_train.shape[0], dtype=bool)
        multi_idx[multi_item_bsk] = True

        h_train = h_train.sign()
        self.h_train = h_train
        self.r_train = r_train
        self.bsk_label_train = bsk_label_train.values
        self.__union_train = h_train.sum(1)

        if self.type == "Normalized":
            if ratio is None:
                self.ratio = bsk_label_train[multi_idx].sum()[0]/multi_idx.sum() /\
                    (bsk_label_train[np.logical_not(multi_idx)].sum()[0]/np.logical_not(multi_idx).sum())
            else:
                self.ratio = ratio
            self.bsk_label_train = self.bsk_label_train.astype(np.float)
            self.bsk_label_train[multi_idx, :] = self.bsk_label_train[multi_idx, :] / self.ratio
        else:
            self.ratio = 1

        self.step = step

    def predict(self, h_test, r_test):
        """
        Predict the result based on given h_test
        :param h_test: a csr_matrix
        :return: a list of prediction (continuous score)
        """
        multi_item_bsk = (h_test>1).nonzero()[0]
        h_test_sign = h_test.sign()
        wgt = np.ones((h_test_sign.shape[0]))
        if self.type == "Normalized":
            #assert test_bsk_name is not None, "test_bsk_name must be supplied for normalization"
            wgt[multi_item_bsk] = self.ratio
        h_mat = self.h_train.sign()
        intersect = h_mat * h_test_sign.T

        union2 = h_test_sign.sum(1)
        r_mat = self.r_train.sign()
        pred = []
        total_return_rate = (r_mat.sum(0)/h_mat.sum(0)).A1
        for i in range(len(union2)):
            ja = intersect[:, i] / (self.__union_train + union2[i] - intersect[:, i])  # Jaccard Index
            pred_bsk = (ja.T * self.bsk_label_train/ja.sum())[0,0] * wgt[i]
            if self.step == 1:
                pred_bsk = 1
            idx = self.bsk_label_train[:,0] > 0
            with np.errstate(divide='ignore',invalid='ignore'):
                pred_prod = ja[idx].T * r_mat[idx, :]/ \
                            (ja[idx].T * h_mat[idx, :]) * pred_bsk

            idx_purchase = (h_test_sign[i, :].todense().A1 > 0)
            pred_prod = pred_prod[:, idx_purchase].A1
            nan_prd = np.isnan(pred_prod)
            pred_prod[nan_prd] = total_return_rate[idx_purchase][nan_prd]

            if self.type == "Normalized":
                wgt_prod = (h_test[i, idx_purchase] > 1).todense().A1.astype(np.float) + 1
                pred_prod = pred_prod * wgt_prod

            pred.append((pred_bsk, pred_prod, r_test[i, idx_purchase].todense().A1 > 0))

        pred_rst = pd.DataFrame(pred, columns=['pred_prob', 'pred_prob_prod', 'obs_prod'])
        return pred_rst

    def pred_test_based_on_valid(self, h_validate, bsk_label_valid, h_test, bsk_label_test):
        """
        :param h_validate:
        :param bsk_label_valid:
        :param h_test:
        :return: optimal prec, recall, f_0.5, auc, fpr, tpr
        """
        pred_val = self.predict(h_validate, bsk_label_valid.index.values)

        prec, rec, thr = metrics.precision_recall_curve(bsk_label_valid, pred_val)
        f = f_point_5(prec, rec)
        thr_opt = thr[np.argmax(f)]

        pred_test = self.predict(h_test, bsk_label_test.index.values)
        prec, rec, thr = metrics.precision_recall_curve(bsk_label_test, pred_test)
        f = f_point_5(prec, rec)
        idx = np.where(thr <= thr_opt)[0][-1]

        fpr, tpr, _ = metrics.roc_curve(bsk_label_test, pred_test)
        auc = metrics.auc(fpr, tpr)

        return prec[idx], rec[idx], f[idx], auc, fpr, tpr, thr[idx]

    def pred_test_based_on_valid_prod(self, h_validate, bsk_label_valid, r_validate,
                                 h_test, bsk_label_test, r_test):
        """
        :param h_validate:
        :param bsk_label_valid:
        :param order_no_validate:
        :param r_validate:
        :param h_test:
        :param bsk_label_test:
        :param order_no_test:
        :param r_test:
        :param bs:
        :param phis:
        :return:
        """
        pred_val = self.predict(h_validate, r_validate)

        obs_prod = np.array([item for l in pred_val['obs_prod'] for item in l])
        pred_prod = np.array([item for l in pred_val['pred_prob_prod'] for item in l])

        prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)
        f = f_point_5(prec, rec)
        thr_opt = thr[np.argmax(f)]

        pred_test = self.predict(h_test, r_test)
        obs_prod = np.array([item for l in pred_test['obs_prod'] for item in l])
        pred_prod = np.array([item for l in pred_test['pred_prob_prod'] for item in l])

        prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)
        f = f_point_5(prec, rec)
        idx = np.where(thr <= thr_opt)[0][-1]

        fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod)
        auc = metrics.auc(fpr, tpr)

        return prec[idx], rec[idx], f[idx], auc, fpr, tpr, thr[idx]

 
if __name__ == "__main__":
    import matplotlib.pylab as plt
    # load data
    with open("../data/h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open("../data/r_train.pkl", 'rb') as f:
        r_train = pickle.load(f)

    with open("../data/h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open("../data/h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)

    with open("../data/r_validate.pkl", 'rb') as f:
        r_validate = pickle.load(f)
    with open("../data/r_test.pkl", 'rb') as f:
        r_test = pickle.load(f)

    bsk_label_train = pd.read_pickle("../data/bsk_label_train.pkl")
    bsk_label_valid = pd.read_pickle("../data/bsk_label_validate.pkl")
    bsk_label_test = pd.read_pickle("../data/bsk_label_test.pkl")

    # unnormalized

    p = BaseLinePredictor()
    p.fit(h_train, r_train, bsk_label_train)

    prec, rec, f, auc, fpr, tpr, thr = \
        p.pred_test_based_on_valid_prod(h_validate, bsk_label_valid, r_validate,
                                        h_test, bsk_label_test, r_test)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr:")
    print("%f, %f, %f, %f, %f" % (auc, prec, rec, f, thr))

    # normalized
    p_1 = BaseLinePredictor(type="Normalized")
    p_1.fit(h_train, r_train, bsk_label_train, 2)

    prec_1, rec_1, f_1, auc_1, fpr_1, tpr_1, thr =\
        p_1.pred_test_based_on_valid_prod(h_validate, bsk_label_valid, r_validate,
                                        h_test, bsk_label_test, r_test)

    # plt.plot(fpr_1, tpr_1)
    # plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr:")
    print("%f, %f, %f, %f, %f" % (auc_1, prec_1, rec_1, f_1, thr))