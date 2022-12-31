import pandas as pd
import pickle
import numpy as np
from sklearn import metrics, neighbors
# import matplotlib.pylab as plt
from src.functions import f_point_5, argmax


class knn_Predictor:
    """
    A knn prediction class
    """
    def __init__(self, k=None):
        """
        The constructor of the baseline predictor
        :param method: only Jaccard method is supported
        :param type: one of the two options ('Unnormalized', 'Normalized')
        """
        self.k = k
        self.h_train = None
        self.r_train = None
        self.bsk_label_train = None
        self.clf = None
        self.step = None

    def fit(self, h_train, bsk_label_train, r_train, step):
        """
        The train method of class
        :param h_train: a csr_matrix
        :param bsk_label_train: a single column data frame
        :return: None
        """
        assert self.k is not None, "k cannot be none before train"
        self.h_train = h_train.sign()
        self.r_train = r_train
        if isinstance(bsk_label_train, pd.DataFrame):
            bsk_label_train = bsk_label_train.values
        self.bsk_label_train = bsk_label_train
        self.clf = neighbors.KNeighborsClassifier(self.k, weights='uniform')
        self.clf.fit(h_train, np.squeeze(self.bsk_label_train))
        self.step = step

    def predict(self, h_test, r_test):
        """
        Predict the result based on given h_test
        :param h_test: a csr_matrix
        :return: a list of prediction (continuous score)
        """
        assert self.clf is not None, "The model need to be trained before used for prediction"
        h_test = h_test.sign()

        h_mat = self.h_train.sign()
        r_mat = self.r_train.sign()
        tot_ret_rate = (r_mat.sum(0)/h_mat.sum(0)).A1
        pred = []
        for i in range(h_test.shape[0]):
            nn = self.clf.kneighbors(h_test[i, :], self.k)[1][0]
            if self.step == 1:
                pred_bsk = 1
            else:
                pred_bsk = self.bsk_label_train[nn].sum() / nn.shape[0]
            idx = np.where(self.bsk_label_train[:, 0] > 0)[0]
            idx = np.intersect1d(idx, nn)
            with np.errstate(divide='ignore',invalid='ignore'):
                pred_prod = r_mat[idx, :].sum(0) / h_mat[idx, :].sum(0) * pred_bsk


            idx = (h_test[i, :].todense().A1 > 0)
            pred_prod = pred_prod[:, idx].A1
            nan_prd = np.isnan(pred_prod)
            pred_prod[nan_prd] = tot_ret_rate[idx][nan_prd]
            pred.append((pred_bsk, pred_prod, r_test[i, idx].todense().A1 > 0))

        pred_rst = pd.DataFrame(pred, columns=['pred_prob', 'pred_prob_prod', 'obs_prod'])
        return pred_rst



    def pred_test_based_on_valid_prod(self, h_validate, bsk_label_valid, r_validate,
                                 h_test, bsk_label_test, r_test, ks):
        """
        Loop through a ks is given, otherwise, k could be specified before calling this method
        :param h_validate:
        :param bsk_label_valid:
        :param h_test:
        :return: optimal prec, recall, f_0.5, auc, fpr, tpr
        """

        best_f = []
        best_auc = []
        thr_opt = []
        for k in ks:
            self.k = k
            self.fit(self.h_train, self.bsk_label_train, self.r_train, step=self.step)
            pred_val = self.predict(h_validate, r_validate)

            obs_prod = np.array([item for l in pred_val['obs_prod'] for item in l])
            pred_prod = np.array([item for l in pred_val['pred_prob_prod'] for item in l])
            prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)

            f = f_point_5(prec, rec)
            thr_opt.append(thr[np.argmax(f)])
            best_f.append(np.max(f))
            fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod, pos_label=True)
            auc = metrics.auc(fpr, tpr)
            best_auc.append(auc)

        idx_k = argmax(best_f, best_auc)[0]
        self.k = ks[idx_k]
        self.fit(self.h_train, self.bsk_label_train, self.r_train, self.step)
        pred_test = self.predict(h_test, r_test)

        obs_prod = np.array([item for l in pred_test['obs_prod'] for item in l])
        pred_prod = np.array([item for l in pred_test['pred_prob_prod'] for item in l])
        prec, rec, thr = metrics.precision_recall_curve(obs_prod, pred_prod, pos_label=True)
        f = f_point_5(prec, rec)
        idx_f = np.where(thr <= thr_opt[idx_k])[0][-1]

        fpr, tpr, _ = metrics.roc_curve(obs_prod, pred_prod)
        auc = metrics.auc(fpr, tpr)

        return prec[idx_f], rec[idx_f], f[idx_f], auc, fpr, tpr, thr[idx_f], ks[idx_k]


if __name__ == "__main__":

    with open("../data/h_train.pkl", 'rb') as f:
        h_train = pickle.load(f)
    with open("../data/r_train.pkl", 'rb') as f:
        r_train = pickle.load(f)
    with open("../data/h_validate.pkl", 'rb') as f:
        h_validate = pickle.load(f)
    with open("../data/h_test.pkl", 'rb') as f:
        h_test = pickle.load(f)
    bsk_label_train = pd.read_pickle('../data/bsk_label_train.pkl')
    bsk_label_valid = pd.read_pickle("../data/bsk_label_validate.pkl")
    bsk_label_test = pd.read_pickle('../data/bsk_label_test.pkl')

    with open("../data/r_validate.pkl", 'rb') as f:
        r_validate = pickle.load(f)
    with open("../data/r_test.pkl", 'rb') as f:
        r_test = pickle.load(f)
    # k-d tree
    p = knn_Predictor(k=5)
    p.fit(h_train, bsk_label_train, r_train, step=2)
    prec, rec, f, auc, fpr, tpr, thr, k = \
        p.pred_test_based_on_valid_prod(h_validate, bsk_label_valid, r_validate,
                                        h_test, bsk_label_test, r_test,
                                        [3, 5, 10, 15, 20, 25])

    #  plt.plot(fpr, tpr)
    # plt.plot([0, 1], [0, 1], 'b--')

    print("auc, prec, rec, f1, thr, k:")
    print("%f, %f, %f, %f, %f, %f" % (auc, prec, rec, f, thr, k))