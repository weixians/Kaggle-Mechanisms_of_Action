import os
import shutil
import sys

sys.path.append('../input/pytorch-tabnet')

import torch
import numpy as np
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor


class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)


class MyTabNetRegressor(TabNetRegressor):

    def smooth(self, y_true, n_classes, smoothing=0.001):
        assert 0 <= smoothing <= 1
        with torch.no_grad():
            y_true = y_true * (1 - smoothing) + torch.ones_like(y_true).to(self.device) * smoothing / n_classes
        return y_true

    def compute_loss(self, y_pred, y_true):
        y_true = self.smooth(y_true, y_pred.shape[1])
        return self.loss_fn(y_pred, y_true)


def train_tabnet(tag, params, fit_params, X_train, y_train, kfold, model_dir, it):
    best_val_losses = []
    for n, (tr, te) in enumerate(kfold.split(X_train, y_train)):
        print(f'Train fold {n + 1}')
        xtrain, xval = X_train[tr], X_train[te]
        ytrain, yval = y_train[tr], y_train[te]

        clf = MyTabNetRegressor(**params)
        clf.fit(
            xtrain, ytrain,
            eval_set=[(xval, yval)],
            **fit_params,
        )
        clf.save_model(os.path.join(model_dir, "{}_{}_it_{}.pth".format(tag, n + 1, it)))
        best_val_losses.append(min(clf.history['val_logits_ll']))
    return best_val_losses


def test_tabnet(tag, params, X_test, ntargets, nfolds, model_dir, it):
    seed_preds = np.zeros((len(X_test), ntargets, nfolds))

    for n in range(nfolds):
        clf = MyTabNetRegressor(**params)

        model_path = os.path.join(model_dir, "{}_{}_it_{}.pth.zip".format(tag, n + 1, it))
        # 如果在上传数据的情况下被kaggle解压了，重新压缩
        if not os.path.exists(model_path):
            path = os.path.join(model_dir, "{}_{}_it_{}.pth".format(tag, n + 1, it))
            model_path = "{}_{}_it_{}.pth".format(tag, n + 1, it)
            shutil.make_archive(model_path, "zip", path)
            model_path += '.zip'
        clf.load_model(model_path)

        fold_preds = clf.predict(X_test)
        seed_preds[:, :, n] = 1 / (1 + np.exp(-fold_preds))

    return np.mean(seed_preds, axis=2)
