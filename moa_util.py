import os

from sklearn.metrics import log_loss
import random
import pandas as pd
import torch
import numpy as np


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_zero_count(root_dir):
    result_columns = []
    train_targets = pd.read_csv(os.path.join(root_dir, 'train_targets_scored.csv'))
    columns = train_targets.columns
    one_counts = []
    zero_ratio = []
    data = train_targets.values
    nrows = data.shape[0]
    for col_index in range(1, data.shape[1]):
        num = np.sum(data[:, col_index])
        one_counts.append(num)
        zero_ratio.append(1 - num / nrows)
    for i in range(len(one_counts)):
        if one_counts[i] <= 1:
            result_columns.append(columns[i])
    return result_columns, zero_ratio


def reset_to_zero(submit, zero_ratio, proportion=1.0):
    data = submit.values
    nrows = data.shape[0]
    for col_index in range(1, data.shape[1]):
        n = int(nrows * zero_ratio[col_index - 1] * proportion)
        submit.iloc[get_min_n_indexes(data[:, col_index], n), col_index] = 0
    return submit


def get_min_n_indexes(arr, n):
    import copy
    temp = copy.deepcopy(list(arr))
    min_index = []
    for _ in range(n):
        num = min(temp)
        index = temp.index(num)
        temp[index] = np.inf
        min_index.append(index)
    return min_index


def write_result(root_dir, preds, target_columns, filename=None):
    filename = 'submission.csv' if filename is None else filename
    origin_test_data = pd.read_csv(os.path.join(root_dir, 'test_features.csv'))
    submit = pd.read_csv(os.path.join(root_dir, 'sample_submission.csv'))
    submit[target_columns] = preds
    # 将cp_type为ctl_vehicle的行设为0
    submit.loc[origin_test_data['cp_type'] == 'ctl_vehicle', target_columns] = 0

    zero_columns, zero_ratio = get_zero_count(root_dir)
    submit = reset_to_zero(submit, zero_ratio, 0.01)

    submit.to_csv(filename, index=False)


def write_val_result(root_dir, drop_ctl, preds, target_columns, filename):
    origin_test_data = pd.read_csv(os.path.join(root_dir, 'test_features.csv'))
    submit = pd.read_csv(os.path.join(root_dir, 'train_targets_scored.csv'))
    X_train = pd.read_csv(os.path.join(root_dir, 'train_features.csv'))
    if drop_ctl:
        X_train = X_train[X_train['cp_type'] != 'ctl_vehicle']
    submit = submit.iloc[X_train.index]
    submit.reset_index(drop=True, inplace=True)
    submit[target_columns] = preds
    submit.to_csv(filename, index=False)


def ensemble_result(root_dir, drop_ctl, result_dir, out_filename=None, val=False, sample_size=0, weights=None):
    # 获取target columns
    sample_submit = pd.read_csv(os.path.join(root_dir, 'sample_submission.csv'))
    del sample_submit['sig_id']
    target_columns = [col for col in sample_submit.columns]

    # 融合所有模型的预测结果
    filenames = os.listdir(result_dir)
    if weights is None:
        weights = [float(name.split("#")[1]) for name in filenames]
    count = len(filenames)
    if count <= 0:
        return
    if not val:
        seed_preds = np.zeros((len(sample_submit), len(target_columns), count))
    else:
        seed_preds = np.zeros((sample_size, len(target_columns), count))

    val_scores = {}
    for i in range(len(filenames)):
        temp = pd.read_csv(os.path.join(result_dir, filenames[i]))
        seed_preds[:, :, i] = temp[target_columns] * (weights[i] / np.sum(weights))
        if val:
            val_score = calculate_overall_loss(root_dir, drop_ctl, os.path.join(result_dir, filenames[i]))
            val_scores[filenames[i][0:filenames[i].index('_submission')]] = val_score

    submit = np.sum(seed_preds, axis=2)
    # 写出最后的结果
    if not val:
        write_result(root_dir, submit, target_columns, out_filename)
    else:
        print(val_scores)
        write_val_result(root_dir, drop_ctl, submit, target_columns, out_filename)


def calculate_overall_loss(root_dir, drop_ctl, filename='val_predicted.csv'):
    y_true = pd.read_csv(os.path.join(root_dir, 'train_targets_scored.csv'))
    X_train = pd.read_csv(os.path.join(root_dir, 'train_features.csv'))
    if drop_ctl:
        X_train = X_train[X_train['cp_type'] != 'ctl_vehicle']
    y_true = y_true.iloc[X_train.index]
    y_true.reset_index(drop=True, inplace=True)
    del y_true['sig_id']

    y_pred = pd.read_csv(filename)
    del y_pred['sig_id']
    y_true = y_true.values
    y_pred = y_pred.values

    score = 0
    for i in range(y_true.shape[1]):
        score_ = log_loss(y_true[:, i], y_pred[:, i])
        score += score_ / y_true.shape[1]
    return score
