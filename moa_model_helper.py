import os
import sys
from collections import OrderedDict

import torch
from torch import nn, optim
import torch.nn.functional as F

criterion = nn.BCEWithLogitsLoss()


class ModelDeep(nn.Module):
    def __init__(self, n_features, n_targets=206, hidden_size=512, dropratio=0.15):
        super(ModelDeep, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(n_features)
        self.dropout1 = nn.Dropout(dropratio)
        self.dense1 = nn.utils.weight_norm(nn.Linear(n_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropratio)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropratio)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm4 = nn.BatchNorm1d(hidden_size)
        self.dropout4 = nn.Dropout(dropratio)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm5 = nn.BatchNorm1d(hidden_size)
        self.dropout5 = nn.Dropout(dropratio)
        self.dense5 = nn.utils.weight_norm(nn.Linear(hidden_size, n_targets))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = self.relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)

        return x


class Model5122048(nn.Module):
    def __init__(self, n_features, n_targets=206, dropratio=0.4864405918342868):
        super(Model5122048, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(n_features)
        self.dropout1 = nn.Dropout(dropratio)
        self.dense1 = nn.utils.weight_norm(nn.Linear(n_features, 512))

        self.batch_norm2 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(dropratio)
        self.dense2 = nn.utils.weight_norm(nn.Linear(512, 2048))

        self.batch_norm3 = nn.BatchNorm1d(2048)
        self.dropout3 = nn.Dropout(dropratio)
        self.dense3 = nn.utils.weight_norm(nn.Linear(2048, n_targets))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


# 模型结构
class Model512(nn.Module):
    def __init__(self, n_features, n_targets=206, hidden_size=512, dropratio=0.15):
        super(Model512, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(n_features)
        self.dropout1 = nn.Dropout(dropratio)
        self.dense1 = nn.utils.weight_norm(nn.Linear(n_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropratio)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropratio)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, n_targets))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = self.relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class Model1500(nn.Module):
    def __init__(self, num_features, n_targets=206, hidden_size=1500, dropratio=0.2619422201258426):
        super(Model1500, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropratio)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropratio)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, n_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x


class Model15121075(nn.Module):
    def __init__(self, num_features, n_targets=206):
        super(Model15121075, self).__init__()
        self.hidden_sizes = [1500, 1250, 1000, 750]
        self.dropout_rates = [0.5, 0.35, 0.3, 0.25]

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dense1 = nn.Linear(num_features, self.hidden_sizes[0])

        self.batch_norm2 = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dropout2 = nn.Dropout(self.dropout_rates[0])
        self.dense2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])

        self.batch_norm3 = nn.BatchNorm1d(self.hidden_sizes[1])
        self.dropout3 = nn.Dropout(self.dropout_rates[1])
        self.dense3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])

        self.batch_norm4 = nn.BatchNorm1d(self.hidden_sizes[2])
        self.dropout4 = nn.Dropout(self.dropout_rates[2])
        self.dense4 = nn.Linear(self.hidden_sizes[2], self.hidden_sizes[3])

        self.batch_norm5 = nn.BatchNorm1d(self.hidden_sizes[3])
        self.dropout5 = nn.Dropout(self.dropout_rates[3])
        self.dense5 = nn.utils.weight_norm(nn.Linear(self.hidden_sizes[3], n_targets))

    def forward(self, x):
        x = self.batch_norm1(x)
        x = F.leaky_relu(self.dense1(x))

        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = F.leaky_relu(self.dense2(x))

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = F.leaky_relu(self.dense3(x))

        x = self.batch_norm4(x)
        x = self.dropout4(x)
        x = F.leaky_relu(self.dense4(x))

        x = self.batch_norm5(x)
        x = self.dropout5(x)
        x = self.dense5(x)
        return x


class ModelMlp(nn.Module):
    def __init__(self, num_features, n_targets=206, hidden_sizes=None, dropout_rates=None):
        super(ModelMlp, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout_rates = dropout_rates

        arr = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                # input layer
                arr.append((f'batch_norm{i + 1}', nn.BatchNorm1d(num_features)))
                arr.append((f'dense{i + 1}', nn.utils.weight_norm(nn.Linear(num_features, hidden_sizes[i]))))
                arr.append((f'activation{i + 1}', nn.modules.LeakyReLU()))

            else:
                # hidden layer
                arr.append((f'batch_norm{i + 1}', nn.BatchNorm1d(hidden_sizes[i - 1])))
                arr.append((f'dropout{i + 1}', nn.Dropout(self.get_dropout_rate(dropout_rates, i - 1))))
                arr.append((f'dense{i + 1}', nn.utils.weight_norm(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))))
                arr.append((f'activation{i + 1}', nn.modules.LeakyReLU()))
        # output layer
        i = len(hidden_sizes)
        arr.append((f'batch_norm{i + 1}', nn.BatchNorm1d(hidden_sizes[i - 1])))
        arr.append((f'dropout{i + 1}', nn.Dropout(self.get_dropout_rate(dropout_rates, i - 1))))
        arr.append((f'dense{i + 1}', nn.utils.weight_norm(nn.Linear(hidden_sizes[i - 1], n_targets))))

        self.sequential = nn.Sequential(OrderedDict(arr))

    def forward(self, x):
        x = self.sequential(x)
        return x

    def get_dropout_rate(self, dropout_rates, i):
        if isinstance(dropout_rates, list):
            if i >= len(dropout_rates):
                return dropout_rates[len(dropout_rates) - 1]
            else:
                return dropout_rates[i]
        else:
            return dropout_rates
