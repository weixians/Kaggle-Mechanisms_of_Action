from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F
from torch import nn
import torch


class TransferHelper:
    def __init__(self, epochs):
        self.epochs = epochs
        self.epochs_per_step = 0
        self.frozen_layers = []

    def copy_without_top(self, device, model, Model, num_features, n_all_targets, params, num_targets_new):
        self.frozen_layers = []

        model_new = Model(num_features, n_all_targets, **params)
        model_new.load_state_dict(model.state_dict())

        model_depth = len(model_new.hidden_sizes) + 1
        # Freeze all weights
        for name, param in model_new.named_parameters():
            layer_index = name.split('.')[1][-1]

            if layer_index == model_depth:
                continue

            param.requires_grad = False

            # Save frozen layer names
            if layer_index not in self.frozen_layers:
                self.frozen_layers.append(layer_index)

        self.epochs_per_step = self.epochs // len(self.frozen_layers)

        # Replace the top layers with another ones
        model_new.sequential[-3] = nn.BatchNorm1d(model_new.hidden_sizes[-1])
        model_new.sequential[-2] = nn.Dropout(
            model_new.get_dropout_rate(model_new.dropout_rates, model_depth - 2))
        model_new.sequential[-1] = nn.utils.weight_norm(
            nn.Linear(model_new.hidden_sizes[-1], num_targets_new))
        model_new.to(device)
        return model_new

    def step(self, epoch, model):
        if len(self.frozen_layers) == 0:
            return

        if epoch % self.epochs_per_step == 0:
            last_frozen_index = self.frozen_layers[-1]

            # Unfreeze parameters of the last frozen layer
            for name, param in model.named_parameters():
                layer_index = name.split('.')[0][-1]

                if layer_index == last_frozen_index:
                    param.requires_grad = True

            del self.frozen_layers[-1]  # Remove the last layer as unfrozen


class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets: torch.Tensor, n_labels: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
                                           self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets, self.weight)

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class MoaDataset(torch.utils.data.Dataset):
    def __init__(self, df, targets, mode='train'):
        self.mode = mode
        self.data = df
        if mode == 'train':
            self.targets = targets

    def __getitem__(self, idx):
        if self.mode == 'train':
            return torch.FloatTensor(self.data[idx]), torch.FloatTensor(self.targets[idx])
        elif self.mode == 'test':
            return torch.FloatTensor(self.data[idx]), 0

    def __len__(self):
        return len(self.data)
