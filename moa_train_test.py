import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from moa_train_helper import SmoothBCEwLogits, MoaDataset
# 将scored_only的weight_decay从3e-6修改为1e-6
WEIGHT_DECAY = {'ALL_TARGETS': 1e-5, 'SCORED_ONLY': 1e-6}
MAX_LR = {'ALL_TARGETS': 1e-2, 'SCORED_ONLY': 3e-3}
PCT_START = 0.1

criterion_train = SmoothBCEwLogits(smoothing=0.001)
criterion_val = nn.BCEWithLogitsLoss()


def train_model(tag, Model, params, X_train, Y_train, kfold, nepochs, batch_size, model_dir, device,
                early_stop, patience, it, learning_rate=2e-2, weight_decay=1e-5, scheduler_pattern='plateau',
                transfer_helper=None, Y_non_scored=None):
    n_all_targets = Y_train.shape[1]
    # first stage learning, use scored target + non scored targets
    if transfer_helper is not None and Y_non_scored is not None:
        Y = np.hstack((Y_train, Y_non_scored))
        n_all_targets = Y.shape[1]
        do_train(tag, Model, params, X_train, Y, kfold, nepochs, batch_size, model_dir,
                 device, early_stop, 5, it, MAX_LR['ALL_TARGETS'], WEIGHT_DECAY['ALL_TARGETS'], scheduler_pattern, None,
                 n_all_targets, 'transfer-pretrain')
    # use scored targets only to train
    do_train(tag, Model, params, X_train, Y_train, kfold, nepochs, batch_size, model_dir,
             device, early_stop, patience, it, MAX_LR['SCORED_ONLY'], WEIGHT_DECAY['SCORED_ONLY'], scheduler_pattern,
             transfer_helper, n_all_targets)


def do_train(tag, Model, params, X_train, y_train, kfold, nepochs, batch_size, model_dir, device,
             early_stop, patience, it, learning_rate=2e-2, weight_decay=1e-5, scheduler_pattern='plateau',
             transfer_helper=None, n_all_targets=0, log_prefix=''):
    n_features = X_train.shape[1]
    best_val_losses = []
    for n, (tr, te) in enumerate(kfold.split(y_train, y_train)):
        early_step = 0
        print(f'Train fold {n + 1}')
        xtrain, xval = X_train[tr], X_train[te]
        ytrain, yval = y_train[tr], y_train[te]

        train_set = MoaDataset(xtrain, ytrain)
        val_set = MoaDataset(xval, yval)

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
        }

        checkpoint_path = os.path.join(model_dir, "{}_{}_it_{}.pth".format(tag, n + 1, it))
        model = Model(n_features, **params, n_targets=n_all_targets).to(device)
        if transfer_helper is not None:
            # use transfer helper to load the first four layers of the pretrained model
            model.load_state_dict(torch.load(checkpoint_path))
            model = transfer_helper.copy_without_top(device, model, Model, n_features, n_all_targets, params,
                                                     ytrain.shape[1])
        print(model)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if scheduler_pattern == 'plateau':
            print("use plateau scheduler")
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, eps=1e-4,
                                                             verbose=True)
        else:
            print("use oneCycleLR scheduler")
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, max_lr=1e-2,
                                                      epochs=nepochs, steps_per_epoch=len(dataloaders['train']))
        best_loss = {'train': np.inf, 'val': np.inf}
        best_epoch = 0
        for epoch in range(nepochs):
            epoch_loss = {'train': 0.0, 'val': 0.0}

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0

                for i, (x, y) in enumerate(dataloaders[phase]):
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        preds = model(x)

                        if phase == 'train':
                            loss = criterion_train(preds, y)
                            loss.backward()
                            optimizer.step()
                        else:
                            loss = criterion_val(preds, y)

                    running_loss += loss.item() / len(dataloaders[phase])

                epoch_loss[phase] = running_loss

            print("{} - {} - fold {} - Epoch {}/{} - loss: {:5.5f} - val_loss: {:5.5f}".format(
                log_prefix, tag, n + 1, epoch + 1, nepochs, epoch_loss['train'], epoch_loss['val']))

            scheduler.step(epoch_loss['val'])

            if epoch_loss['val'] < best_loss['val']:
                best_loss = epoch_loss
                best_epoch = epoch + 1
                torch.save(model.state_dict(), checkpoint_path)
                early_step = 0

            elif early_stop:
                early_step += 1
                if early_step > patience:
                    print("Early stopping occured at epoch {}, model saved for the best val loss {} at epoch {}".format(
                        epoch + 1, best_loss['val'], best_epoch))
                    best_val_losses.append(best_loss['val'])
                    break
    return best_val_losses


def test_model(tag, Model, params, X_test, ntargets, nfolds, model_dir, device, batch_size, it):
    print("predict for submission")
    seed_preds = np.zeros((len(X_test), ntargets, nfolds))

    for n in range(nfolds):
        fold_preds = []
        test_set = MoaDataset(X_test, None, mode='test')

        dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

        checkpoint_path = os.path.join(model_dir, "{}_{}_it_{}.pth".format(tag, n + 1, it))
        model = Model(X_test.shape[1], **params).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)

            with torch.no_grad():
                batch_preds = torch.sigmoid(model(x))
                fold_preds.append(batch_preds)

        fold_preds = torch.cat(fold_preds, dim=0).cpu().numpy()
        seed_preds[:, :, n] = fold_preds

    return np.mean(seed_preds, axis=2)
