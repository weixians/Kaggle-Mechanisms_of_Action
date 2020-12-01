import os
import sys

# only needed in kaggle. You can install them by pip in your local machine
sys.path.append('../input/iterative-stratification/iterative-stratification-master')
sys.path.append('../input/pytorch-tabnet')
sys.path.append('../input/moatorchscripts')
import pickle
import shutil
from moa_util import write_result, write_val_result, seed_everything, ensemble_result, calculate_overall_loss

import torch
import warnings
from moa_preprocess import MoaPreprocessor
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from moa_tabnet_helper import train_tabnet, test_tabnet
import moa_model_helper
from moa_train_test import train_model, test_model
from moa_train_helper import TransferHelper, SmoothBCEwLogits

warnings.filterwarnings('ignore')
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

# 参数设置
base_seed = 42

# 数据预处理参数
root_dir = "../input/lish-moa"
out_data_dir = "out_data"
out_tabnet_data_dir = "out_tabnet_data"
drop_ctl = True

data_preprocess_param_shallow_mlp = dict(
    tag='shallow',
    base_seed=base_seed,
    scale='rankgauss',
    variance_threshold=0.8,
    use_pca=True,
    use_svd=True,
    ncompo_genes=600,
    ncompo_cells=50,
    encoding='dummy',
    square_feature=True,
    new_feature=False,
    new_feature2=False,
    square2=False,
    cluster_pca=False, ncompo_cluster_pca=5,
    name='data_shallow.pkl',
    out_data_dir=out_data_dir
)

# 模型参数
model_dir = "models"
result_dir = "result"
val_result_dir = "val_result"

# 训练参数
batch_size = 1024
nfolds = 10
nepochs = 150
learning_rate = 2e-2
weight_decay = 1e-5
scheduler_pattern = 'plateau'
patience = 10
early_stop = True
niter = 1

# 其他参数
test_only = False
read_directly = False
write_val = True

tabnet_params1 = dict(
    n_d=32, n_a=32, n_steps=1, gamma=1.3, seed=base_seed,
    lambda_sparse=0, optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    mask_type='entmax',
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9, ),
    # epoch打印间隔
    verbose=1
)
tabnet_params2 = dict(
    n_d=24, n_a=128, n_steps=1, gamma=1.3, seed=base_seed,
    lambda_sparse=0, optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    mask_type='entmax',
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9, ),
    # epoch打印间隔
    verbose=1
)

tabnet_fit_params = dict(
    eval_name=["val"],
    eval_metric=["logits_ll"],
    max_epochs=nepochs,
    patience=15, batch_size=batch_size, virtual_batch_size=32,
    num_workers=0, drop_last=False,
    loss_fn=SmoothBCEwLogits(smoothing=5e-5)
)
# 不带tabnet2: 0.011367210640428164
# 不带tabnet2+model608: 0.011348517795743793
# 不带tabnet2+model608+model1024720: 0.01132097227785723
model_choices = [
    {'tag': 'tabnet1', 'data_type': 'tabnet', 'params': tabnet_params1, 'weight': 1},
    # {'tag': 'tabnet2', 'data_type': 'tabnet', 'params': tabnet_params2, 'weight': 1},
    # {'tag': 'model15121075', 'class': moa_model_helper.ModelMlp,
    #  'params': {'hidden_sizes': [1500, 1250, 1000, 750], 'dropout_rates': [0.5, 0.35, 0.3, 0.25]}, 'train': True,
    #  'transfer': True, 'data_type': 'transfer', 'weight': 1},
    # {'tag': 'model151075', 'class': moa_model_helper.ModelMlp,
    #  'params': {'hidden_sizes': [1500, 1250, 750], 'dropout_rates': [0.5, 0.35, 0.25]}, 'train': True, 'transfer': True,
    #  'data_type': 'transfer', 'weight': 1},
    # {'tag': 'model1500', 'class': moa_model_helper.ModelMlp,
    #  'params': {'hidden_sizes': [1500, 1500], 'dropout_rates': 0.2619422201258426},
    #  'train': True, 'transfer': True, 'data_type': 'transfer', 'weight': 0.8},
    # {'tag': 'model1024', 'class': moa_model_helper.ModelMlp,
    #  'params': {'hidden_sizes': [1024, 1024], 'dropout_rates': 0.18}, 'train': True,
    #  'transfer': True, 'data_type': 'transfer', 'weight': 2},
    # {'tag': 'model129672', 'class': moa_model_helper.ModelMlp,
    #  'params': {'hidden_sizes': [1280, 960, 720], 'dropout_rates': [0.4, 0.3, 0.18]}, 'train': True,
    #  'transfer': True, 'data_type': 'transfer', 'weight': 1},
    # {'tag': 'model1280', 'class': moa_model_helper.ModelMlp,
    #  'params': {'hidden_sizes': [1280, 1280], 'dropout_rates': 0.21}, 'train': True,
    #  'transfer': True, 'data_type': 'transfer', 'weight': 1},
    # {'tag': 'model720', 'class': moa_model_helper.ModelMlp,
    #  'params': {'hidden_sizes': [720, 720], 'dropout_rates': 0.1}, 'train': True,
    #  'transfer': True, 'data_type': 'transfer', 'weight': 0.5},
]

# 根据对应类型的模型是否存在，决定是否创建该类型的数据集
create_data_tabnet = False
create_data_shallow = False
create_data_transfer = False
for choice in model_choices:
    if choice['data_type'] == 'tabnet':
        create_data_tabnet = True
    elif choice['data_type'] == 'shallow':
        create_data_shallow = True
    elif choice['data_type'] == 'transfer':
        create_data_transfer = True

kfold = MultilabelStratifiedKFold(n_splits=nfolds, random_state=0, shuffle=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def create_data(data_preprocess_param):
    if read_directly:
        with open(os.path.join(out_data_dir, data_preprocess_param['name']), 'rb') as f:
            print(f"read data from {os.path.join(out_data_dir, data_preprocess_param['name'])} directly")
            X_train, Y_train, non_scored_targets, X_test, features, target_columns = pickle.load(f)
    else:
        preprocessor = MoaPreprocessor(root_dir=root_dir, **data_preprocess_param,
                                       test_only=test_only)
        X_train, Y_train, non_scored_targets, X_test, features, target_columns = preprocessor.get_data()
    return X_train, Y_train, non_scored_targets, X_test, features, target_columns


def get_data(data_type, data_tabnet, data_transfer, data_shallow):
    if data_type == 'tabnet':
        return data_tabnet
    elif data_type == 'transfer':
        return data_transfer
    elif data_type == 'shallow':
        return data_shallow
    return None


pytorch_preprocess_param = {
    'is_drop_ctl_vehicle': True,

    'is_add_square_feature': True,

    'is_delete_feature': False,

    'is_gauss_rank': True,

    'is_pca': True,
    'is_svd': True,
    'n_gene_comp': 600,
    'n_cell_comp': 50,
    'is_add_cluster_gene_cell': True,
    'n_clusters_g': 22, 'n_clusters_c': 4,
    'is_add_cluster_pca': True,
    'n_comp_cluster_pca': 5,

    'is_filtered_by_var': True,
    'variance_thresh': 0.8,

    'is_encoding': True,
    'encoding': 'dummy',

    'is_add_statistic_feature': False,
}


def work(test_only, write_val=False):
    data_tabnet, data_transfer, data_shallow = None, None, None
    if create_data_tabnet:
        print('Create data for type tabnet')
        from moa_tabnet_data_preprocess_helper import TabnetPreprocessHelper
        helper = TabnetPreprocessHelper('../input', out_tabnet_data_dir, not test_only, read_directly)
        data_tabnet = helper.process()

    if create_data_shallow:
        print('Create data for type shallow')
        data_shallow = create_data(data_preprocess_param_shallow_mlp)
    if create_data_transfer:
        print('Create data for type transfer')
        # data_transfer = create_data(data_preprocess_param_transfer)
        from moa_torch_data_preprocess_helper import PytorchPreprocessHelper
        helper = PytorchPreprocessHelper('../input', out_data_dir, not test_only, read_directly)
        data_transfer = helper.process(pytorch_preprocess_param, base_seed)

    best_val_losses_dict = {}
    best_val_losses = []
    val_preds = []
    train_sample_size = 0
    for choice in model_choices:
        X_train, Y_train, non_scored_targets, X_test, features, target_columns = get_data(choice['data_type'],
                                                                                          data_tabnet, data_transfer,
                                                                                          data_shallow)
        ntargets = Y_train.shape[1]
        train_sample_size = X_train.shape[0]
        X_train = X_train.values
        Y_train = Y_train.values
        Y_non_scored = non_scored_targets.values
        X_test = X_test.values

        print(f"X_train shape: {X_train.shape}")
        print(f"Y_train shape: {Y_train.shape}")
        print(f"X_test shape: {X_test.shape}")

        # 每种模型重复训练多次
        for it in range(1, niter + 1):
            print(f"Processing model: {choice['tag']}")
            if choice['tag'].startswith('tabnet'):
                choice['params']['seed'] = 19 + it
                if not test_only:
                    best_val_losses = train_tabnet(choice['tag'], choice['params'], tabnet_fit_params, X_train, Y_train,
                                                   kfold, model_dir, it)
                preds = test_tabnet(choice['tag'], choice['params'], X_test, ntargets, nfolds, model_dir, it)
                if write_val:
                    # 重新预测train数据集，用于检验val loss
                    val_preds = test_tabnet(choice['tag'], choice['params'], X_train, ntargets, nfolds, model_dir, it)
            # elif choice['tag'] == 'lightgbm':
            #     preds = train_lightgbm(choice['tag'], X_train, y_train, X_test, ntargets, kfold, nfolds, model_dir, it)
            else:
                if choice['transfer']:
                    transfer_helper = TransferHelper(nepochs)
                else:
                    transfer_helper = None
                if not test_only and choice['train']:
                    best_val_losses = train_model(choice['tag'], choice['class'], choice['params'], X_train, Y_train,
                                                  kfold, nepochs, batch_size, model_dir, device, early_stop, patience,
                                                  it, learning_rate, weight_decay, scheduler_pattern, transfer_helper,
                                                  Y_non_scored)
                preds = test_model(choice['tag'], choice['class'], choice['params'],
                                   X_test, ntargets, nfolds, model_dir, device, batch_size, it)
                if write_val:
                    val_preds = test_model(choice['tag'], choice['class'],
                                           choice['params'],
                                           X_train, ntargets, nfolds, model_dir, device,
                                           batch_size, it)
            write_result(root_dir, preds, target_columns,
                         os.path.join(result_dir,
                                      '{}_{}_submission#{}#.csv'.format(choice['tag'], it, choice['weight'])))
            if write_val:
                write_val_result(root_dir, drop_ctl, val_preds, target_columns,
                                 os.path.join(val_result_dir,
                                              '{}_{}_submission#{}#.csv'.format(choice['tag'], it, choice['weight'])))

        if not test_only:
            best_val_losses_dict[choice['tag']] = best_val_losses
    print(best_val_losses_dict)
    return train_sample_size


if __name__ == '__main__':
    seed_everything(base_seed)

    for directory in [result_dir, val_result_dir]:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    for directory in [model_dir, result_dir, val_result_dir, out_data_dir, out_tabnet_data_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # copy models to work directory
    input_model_dir = "../input/moamodels/models"
    if os.path.exists(input_model_dir):
        shutil.rmtree(model_dir)
        shutil.copytree(input_model_dir, model_dir)
    # copy pca,svd,variance...pkl
    input_data_dir = "../input/moamodels/out_data"
    if os.path.exists(input_data_dir):
        shutil.rmtree(out_data_dir)
        shutil.copytree(input_data_dir, out_data_dir)
    # tabnet数据
    input_tabnet_data_dir = "../input/moamodels/out_tabnet_data"
    if os.path.exists(input_tabnet_data_dir):
        shutil.rmtree(out_tabnet_data_dir)
        shutil.copytree(input_tabnet_data_dir, out_tabnet_data_dir)

    # 训练模型并分别预测出结果
    train_sample_size = work(test_only, write_val)
    # 融合并写出最终结果
    ensemble_result(root_dir, drop_ctl, result_dir)

    if write_val:
        # 评估val loss
        ensemble_result(root_dir, drop_ctl, val_result_dir, 'val_predicted.csv', True, train_sample_size)
        overall_val_loss = calculate_overall_loss(root_dir, drop_ctl)
        print("overall val loss: {}".format(overall_val_loss))
