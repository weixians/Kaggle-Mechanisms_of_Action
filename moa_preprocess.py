import os
import sys

sys.path.append('../input/rank-gauss')

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from scipy import stats
from gauss_rank_scaler import GaussRankScaler
from sklearn.cluster import KMeans
import pickle


class MoaPreprocessor:
    def __init__(self, root_dir, tag, variance_threshold=0.8, scale='rankgauss', use_pca=True, use_svd=True,
                 ncompo_genes=80, ncompo_cells=10, base_seed=2020, encoding='lb', n_clusters_g=22, n_clusters_c=4,
                 new_feature=False, cluster_feature=False, drop_ctl=True, square_feature=True, new_feature2=True,
                 square2=True, cluster_pca=False, ncompo_cluster_pca=5, name='data.pkl', test_only=False,
                 out_data_dir=None):
        self.tag = tag
        self.name = name
        self.test_only = test_only
        self.out_data_dir = out_data_dir

        if not os.path.exists(os.path.join(out_data_dir, tag)):
            os.makedirs(os.path.join(out_data_dir, tag))

        X_train, y_train, X_test, non_scored_targets = self.read_data(root_dir)
        X_train, y_train, non_scored_targets = self.drop_ctl_vehicle_samples(X_train, y_train, non_scored_targets,
                                                                             drop_ctl)
        data_all = pd.concat([X_train, X_test], ignore_index=True)

        genes = [col for col in data_all.columns if col.startswith('g-')]
        cells = [col for col in data_all.columns if col.startswith('c-')]
        # 加入新生成的特征
        data_all = self.generate_new_features(data_all, genes, cells, new_feature, new_feature2, square_feature,
                                              square2)
        # 归一化数据
        data_all = self.normalize_ori_data(data_all, scale, genes, cells)
        # 加入降维后的特征
        data_all = self.decomposition(data_all, genes, cells, ncompo_genes, ncompo_cells, base_seed, use_pca, use_svd,
                                      cluster_pca, ncompo_cluster_pca)
        # 加入聚类后的特征
        data_all = self.add_cluster_features(data_all, genes, cells, n_clusters_g, n_clusters_c, base_seed,
                                             cluster_feature)
        # 删除无用的列
        data_all = self.delete_useless_columns(data_all, drop_ctl)
        # 筛选数据，删除方差过小的特征
        data_all = self.filter_feature_by_variance(data_all, variance_threshold)
        # 编码特征，放到最后再做
        data_all = self.encode_feature(data_all, encoding, drop_ctl)

        self.X_train = data_all[:X_train.shape[0]]
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test = data_all[X_train.shape[0]:]
        self.X_test.reset_index(drop=True, inplace=True)
        self.y_train = y_train
        self.non_scored_targets = non_scored_targets

    def read_data(self, root_dir):
        X_train = pd.read_csv(os.path.join(root_dir, 'train_features.csv'))
        y_train = pd.read_csv(os.path.join(root_dir, 'train_targets_scored.csv'))
        non_scored_targets = pd.read_csv(os.path.join(root_dir, 'train_targets_nonscored.csv'))
        X_test = pd.read_csv(os.path.join(root_dir, 'test_features.csv'))
        # 删掉不要的列
        del y_train['sig_id']
        del non_scored_targets['sig_id']

        return X_train, y_train, X_test, non_scored_targets

    def get_data(self):
        feature_columns = [col for col in self.X_train.columns]
        target_columns = [col for col in self.y_train.columns]

        # 保存数据到本地，方便以后直接读
        with open(os.path.join(self.out_data_dir, self.name), 'wb') as f:
            data_store = (self.X_train, self.y_train, self.non_scored_targets, self.X_test, feature_columns,
                          target_columns)
            pickle.dump(data_store, f)

        return self.X_train, self.y_train, self.non_scored_targets, self.X_test, feature_columns, target_columns

    def drop_ctl_vehicle_samples(self, X_train, y_train, non_scored_targets, drop_ctl):
        if not drop_ctl:
            return X_train, y_train
        X_train = X_train[X_train['cp_type'] != 'ctl_vehicle']
        y_train = y_train.iloc[X_train.index]
        non_scored_targets = non_scored_targets.iloc[X_train.index]

        X_train.reset_index(drop=True, inplace=False)
        y_train.reset_index(drop=True, inplace=False)
        non_scored_targets.reset_index(drop=True, inplace=False)
        # Note: 不删除test数据集中cp_type=0的行，直接在预测后将其值设为0即可，防止index不对
        return X_train, y_train, non_scored_targets

    def filter_feature_by_variance(self, data_all, variance_threshold):
        if variance_threshold == 0:
            return data_all
        print("filter features by variance")
        # 筛掉方差小于 variance_threshould 的特征
        cols_numeric = [feat for feat in list(data_all.columns) if
                        feat not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]
        cols_non_numeric = [feat for feat in list(data_all.columns) if
                            feat in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]

        if not self.test_only:
            var_thresh = VarianceThreshold(variance_threshold).fit(data_all[cols_numeric])
            pickle.dump(var_thresh, open(os.path.join(self.out_data_dir, self.tag, "variance.pkl"), 'wb'))
        else:
            var_thresh = pickle.load(open(os.path.join(self.out_data_dir, self.tag, "variance.pkl"), 'rb'))

        data_transformed = var_thresh.transform(data_all[cols_numeric])
        # data_transformed = data_all[data_all.columns[var_thresh.get_support(indices=True)]]
        data_all = pd.concat([data_all[cols_non_numeric], pd.DataFrame(data_transformed)], axis=1)
        print(f"data shape after variance filter: {data_all.shape}")
        return data_all

    def normalize_ori_data(self, data_all, scale, genes, cells):
        cols_numeric = [feat for feat in list(data_all.columns) if
                        feat not in ['sig_id', 'cp_type', 'cp_time', 'cp_dose']]

        def scale_minmax(col):
            return (col - col.min()) / (col.max() - col.min())

        def scale_norm(col):
            return (col - col.mean()) / col.std()

        if scale == 'boxcox':
            # 通过 BoxCox 正态化
            print('boxcox')
            data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis=0)
            trans = []
            for feat in cols_numeric:
                trans_var, lambda_var = stats.boxcox(data_all[feat].dropna() + 1)
                trans.append(scale_minmax(trans_var))
            data_all[cols_numeric] = np.asarray(trans).T

        elif scale == 'norm':
            # 通过标准化正态化
            print('norm')
            data_all[cols_numeric] = data_all[cols_numeric].apply(scale_norm, axis=0)

        elif scale == 'minmax':
            # 归一化
            print('minmax')
            data_all[cols_numeric] = data_all[cols_numeric].apply(scale_minmax, axis=0)

        elif scale == 'rankgauss':
            print('rankgauss')
            for col in cols_numeric:
                transformer = QuantileTransformer(n_quantiles=100, random_state=0, output_distribution="normal")
                vec_len = len(data_all[col].values)
                raw_vec = data_all[col].values.reshape(vec_len, 1)
                transformer.fit(raw_vec)
                data_all[col] = transformer.transform(raw_vec)

        elif scale == 'standard_rankgauss':
            # 标准高斯化
            print('standard rankgauss')
            scaler = GaussRankScaler()
            data_all[cols_numeric] = scaler.fit_transform(data_all[cols_numeric])
        print(f"data shape after normalization: {data_all.shape}")

        return data_all

    def decomposition(self, data_all, genes, cells, ncompo_genes, ncompo_cells, base_seed, use_pca, use_svd,
                      cluster_pca=False, ncompo_cluster_pca=5):

        if use_pca:
            print('PCA')
            if self.test_only:
                g_pca = pickle.load(open(os.path.join(self.out_data_dir, self.tag, 'g_pca.pkl'), 'rb'))
                c_pca = pickle.load(open(os.path.join(self.out_data_dir, self.tag, 'c_pca.pkl'), 'rb'))
            else:
                g_pca = PCA(n_components=ncompo_genes, random_state=base_seed).fit(data_all[genes])
                c_pca = PCA(n_components=ncompo_cells, random_state=base_seed).fit(data_all[cells])
                pickle.dump(g_pca, open(os.path.join(self.out_data_dir, self.tag, 'g_pca.pkl'), 'wb'))
                pickle.dump(c_pca, open(os.path.join(self.out_data_dir, self.tag, 'c_pca.pkl'), 'wb'))

            pca_genes = g_pca.transform(data_all[genes])
            pca_cells = c_pca.transform(data_all[cells])
            pca_genes = pd.DataFrame(pca_genes, columns=[f'pca_g-{i}' for i in range(ncompo_genes)])
            pca_cells = pd.DataFrame(pca_cells, columns=[f'pca_c-{i}' for i in range(ncompo_cells)])
            data_all = pd.concat([data_all, pca_genes, pca_cells], axis=1)
            print(f"data shape after PCA: {data_all.shape}")

            if cluster_pca:
                print("Cluster PCA")
                data_pca = pd.concat((pca_genes, pca_cells), axis=1)
                if self.test_only:
                    kmeans_pca = pickle.load(open(os.path.join(self.out_data_dir, self.tag, 'kmeans_pca.pkl'), 'rb'))
                else:
                    kmeans_pca = KMeans(n_clusters=ncompo_cluster_pca, random_state=base_seed).fit(data_pca)
                    pickle.dump(kmeans_pca, open(os.path.join(self.out_data_dir, self.tag, 'kmeans_pca.pkl'), 'wb'))
                data_all[f'cluster_pca'] = kmeans_pca.predict(data_pca.values)
                data_all = pd.get_dummies(data_all, columns=[f'cluster_pca'])
                print(f"data shape after cluster PCA: {data_all.shape}")

        if use_svd:
            print('SVD')
            if self.test_only:
                g_svd = pickle.load(open(os.path.join(self.out_data_dir, self.tag, 'g_svd.pkl'), 'rb'))
                c_svd = pickle.load(open(os.path.join(self.out_data_dir, self.tag, 'c_svd.pkl'), 'rb'))
            else:
                g_svd = TruncatedSVD(n_components=ncompo_genes, random_state=base_seed).fit(data_all[genes])
                c_svd = TruncatedSVD(n_components=ncompo_cells, random_state=base_seed).fit(data_all[cells])
                pickle.dump(g_svd, open(os.path.join(self.out_data_dir, self.tag, 'g_svd.pkl'), 'wb'))
                pickle.dump(c_svd, open(os.path.join(self.out_data_dir, self.tag, 'c_svd.pkl'), 'wb'))

            svd_genes = g_svd.transform(data_all[genes])
            svd_cells = c_svd.transform(data_all[cells])
            svd_genes = pd.DataFrame(svd_genes, columns=[f'svd_g-{i}' for i in range(ncompo_genes)])
            svd_cells = pd.DataFrame(svd_cells, columns=[f'svd_c-{i}' for i in range(ncompo_cells)])
            data_all = pd.concat([data_all, svd_genes, svd_cells], axis=1)
            print(f"data shape after SVD: {data_all.shape}")

        return data_all

    def encode_feature(self, data_all, encoding, drop_ctl):
        if drop_ctl:
            columns = ['cp_time', 'cp_dose']
        else:
            columns = ['cp_time', 'cp_dose', 'cp_type']

        # Encoding
        if encoding == 'lb':
            print('Label Encoding')
            for feat in columns:
                data_all[feat] = LabelEncoder().fit_transform(data_all[feat])
        elif encoding == 'dummy':
            print('One-hot')
            data_all = pd.get_dummies(data_all, columns=columns)

        print(f"data shape after encoding: {data_all.shape}")
        return data_all

    def generate_new_features(self, data_all, genes, cells, new_feature=True, new_feature2=True, square_feature=True,
                              square2=True):
        if new_feature:
            print("generate new feature")
            # 特征生成
            for stats in ['sum', 'mean', 'std', 'kurt', 'skew']:
                data_all['g_' + stats] = getattr(data_all[genes], stats)(axis=1)
                data_all['c_' + stats] = getattr(data_all[cells], stats)(axis=1)
                data_all['gc_' + stats] = getattr(data_all[genes + cells], stats)(axis=1)
            print(f"data shape after new feature: {data_all.shape}")

        if new_feature2:
            data_all['c52_c42'] = data_all['c-52'] * data_all['c-42']
            data_all['c13_c73'] = data_all['c-13'] * data_all['c-73']
            data_all['c26_c13'] = data_all['c-23'] * data_all['c-13']
            data_all['c33_c6'] = data_all['c-33'] * data_all['c-6']
            data_all['c11_c55'] = data_all['c-11'] * data_all['c-55']
            data_all['c38_c63'] = data_all['c-38'] * data_all['c-63']
            data_all['c38_c94'] = data_all['c-38'] * data_all['c-94']
            data_all['c13_c94'] = data_all['c-13'] * data_all['c-94']
            data_all['c4_c52'] = data_all['c-4'] * data_all['c-52']
            data_all['c4_c42'] = data_all['c-4'] * data_all['c-42']
            data_all['c13_c38'] = data_all['c-13'] * data_all['c-38']
            data_all['c55_c2'] = data_all['c-55'] * data_all['c-2']
            data_all['c55_c4'] = data_all['c-55'] * data_all['c-4']
            data_all['c4_c13'] = data_all['c-4'] * data_all['c-13']
            data_all['c82_c42'] = data_all['c-82'] * data_all['c-42']
            data_all['c66_c42'] = data_all['c-66'] * data_all['c-42']
            data_all['c6_c38'] = data_all['c-6'] * data_all['c-38']
            data_all['c2_c13'] = data_all['c-2'] * data_all['c-13']
            data_all['c62_c42'] = data_all['c-62'] * data_all['c-42']
            data_all['c90_c55'] = data_all['c-90'] * data_all['c-55']
            print(f"data shape after new feature2: {data_all.shape}")

        if square_feature:
            print("add square feature")
            # 这四个特征是平方后，并使用lightgbm训练后，显示最重要的平方特征
            features_cols = ['g-7', 'g-91', 'g-100', 'g-130', 'g-175', 'g-300', 'g-608', 'c-98']
            for col in features_cols:
                data_all[col + "_square"] = data_all[col].apply(lambda a: a ** 2)
            print(f"data shape after square own features: {data_all.shape}")

        if square2:
            gsquarecols = ['g-574', 'g-211', 'g-216', 'g-0', 'g-255', 'g-577', 'g-153', 'g-389', 'g-60', 'g-370',
                           'g-248', 'g-167', 'g-203', 'g-177', 'g-301', 'g-332', 'g-517', 'g-6', 'g-744', 'g-224',
                           'g-162', 'g-3', 'g-736', 'g-486', 'g-283', 'g-22', 'g-359', 'g-361', 'g-440', 'g-335',
                           'g-106', 'g-307', 'g-745', 'g-146', 'g-416', 'g-298', 'g-666', 'g-91', 'g-17', 'g-549',
                           'g-145', 'g-157', 'g-768', 'g-568', 'g-396']
            for feature in gsquarecols:
                data_all[f'{feature}_squared'] = data_all[feature] ** 2
            # for feature in cells:
            #     data_all[f'{feature}_squared'] = data_all[feature] ** 2

            print(f"data shape after square2: {data_all.shape}")

        return data_all

    def add_cluster_features(self, data_all, genes, cells, n_clusters_g=22, n_clusters_c=4, SEED=123,
                             cluster_feature=True):
        if cluster_feature:
            print("generate cluster feature")

            def create_cluster(data_all, features, kind='g', n_clusters=n_clusters_g):
                data = data_all[features].copy()
                if self.test_only:
                    kmeans = pickle.load(open(os.path.join(self.out_data_dir, self.tag, f'kmeans_{kind}.pkl'), 'rb'))
                else:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(data)
                    pickle.dump(kmeans, open(os.path.join(self.out_data_dir, self.tag, f'kmeans_{kind}.pkl'), 'wb'))

                data_all[f'clusters_{kind}'] = kmeans.predict(data.values)
                data_all = pd.get_dummies(data_all, columns=[f'clusters_{kind}'])
                return data_all

            data_all = create_cluster(data_all, genes, kind='g', n_clusters=n_clusters_g)
            data_all = create_cluster(data_all, cells, kind='c', n_clusters=n_clusters_c)
            print(f"data shape after cluster feature: {data_all.shape}")

        return data_all

    def delete_useless_columns(self, data_all, drop_ctl):
        if drop_ctl:
            features_todrop = ['sig_id', 'cp_type']
        else:
            features_todrop = ['sig_id']
        data_all.drop(features_todrop, axis=1, inplace=True)
        print(f"data shape after drop useless columns: {data_all.shape}")

        return data_all
