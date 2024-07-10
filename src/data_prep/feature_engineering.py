import pandas as pd
from sklearn.cluster._kmeans import KMeans
from sklearn.decomposition import PCA
import numpy as np 

class FeatureTrain():
    def __init__(self, features_df, labels_df):
        self.features = features_df
        self.labels = labels_df
        self.X = pd.DataFrame()
    def merge(self):
        l = self.labels.drop(columns='night')
        l['event'] = self.labels['event'].apply(lambda x: 1 if x=="onset" else 2)
        l['step'] = l['step'].astype("int32")
        self.X = self.features.merge(l,how='left', on=["series_id","step"]).fillna(0)
        self.X['event'] = self.X['event'].astype("int32")
        return self

    def apply_kmeans(self):
        km = KMeans(n_clusters=2, n_init=5, max_iter=55)
        # print(self.X)
        self.X['cluster'] = km.fit(self.X.drop(columns=['series_id', 'step', 'event'])).labels_
        return self
    def apply_PCA(self):
        pca = PCA()
        # print(self.features)
        features_PCA = pca.fit_transform(self.X.drop(columns=['series_id', 'step', 'event']))
        features_PCA = pd.DataFrame(features_PCA, columns=[f"PC{i+1}" for i in range(features_PCA.shape[1])])
        self.X = self.X.merge(features_PCA, left_index=True, right_index=True)
        return self
    def apply_mult(self):
        self.X['mult'] = self.X['anglez'] * self.X['enmo']
        return self
    def apply_lag(self, lag_nb):
        for i in range(1, lag_nb+1):
            self.X[f'lag_{i}_anglez'] = self.X['anglez'].shift(i)
            self.X[f'lag_{i}_enmo'] = self.X['enmo'].shift(i)
        self.X = self.X.dropna()
        return self

    def save(self, path_str):
        self.X.to_parquet(f"data/custom/X_fe_{path_str}.parquet")
    def fit(self):
        self.merge()
        self.apply_PCA()
        self.apply_kmeans()
        self.apply_mult()
        self.apply_lag(2)
        return self

class FeatureEng():
    features = pd.DataFrame()
    labels = pd.DataFrame()
    def __init__(self, features_path, labels_path):
        self.features = features_path
        self.labels = labels_path
    def apply_kmeans(self):
        km = KMeans(n_clusters=2, n_init=5, max_iter=55)
        self.features['cluster'] = km.fit(self.features[["step", "anglez", "enmo"]]).labels_
        return self
    def apply_PCA(self):
        pca = PCA()
        # print(self.features)
        features_PCA = pca.fit_transform(self.features.drop(columns=['series_id', 'step']))
        features_PCA = pd.DataFrame(features_PCA, columns=[f"PC{i+1}" for i in range(features_PCA.shape[1])])
        self.features = self.features.merge(features_PCA, left_index=True, right_index=True)
        return self
    def apply_rolling_average(self):
        return self
    def apply_mult(self):
        self.features['mult'] = self.features['anglez'] * self.features['enmo']
        return self
    def apply_lag(self, lag_nb):
        for i in range(1, lag_nb+1):
            self.features[f'lag_{i}_anglez'] = self.features['anglez'].shift(i)
            self.features[f'lag_{i}_enmo'] = self.features['enmo'].shift(i)
        self.features = self.features.dropna()
        return self
    def apply_label_conversion(self):
        self.labels.loc[:, 'event'] = self.labels['event'].replace(['onset', 'wakeup'], [1,2])
        return self
    def apply_series_conversion(self):
        uniq = self.features['series_id'].unique()
        self.labels['_series_id'] = self.labels['series_id'].replace(uniq, [i for i in range(len(uniq))])
        self.features['_series_id'] = self.labels['series_id'].replace(uniq, [i for i in range(len(uniq))])
        return self
    def save(self, path_str):
        self.labels.to_parquet(f"data/custom/event_fe_{path_str}.parquet")
        self.features.to_parquet(f"data/custom/series_fe_{path_str}.parquet")
    def merging(self):

        self.labels = self.labels.drop(columns=['night']) # drop "unuseful" columns

        self.labels['step'] = self.labels['step'].astype(np.int64)
        self.features.insert(2, "event", 0)

        self.labels = self.labels.set_index(['series_id', 'step'])
        self.labels = self.features.set_index(['series_id', 'step'])

        for i in range(len(self.labels)):
            try:
                self.features[self.labels.iloc[i].name, 'event'] = self.labels.iloc[i]['event']
            except:
                pass
        return self
    def fit(self):
        return self.apply_label_conversion().merging()

if __name__=="__main__":
    fe = FeatureEng(pd.read_parquet("data/custom/series_dropna_timestamp.parquet"), pd.read_parquet("data/custom/event_dropna_timestamp.parquet"))
    # print(fe.apply_label_conversion().labels.head())
    fee = fe.fit()
    fee.save("simple_merged")

