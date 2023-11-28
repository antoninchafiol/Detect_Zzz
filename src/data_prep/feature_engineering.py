import pandas as pd
from sklearn.cluster._kmeans import KMeans
from sklearn.decomposition import PCA

# Adding simple kmeans

# Adding some PCA

# Getting some rolling average

# Mult anglez with enmo

# Adding some lag of both features

# Create a class that group it all

# 

class Feature_eng():
    features = pd.DataFrame()
    labels = pd.DataFrame()
    def __init__(self, features_path, labels_path):
        self.features = features_path
        self.labels = labels_path
    def apply_kmeans(self):
        km = KMeans(n_clusters=2, n_init=5, max_iter=55)
        self.features['cluster'] = km.fit(self.features[["step", "anglez", "enmo"]].labels)
        return self
    def apply_PCA(self):
        pca = PCA()
        features_PCA = pca.fit_transform(self.features)
        features_PCA = pd.DataFrame(features_PCA, columns=[f"PC{i+1}" for i in range(features_PCA.shape[1])])
        self.features = self.features.merge(features_PCA, left_index=True, right_index=True)
        return self
    def apply_rolling_average(self):
        return self
    def apply_lag(self, lag_nb):
        for i in range(lag_nb):
            self.features[f'lag_{i}_anglez'] = self.features['anglez'].shift(i)
            self.features[f'lag_{i}_enmo'] = self.features['enmo'].shift(i)
        return self
    def apply_label_conversion(self):
        self.labels['digit_event'] = self.labels['event'].apply(lambda x: 0 if x=="onset" else 1)
        return self
    

    if __name__=="__main__":
        fe = Feature_eng(pd.read_parquet("data/custom/series_dropna_timestamp.parquet"), pd.read_parquet("data/custom/event_dropna_timestamp.parquet"))