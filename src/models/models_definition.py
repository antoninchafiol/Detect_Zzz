import pandas as pd 

from sklearn.model_selection import train_test_split
# from ..data_prep.feature_engineering import *



def generate_split_sets(path):
    X = pd.read_parquet(path)
    t = {'series_id': [], "min_index": [], "max_index": []}
    for i in X['series_id'].unique():
        t['series_id'].append(i)
        t['min_index'].append(X[X['series_id'] == i].index.min())
        t['max_index'].append(X[X['series_id'] == i].index.max())
    t= pd.DataFrame(t)
    t.to_parquet("data/custom/train_test_splitting_set.parquet")


def splitting_rows(source_split_set, sample_percent, full_set):
    def generate_new_training(split_set):
        df = pd.DataFrame(columns=full_set.columns)
        for i in range(split_set.shape[0]):
            df = pd.concat([df, full_set.loc[split_set.iloc[i]['min_index']:split_set.iloc[i]['max_index']]])
        return df
    
    X_train = generate_new_training(source_split_set.drop(source_split_set.sample(frac=sample_percent, random_state=42).index))
    X_test = generate_new_training(source_split_set.sample(frac=sample_percent, random_state=42))
    y_train = X_train.pop('event')
    y_test = X_test.pop('event')
    return X_train, X_test, y_train, y_test

def ram_splitting_rows(source_split_set_path, sample_percent, full_set_path, ram_eco_frac):
    X = pd.read_parquet(full_set_path)
    split_set = pd.read_parquet(source_split_set_path)
    split_set = split_set[:int(split_set.shape[0]*ram_eco_frac)]
    X = X[:split_set.loc[split_set.shape[0]-1]['max_index']]
    return splitting_rows(split_set, sample_percent, X)

if __name__=='__main__':

    xt, xd, yt, yd = ram_splitting_rows("data/custom/train_test_splitting_set.parquet", .35, "data/custom/X_fe_full0.parquet", 0.35)
    print(yt)
    print(yd)