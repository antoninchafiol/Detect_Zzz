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




if __name__=='__main__':
    generate_split_sets("data/custom/X_fe_full0.parquet") 

    xt, xd, yt, yd = splitting_rows(pd.read_parquet("data/custom/train_test_splitting_set.parquet"), .35, pd.read_parquet("data/custom/X_fe_full0.parquet"))
    print(yt)
    print(yd)