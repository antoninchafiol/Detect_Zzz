import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score



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
    # X.to_parquet("data/custom/ram_x.parquet")
    return splitting_rows(split_set, sample_percent, X)


class SlModel():
    def __init__(self, model, ft=None, ft_params=None) -> None:
        self.predictions = None
        self.model = model
        if ft!=None:
            self.model = ft(**ft_params)
    def train(self, X, y):
        self.model.fit(X, y)
    def predict(self, X, y):
        self.y = y
        self.predictions = self.model.predict(X, y)
        return self.predictions
    def metrics_show(self, returning_value=False):
        metric = None
        try: 
            metric = {
            "accuracy": accuracy_score(self.y, self.predictions),
            "precision":precision_score(self.y, self.predictions),
            "recall":f1_score(self.y, self.predictions),
            "f1":recall_score(self.y, self.predictions),
        }
        except AttributeError:
            print("try using the predict function beforehand")
        if returning_value:
            return metric
        for k, v in metric.items():
            print(f"{k} - {v}")



from sklearn.linear_model import LogisticRegression
if __name__=='__main__':
    model = SlModel(LogisticRegression(multi_class="ovr"))
    # xt, xd, yt, yd = ram_splitting_rows("data/custom/train_test_splitting_set.parquet", .35, "data/custom/X_fe_full0.parquet", 0.35)
    ram_x = pd.read_parquet("data/custom/ram_x.parquet")
    xt, xd, yt, yd = splitting_rows(pd.read_parquet("data/custom/train_test_splitting_set.parquet"), .35, ram_x)
    # print(yt)
    # print(yt.ravel())
    model.train(xt,yt)
    print(model.predict(xd))
