import pandas as pd

class Cleaner():
    events_df = pd.DataFrame()
    series_df = pd.DataFrame()
    def __init__(self):
        self.events_df = pd.read_csv("data/original/train_events.csv")
        self.series_df = pd.read_parquet("data/original/train_series.parquet")
    def dropna(self):
        self.events_df = self.events_df.dropna()
        self.series_df = self.series_df.dropna()
        return self
    def drop_timestamp(self):
        self.events_df = self.events_df.drop(columns=['timestamp'])
        self.series_df = self.series_df.drop(columns=['timestamp'])
        return self
    def save(self, path_str):
        self.events_df.to_parquet(f"data/custom/event_{path_str}.parquet")
        self.series_df.to_parquet(f"data/custom/series_{path_str}.parquet")

if __name__=="__main__":
    cleaner = Cleaner()
    cleaner = cleaner.dropna().drop_timestamp()
    cleaner.save("dropna_timestamp")
    print(cleaner.events_df)
