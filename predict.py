import pickle

import pandas as pd

from train import ReModel

test_df = pd.read_csv("data/addresses-close.csv", sep=",", encoding="windows-1251")
re_model = pickle.load(open("re_model.pkl", mode="rb"))
test_df = re_model.predict(test_df)
test_df.to_csv("subm.csv", index=False, encoding="windows-1251")
