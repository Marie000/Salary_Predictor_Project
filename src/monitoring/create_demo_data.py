import pandas as pd
from predictor_model.processing.data_handling import load_dataset

batch_1 = load_dataset("tiny_1.csv")
batch_2 = load_dataset("tiny_2.csv")
batch_3 = load_dataset("tiny_3.csv")
batch_4 = load_dataset("tiny_4.csv")
batch_5 = load_dataset("tiny_5.csv")

dfs = [batch_1, batch_2, batch_3, batch_4, batch_5]
