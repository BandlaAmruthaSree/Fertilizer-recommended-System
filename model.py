import pandas as pd

df = pd.read_csv("dataset/fertilizer.csv")

print(df.head())

print(df.info())

print(df.describe())
