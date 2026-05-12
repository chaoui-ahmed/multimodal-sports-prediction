import pandas as pd

df = pd.read_csv("data/processed/train_stats.csv")
df['date'] = pd.to_datetime(df['date'])

df[df['date'].dt.year < 2025].to_csv("data/processed/train_stats.csv", index=False)
df[df['date'].dt.year >= 2025].to_csv("data/processed/test_stats.csv", index=False)