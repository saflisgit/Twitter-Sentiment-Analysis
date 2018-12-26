import pandas as pd

df = pd.read_csv('train1500k.csv', encoding='latin-1', header=None)


print(df)

ds = df.sample(frac=1)

print(ds)

ds.to_csv('train1500k_shuffled.csv', index=False, header=None)
