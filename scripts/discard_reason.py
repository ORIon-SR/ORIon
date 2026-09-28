import pandas as pd

df = pd.read_csv("apod_data/apod_discarded.csv")
filt = pd.read_csv("apod_data/apod_metadata_filtered.csv")

# print(df['discard_reason'].tail(30))
print(df['discard_reason'].value_counts(dropna=False))

print(filt.info())