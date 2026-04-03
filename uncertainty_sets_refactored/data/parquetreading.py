import pandas as pd
df = pd.read_parquet('forecasts_filtered_rts3_constellation_v2.parquet')
print(df.columns.tolist())
print(df.dtypes)
print(df.shape)
print(df.head(2))