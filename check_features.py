# check_features.py
import pandas as pd

df = pd.read_parquet('data/modeling/dataset.parquet')

features = [
    'smart_score','S_recency','S_events','S_breadth','S_volume',
    'total','pos','neg',
    'ret_lag1','ret_lag2',
    'fii_net','dii_net',
    'vix_change','oil_change','usdinr_change'
]

print("Feature ranges:")
print(df[features].describe().round(3))

print("\nFII/DII specific:")
print(f"fii_net: {df['fii_net'].min():.2f} to {df['fii_net'].max():.2f}")
print(f"dii_net: {df['dii_net'].min():.2f} to {df['dii_net'].max():.2f}")
print(f"fii_net unique values: {df['fii_net'].nunique()}")
print(f"fii_net zeros: {(df['fii_net'] == 0).sum()}")

print("\nRows where fii_net is non-zero:")
print(df[df['fii_net'] != 0][['date','ticker','fii_net','dii_net']].head(10))

print("\nExtreme FII/DII values:")
print(df.nlargest(5, 'fii_net')[['date','ticker','fii_net','dii_net']])
print(df.nsmallest(5, 'fii_net')[['date','ticker','fii_net','dii_net']])