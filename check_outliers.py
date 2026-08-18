import pandas as pd

df = pd.read_parquet('data/modeling/dataset.parquet')

print('Top 5 extreme ret_fwd:')
print(df.nlargest(5, 'ret_fwd')[['date','ticker','ret_fwd','ret_lag1','ret_lag2']])

print('\nBottom 5 extreme ret_fwd:')
print(df.nsmallest(5, 'ret_fwd')[['date','ticker','ret_fwd','ret_lag1','ret_lag2']])

print('\nTop 5 extreme vix_change (after clip):')
print(df.nlargest(5, 'vix_change')[['date','ticker','india_vix','vix_change']])

print('\nBottom 5 extreme vix_change (after clip):')
print(df.nsmallest(5, 'vix_change')[['date','ticker','india_vix','vix_change']])

print('\nTop 5 extreme crude_oil:')
print(df.nlargest(5, 'crude_oil')[['date','ticker','crude_oil','oil_change']])

print('\nTop 5 extreme oil_change (after clip):')
print(df.nlargest(5, 'oil_change')[['date','ticker','crude_oil','oil_change']])

print('\nBottom 5 extreme oil_change (after clip):')
print(df.nsmallest(5, 'oil_change')[['date','ticker','crude_oil','oil_change']])

print('\nTop 5 extreme usd_inr:')
print(df.nlargest(5, 'usd_inr')[['date','ticker','usd_inr','usdinr_change']])

print('\nTop 5 extreme usdinr_change (after clip):')
print(df.nlargest(5, 'usdinr_change')[['date','ticker','usd_inr','usdinr_change']])

print('\nBottom 5 extreme usdinr_change (after clip):')
print(df.nsmallest(5, 'usdinr_change')[['date','ticker','usd_inr','usdinr_change']])

# Check all feature value ranges
print('\nFeature ranges:')
features = ['ret_fwd','ret_lag1','ret_lag2','india_vix','crude_oil','usd_inr',
            'vix_change','oil_change','usdinr_change','smart_score',
            'S_recency','S_events','S_breadth','S_volume']
print(df[features].describe().round(3))