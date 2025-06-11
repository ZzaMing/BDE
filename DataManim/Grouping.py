import numpy as np
import pandas as pd
df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/AB_NYC_2019.csv')
result = 0

# result = df.groupby('host_name').size()
# result = df['host_name'].value_counts().sort_index()

# result = (df.groupby('host_name').size()
#           .to_frame()
#           .rename(columns={0: 'counts'})
#           .sort_values('counts',ascending=False))

# result = df.groupby('neighbourhood_group')['price'].agg(['mean','var','max', 'min'])

# result = df.groupby(['neighbourhood', 'neighbourhood_group'])['price'].mean()

# result = df.groupby(['neighbourhood', 'neighbourhood_group'])['price'].mean().unstack().fillna(-999)

# result = (df.loc[df['neighbourhood_group'] == 'Queens']
#           .groupby('neighbourhood')
#           ['price']
#           .agg(['mean', 'var','max','min']))

print(result)