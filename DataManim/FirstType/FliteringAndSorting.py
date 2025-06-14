import pandas as pd
import numpy as np
# pd.set_option('display.max_columns', None)


DataUrl = 'https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv'
df = pd.read_csv(DataUrl)
result = 0

# result = df.loc[df['quantity'] == 3].reset_index(drop=True)

# result = df[['quantity', 'item_price']]

df['new_price'] = df['item_price'].str[1:].astype('float')

# result = len(df.loc[df['new_price'] <= 5])

# result = df.loc[df['item_name'] == 'Chicken Salad Bowl'].reset_index(drop=True)

# result = df.loc[(df['new_price'] <= 9) & (df['item_name'] == 'Chicken Salad Bowl')]

# result = df.sort_values('new_price').reset_index(drop=True)

# result = df.loc[df['item_name'].str.contains('Chips')]

# result = df.iloc[:, ::2]

# result = df.sort_values('new_price', ascending=False).reset_index(drop=True)

# temp = df.loc[(df['item_name'] == 'Steak Salad') |( df['item_name']== 'Bowl')]

# result = temp.drop_duplicates('item_name', keep='last')

# result = df.loc[df['new_price'] >= df['new_price'].mean()]

# df.loc[df['item_name'] == 'Izze', 'item_name'] = 'Fizzy Lizzy'
# result = df['item_name'].head()

# result = df['choice_description'].isna().sum()

df.loc[df['choice_description'].isna(), 'choice_description'] = 'NoData'
# result = df['choice_description'].head(10)

# result = df.loc[df['choice_description'].str.contains("Black")]

# result = len(df.loc[~df['choice_description'].str.contains('Vegetables')])
# result = df[~df['choice_description'].str.contains('Vegetables')].shape[0]

# result = df[df['item_name'].str[0] == 'N']
# result = df[df['item_name'].str.startswith('N')]

# result = df.loc[df['item_name'].str.len() >= 15]

# lst = [1.69, 2.39, 3.39, 4.45, 9.25, 10.98, 11.75, 16.98]
# result = len(df.loc[df['new_price'].isin(lst)])

print(result)