import numpy as np
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/under5MortalityRate.csv')
result = 0

df = df.drop('Indicator', axis=1)
df['First Tooltip'] = df['First Tooltip'].map(lambda x: float(x.split('[')[0]))

# tmp = df.loc[(df['Period'] >= 2015) & (df['Dim1'] == 'Both sexes')]
# result = tmp.pivot(index = 'Location', columns='Period',values = 'First Tooltip')

# result = df.pivot_table(index='Dim1', columns='Period', values = 'First Tooltip', aggfunc='mean')

df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/winter.csv')

# kr = df.loc[df['Country'] == 'KOR']
# result = kr.pivot_table(index='Year', columns='Medal', aggfunc='size').fillna(0)

# result = df.pivot_table(index='Sport', columns='Gender', aggfunc='size')

result = df.pivot_table(index='Discipline', columns='Medal', aggfunc='size')

print(result)