import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/bike_data.csv')

# print(df.info())

df = df.astype({'datetime': 'datetime64[ns]',
                'weather' : 'int64',
                'season': 'object',
                'workingday': 'object',
                'holiday': 'object'})

