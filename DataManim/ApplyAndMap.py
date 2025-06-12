import numpy as np
import pandas as pd

df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/BankChurnersUp.csv',index_col=0)
print(df.shape)

result = 0


dic = {
    'Unknown'        : 'N',
    'Less than $40K' : 'a',
    '$40K - $60K'    : 'b',
    '$60K - $80K'    : 'c',
    '$80K - $120K'   : 'd',
    '$120K +'        : 'e' 
}
df['new_income'] = df['Income_Category'].map(dic)
# df['new_income'] = df['Income_Category'].apply(dic.get)
# result = df['new_income']

df['age_state'] = df['Customer_Age']//10 * 10
# result = df.groupby('age_state').size()

# df['new_edu_level'] = df['Education_Level'].str.contains('Graduate').astype(int)
# df['new_edu_level'] = df['Education_Level'].map(lambda x: 1 if 'Graduate' in x else 0)
# result = df.groupby('new_edu_level').size()

#------------------------60 --------------------------------

df['new_limit'] = (df['Credit_Limit'] >= 4500).astype(int)
# result = df['new_limit'].value_counts()
# result = df.groupby('new_limit').size()

df['new_state'] = ((df['Marital_Status'] == 'Married') & (df['Card_Category'] == 'Platinum')).astype(int)
# result = df['new_state'].value_counts()

df['Gender'] = df['Gender'].map(lambda x: 'male' if x == 'M' else 'female')
result = df.groupby('Gender').size()

print(result)