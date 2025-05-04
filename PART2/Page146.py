import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/YoungjinBD/data/main/grade.csv')

# print(df.info())

# filtered_df = df.loc[df['midterm'] >= 85]
# print(filtered_df)

# sorted_df = df.sort_values(by='final', ascending=False)
# print(sorted_df.head())

# grouped_df = df.groupby(by='gender')[['midterm','final']].mean()
# print(grouped_df)

# df['student_id'] = df['student_id'].astype('str')
# print(df.info())

# max_idx = df['assignment'].idxmax()
# min_idx = df['assignment'].idxmin()
# print(df.loc[max_idx])
# print(df.loc[min_idx])

# df['average'] = df[['midterm', 'final', 'assignment']].mean(axis=1)
# print(df.head())

# print(df.isna().sum())
# cleaned_df = df.dropna()
# print(cleaned_df)
# print(cleaned_df.info())

# addtional_data = {
#     'student_id' : ['1', '3', '5', '7', '9'],
#     'club' : ['Art', 'Science', 'Math', 'Music', 'Drama']
# }
# addtional_df = pd.DataFrame(addtional_data)
# df['student_id'] = df['student_id'].astype('str')
# merged_df = pd.merge(df, addtional_df, on='student_id', how='left')
# print(merged_df)

# df['average'] = df[['midterm', 'final', 'assignment']].mean(axis=1)
# pivot_table = df.pivot_table(values='average', index='gender',columns='student_id')
# print(pivot_table)

# df['average'] = df[['midterm', 'final', 'assignment']].mean(axis=1)
# metled_df = pd.melt(df, id_vars=['student_id', 'name', 'gender'],
#                     value_vars=['midterm','final','assignment', 'average'],
#                     var_name='variable', value_name='score')
# grouped_mean = metled_df.groupby(['gender', 'variable'])['score'].mean().reset_index()
# print(grouped_mean)

# df['average'] = df[['midterm', 'final','assignment']].mean(axis=1)
# # print(df.loc[df['average'].idxmax()][['name','average']])
# print(df.loc[df['average'].idxmax(),['name','average']])