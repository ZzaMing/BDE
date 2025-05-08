import pandas as pd

df = pd.read_csv("https://raw.githubusercontent.com/YoungjinBD/data/main/bike_data.csv")

# print(df.info())

df = df.astype(
    {
        "datetime": "datetime64[ns]",
        "weather": "int64",
        "season": "object",
        "workingday": "object",
        "holiday": "object",
    }
)

# df_sub = df.loc[df["season"] == 1]
# df_sub["hour"] = df_sub["datetime"].dt.hour
# summary_data = df_sub.groupby(["season", "hour"]).agg({"count": "sum"}).reset_index()
# max_count_hour = summary_data.loc[summary_data["count"].idxmax(), "hour"]
# print(max_count_hour)

# season_avg = df.groupby('season')['count'].mean()
# print(season_avg)

# groupmonth = df.groupby(df['datetime'].dt.month).agg({"count": 'sum'}).reset_index()
# print(groupmonth.loc[groupmonth['datetime'] == 1 , 'count'])
# result = df[df['datetime'].dt.month == 1]['count'].sum()
# print(result)

# df['date'] = df['datetime'].dt.date
# gro_df = df.groupby('date')['count'].sum().reset_index()
# print(gro_df['count'].max())

# df['hour'] = df['datetime'].dt.hour
# result = df.groupby('hour')['count'].mean().reset_index()
# print(result)

# df['weekday'] = df['datetime'].dt.weekday
# result = df[df["weekday"] == 0]['count'].sum()
# print(result)

# melted_df = df.melt(id_vars=['datetime', 'season'],
#                  value_vars=['casual', 'registered'],
#                  var_name='user_type',
#                  value_name='total_count')
# result = (
#     melted_df
#     .groupby(['season','user_type'])['total_count']
#     .mean()
#     .reset_index()
# )
# print(result)

df = pd.read_csv('https://raw.githubusercontent.com/YoungJinBD/data/main/logdata.csv')

# df['연도 정보'] = df['로그'].str.extract(r'(\d+)')
# print(df.head())

# df['시간 정보'] = df['로그'].str.extract(r'(\d{2}:\d{2}:\d{2})')
# print(df.head())

df['한글 정보'] = df['로그'].str.extract(r'([가-힣]+)')
print(df.head())