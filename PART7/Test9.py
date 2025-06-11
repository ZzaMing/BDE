import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None) # 모든 칼럼이 출력되게 조절
# 1 작업형
# 1. 각 연도 및 성별의 총 대출액의 절댓값 차이를 구하고, 절댓값 차이가 가장 큰 지역코드를 구하시요.
df = pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_1_1.csv')

df['총대출액'] = df['금액1'] + df['금액2']
dfgender = df.groupby(['year', 'gender', '지역코드'])['총대출액'].sum().reset_index()

df_pivot = dfgender.pivot_table(index=['year', '지역코드'], columns='gender', values='총대출액', fill_value=0)

df_pivot['abs_diff'] = abs(df_pivot[0] - df_pivot[1])
df_pivot_idxmax = df_pivot['abs_diff'].idxmax()
# print(df_pivot.loc[[df_pivot_idxmax]])

#---------------------------------------------------------------------------------------

# 2. 각 연도별 최대 검거율을 가진 범죄유형을 찾아서 해당 연도 및 유형의 검거건수들의 총합을 구하시오.
df= pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_1_2.csv')

df1 = df[df['구분'] == '검거건수'].set_index('연도').drop(columns='구분')
df2 = df[df['구분'] == '발생건수'].set_index('연도').drop(columns='구분')
dfratio = df1 / df2

dfmax = dfratio.idxmax(axis=1).reset_index().rename({0: '범죄유형'}, axis =1 )
df_melt = df1.reset_index().melt(id_vars='연도', var_name='범죄유형', value_name='검거건수')
dffinal = pd.merge(dfmax, df_melt, on= ['연도', '범죄유형'], how='left')

result = dffinal['검거건수'].sum()

# print(result)

#-----------------------------------------------------------------------------

# 3. 제시된 문제를 순서대로 풀고, 해답을 제시하시오.
# 결측치 처리
# - 평균만족도: 결측치는 평균만족도 컬럼의 전체 평균으로 채우시오.
# - 근속연수: 결측치는 각 부서와 등급별 평균 근속연수로 채우시오 (평균값의 소수점은 버림 처리)
# 조건에 따른 평균 계산
# - A: 부서가 'HR' 이고 등급이 'A'인 사람들의 평균 근속연수를 계산하시오.
# - B: 부서가 'Sales'이고 등급이 'B'인 사람들의 평균 교육참가횟수를 계산하시오.
# - A와 B를 더한 값을 구하시오.

df= pd.read_csv('https://raw.githubusercontent.com//YoungjinBD/data/main/exam/9_1_3.csv')
print(df)
df['평균만족도'] = df['평균만족도'].fillna(df['평균만족도'].mean())

mena_tenure = (
    df.groupby(['부서','등급'])['근속연수']
    .mean()
    .apply(np.floor)
    .reset_index()
    .rename(columns={'근속연수': '평균근속연수'})
)
print(df)
df = df.merge(mena_tenure, on=['부서','등급'], how='left')
print(df)
# df['근속연수'] 

# print(df.isna().sum())
#===========================================================
# 2 작업형
#==============================================================
# 3 작업형