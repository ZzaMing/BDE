import numpy as np

# 1 / 배열 생성
a = np.array([[3, 5, 7], 
              [2, 3, 6]])

print(a)

# 2 / 배열의 2, 4, 5 번째 행만 선택
B = np.array([[8, 10, 7, 8],
              [2, 4, 5, 5],
              [7, 6, 1, 7],
              [2, 6, 8, 6], 
              [9, 3, 4, 2]])

B_seleted = B[[1, 3, 4], :]
print(B_seleted)

# 3 / 2번의 행렬 B에서 3번째 열의 값이 3보다 큰 행
B_filtered = B[B[:,2] > 3, :]
print(B_filtered)

# 4 / 행렬 B에서 행별 합이 20보다 크거나 같은 행만 선택하여 새로운 행렬을 작성
row_sums = np.sum(B, axis=1)
print(row_sums)
B_row_sums_filtered = B[row_sums > 20, :]
print(B_row_sums_filtered)

# 5 / 행렬 B에서 열별 평균이 5보다 큰 열이 몇 번째 열에 위치하는가
col_means = np.mean(B, axis=0)
print(col_means)
col_indices = np.where(col_means >5)[0]
print(col_indices)

# 6 / 행렬 B의 각 행에 7보다 큰 숫자가 하나라도 들어있는 행을 찾아 출력
print(B > 7)
print(B[np.sum(B>7, axis=1) > 0 , :])
