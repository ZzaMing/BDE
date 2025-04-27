import numpy as np

# 1
a = np.array([1, 2, 3, 4, 5])
a = a + 5

print(a)

# 2
a = np.array([12, 21, 35, 48, 5])
a = a[::2]
print(a)

# 3
a = np.array([1, 22, 93, 64, 54])
a = np.max(a)
print(a)

# 4
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
a = np.unique(a)
print(a)

# 5
a = np.array([21, 31, 58])
b = np.array([24, 48, 67])
c = np.empty(a.size + b.size, dtype=a.dtype)
c[0::2] = a
c[1::2] = b
print(c)

# 6
a = np.array([1, 2, 3, 4, 5])
b = np.array([6, 7, 8, 9])
c = a[:-1] + b
print(c)
