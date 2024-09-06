import numpy as np

sum_decreasing_order: np.single = np.single(0)
N: int = 1000000
for i in range(N, 1, -1):  # Decreasing order but! increasing magnitude of summands
    sum_decreasing_order += 1 / i

sum_increasing_order: np.single = np.single(0)
for i in range(1, N, +1):  # Increasing order but! Decreasing magnitude of summands
    sum_increasing_order += 1 / i

print(sum_decreasing_order, sum_increasing_order)
