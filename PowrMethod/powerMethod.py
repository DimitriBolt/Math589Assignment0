import numpy as np
# Define the matrix C
C = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [-6, 5, -2, 3]
], dtype=float)
# Initial vector x^(0)
x = np.array([1, 1, 1, 1], dtype=float)
# Maximum number of iterations
max_iter = 60
# Convergence tolerance
tolerance = 1e-12
# Initialize variables
for k in range(max_iter):
    # Multiply C by the current vector
    y = np.dot(C, x)
    # Normalize the resulting vector
    x_new = y / np.linalg.norm(y)
    # Estimate the eigenvalue
    lambda_new = np.dot(x_new.T, np.dot(C, x_new))
    # Compute the residual for convergence check
    residual = np.linalg.norm(np.dot(C, x_new) - lambda_new * x_new)
    # Check convergence criterion
    if residual < tolerance:
        break
    # Update the vector for the next iteration
    x = x_new
print(f'Eigenvalue: {lambda_new}')
