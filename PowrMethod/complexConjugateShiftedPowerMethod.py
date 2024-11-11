import numpy as np

# Define the companion matrix C
def companion_matrix(a):
    """
    Constructs the companion matrix for a monic polynomial with coefficients a.
    """
    n = len(a)
    C = np.zeros((n-1, n-1))
    C[0, :] = -np.array(a[1:])  # Convert to NumPy array to avoid TypeError
    if n > 2:
        C[1:, :-1] = np.eye(n-2)
    return C

# Given polynomial coefficients (monic polynomial)
# p(x) = x^3 - 0.367037043715 x^2 + 1.033722503934 x - 2.277444936001
coefficients = [1, -0.367037043715, 1.033722503934, -2.277444936001]

# Construct the companion matrix C
C = companion_matrix(coefficients)

# Initial vectors v0 and v1 (arbitrary but non-collinear)
v0 = np.array([1.0, 0.0, 0.0])
v1 = np.array([0.0, 1.0, 0.0])

# Orthonormalize initial vectors
v0 = v0 / np.linalg.norm(v0)
v1 = v1 - np.dot(v1, v0) * v0
v1 = v1 / np.linalg.norm(v1)

# Initialize variables
tolerance = 1e-12
max_iterations = 1000
iteration = 0

# Start the iteration process
while iteration < max_iterations:
    iteration += 1

    # Compute C*v0 and C*v1
    Cv0 = np.dot(C, v0)
    Cv1 = np.dot(C, v1)

    # Orthonormalize Cv0 and Cv1 using Gram-Schmidt
    w0 = Cv0
    w0 = w0 / np.linalg.norm(w0)
    w1 = Cv1 - np.dot(Cv1, w0) * w0
    w1 = w1 / np.linalg.norm(w1)

    # Form the 2x2 matrix B
    B = np.array([[np.dot(w0, np.dot(C, w0)), np.dot(w0, np.dot(C, w1))],
                  [np.dot(w1, np.dot(C, w0)), np.dot(w1, np.dot(C, w1))]])

    # Compute eigenvalues and eigenvectors of B
    eigenvalues, eigenvectors = np.linalg.eig(B)

    # Choose the eigenvalue with the largest magnitude
    idx = np.argmax(np.abs(eigenvalues))
    dominant_eigenvalue = eigenvalues[idx]
    dominant_eigenvector = eigenvectors[:, idx]

    # Form the approximate eigenvector in the original space
    v = dominant_eigenvector[0] * w0 + dominant_eigenvector[1] * w1

    # Compute the residual
    residual = np.linalg.norm(np.dot(C, v) - dominant_eigenvalue * v)

    # Check for convergence
    if residual < tolerance:
        break

    # Update vectors for next iteration
    v0 = w0
    v1 = w1

# Display the results
print(f"Converged after {iteration} iterations.")
print(f"Dominant eigenvalue: {dominant_eigenvalue}")
print(f"Residual: {residual}")

# For verification
roots = np.roots(coefficients)
print("Polynomial roots:", roots)
