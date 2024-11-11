import numpy as np
def synthetic_division(coeffs, root):
    #  Perform synthetic division on a polynomial with given coefficients by (x - root).
    # Parameters:
    # coeffs (list or numpy array): Coefficients of the polynomial in descending order.
    # root (float): The root to divide by (i.e., for divisor x - root).

    # Returns:
    # tuple: (quotient_coeffs, remainder)
    n = len(coeffs)
    # Copy coefficients to avoid modifying the original list
    b = np.zeros(n)
    b[0] = coeffs[0]
    for i in range(1, n):
        b[i] = coeffs[i] + root * b[i - 1]
    remainder = b[-1]
    quotient_coeffs = b[:-1]
    return quotient_coeffs, remainder

# Given coefficients of p(x)
coeffs = np.array([1, -3, 2, -5, 6], dtype=float)
# Dominant eigenvalue r1
r1 = 2.632962956285

# Perform synthetic division
quotient_coeffs, remainder = synthetic_division(coeffs, r1)

# Display the results
print("Quotient coefficients (p1(x)):")
for i, coeff in enumerate(quotient_coeffs):
    power = len(quotient_coeffs) - i - 1
    print(f"Coefficient of x^{power}: {coeff}")

