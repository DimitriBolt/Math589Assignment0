# ----------------------------------------------------------------
# File:     solve_quadratic_equation.py
# ----------------------------------------------------------------
#
# Author:   Dimitri Bolt (DimitriBolt@arizona.edu)
# Date:     Tue Jul 30 09:37:29 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
# ----------------------------------------------------------------
# A basic quadratic equation solver. High-school method.

import math
from decimal import Decimal


def solve_quadratic_equation(a, b, c):
    """
    Solve the quadratic equation a*x^2 + b*x + c = 0 using the standard quadratic formula.
    
    This function calculates the roots using the basic quadratic formula without any adjustments
    for numerical stability. It assumes real coefficients and only returns real roots.

    Parameters:
    a (float): Coefficient of x^2.
    b (float): Coefficient of x.
    c (float): Constant term.

    Returns:
    tuple:
    - (float): The first root.
    - (float or None): The second root, or None if there is only one distinct root due to a zero discriminant.

    Raises:
    """
    a = Decimal(a)
    b = Decimal(b)
    c = Decimal(c)

    # Check if the equation is actually quadratic
    if a == 0:
        raise InvalidEquationError("The equation is not quadratic (a = 0)") # Not a quadratic equation  (a = 0)

    # Calculate the discriminant
    discriminant: Decimal = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise InvalidEquationError("The equation has no real roots (discriminant < 0)")   # No real roots (discriminant < 0)  # No real roots (discriminant < 0)

    # Calculate the discriminant's square root
    sqrt_discriminant: Decimal = Decimal.sqrt(discriminant)

    # Compute both roots using the standard quadratic formula
    root1: float = float((-b + sqrt_discriminant) / (2 * a))
    root2: float = float((-b - sqrt_discriminant) / (2 * a)) if discriminant != 0 else None

    return root1, root2


class InvalidEquationError(Exception):
    def __init__(self, message):
        self.message = message


# Example usage:
# NOTE: Also, as simple testing framework.
if __name__ == "__main__":
    try:
        roots = solve_quadratic_equation(1, -10.001, 1)  # Using the earlier example coefficients
        print("Roots:", roots)
    except ValueError as e:
        print("Error:", e)
