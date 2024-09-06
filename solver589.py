from math import sin, cos, asin

# Python program for implementation of Bisection Method for solving equations
# An example function whose # solution is determined using Bisection Method.
func = lambda x: sin(x) - (1 - x)


# Prints root of func(x) with error of EPSILON
def bisection_method(function, a: float, b: float, tolerance: float):

    if function(a)*function(b) >= 0 :
        raise InvalidBracket("expects a bracket, but receives an invalid bracket")

    c = a
    n: int = 0
    while (b - a) >= tolerance:
        n: int = n + 1
        # Find middle point
        c = (a + b) / 2
        fc = function(c)
        # Check if middle point is root
        if fc == 0.0:
            break

        # Decide the side to repeat the steps
        if fc * function(a) < 0:
            b = c
        else:
            a = c

    return c


def fixed_point(given_function, x0, tolerance=0.001, max_iteration=3):  # -> Tuple[float, List]:
    i = 0
    error = 1
    # xp = []
    x = None
    while error > tolerance and i < max_iteration:
        x = given_function(x0)
        error = abs(x0 - x)
        x0 = x
        # xp.append(x0)
        i += 1
    # print(xp)
    return x  #, xp


# Python3 code for implementation of Newton Raphson Method for solving equations
# An example function whose solution is determined using Newton Raphson Method.
# The function is sin(x) - (1 - x)
df = lambda x: (sin(x) - (1 - x), cos(x) + 1)


# Function to find the root
def newton_raphson(f_df, x, tolerance: float, max_iteration):
    h = f_df(x)[0] / f_df(x)[1]
    while abs(h) >= tolerance:
        h = f_df(x)[0] / f_df(x)[1]

        # x(i+1) = x(i) - f(x) / f'(x)
        x = x - h
    return x


# Python3 Program to find root of an equations using secant method
# function takes value of x and returns f(x)



def secant_method(function, x0, x1, tolerance, max_iteration):
    step = 1
    condition = True
    while condition:
        x2 = x1 - (x1 - x0) * function(x1) / (function(x1) - function(x0))
        x0 = x1
        x1 = x2
        step = step + 1

        if step > max_iteration:
            print('Not Convergent!')
            break

        # condition = abs(function(x2)) >= tolerance
        condition = abs(x1-x0) > tolerance

    return x1


# Implementing False Position Method
def regula_falsi(f, a, b, tol):
    fa, fb = f(a), f(b)
    if fa * fb >= 0:
        raise InvalidBracket("Invalid bracket.")
    while (b - a) / 2 > tol:
        c = b - (fb * ((b - a) / (fb - fa)))
        fc = f(c)
        if fc < tol:
            return c #c is the root
        elif fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return (a + b) / 2


class ToleranceNotMet(Exception):
    def __init__(self, message):
        self.message = message


class InvalidBracket(Exception):
    def __init__(self, message):
        self.message = message


# Driver code
if __name__ == "__main__":
    # Initial values assumed
    a1 = 0
    b1 = 1.57
    bisection_method(func, a1, b1, tolerance=0.01)
    print(f"Bisection Method: {bisection_method(func, a1, b1, tolerance=1e-6)}")

    # Driver program to test above newton_raphson(x)
    x0 = 0.58  # Initial values assumed
    print(f"Newton-Raphson: {newton_raphson(df, x0, tolerance=1e-6, max_iteration=100)}")

    f = lambda x: sin(x) - (1 - x)

    # initializing the values
    x0g = 0.58
    x2g = 0.8
    tol = 1e-6
    print(f"Secant: {secant_method(f, x0g, x2g, tol, max_iteration=90)}")

    a = 0
    b = 1.57
    tol = 1e-6
    print(f"Regula Falsi: {regula_falsi(f, a, b, tol)}")

# This code is contributed by Dimitri Bolt.
