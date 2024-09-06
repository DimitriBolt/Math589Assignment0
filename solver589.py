from math import sin, cos, asin

# Python program for implementation of Bisection Method for solving equations
# An example function whose # solution is determined using Bisection Method.
func = lambda x: sin(x) - (1 - x)


# Prints root of func(x) with error of EPSILON
def bisection_method(function, a: float, b: float, tolerance: float):

    try:
        function(a)
    except TypeError:
        raise InvalidBracket("expects a bracket, but receives an invalid bracket")

    c = a
    n: int = 0
    while (b - a) > tolerance:
        n: int = n + 1
        # Find middle point
        c = (a + b) / 2
        # Check if middle point is root
        if function(c) == 0.0:
            break

        # Decide the side to repeat the steps
        if function(c) * function(a) < 0:
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
f = lambda x: sin(x) - (1 - x)


def secant_method(function, x0, x1, tolerance, max_iteration):
    step = 1
    condition = True
    while condition:
        # if function(x0) == function(x1):
        #     print('Divide by zero error!')
        #     break

        x2 = x0 - (x1 - x0) * function(x0) / (function(x1) - function(x0))
        x0 = x1
        x1 = x2
        step = step + 1

        if step > max_iteration:
            print('Not Convergent!')
            break

        condition = abs(function(x2)) >= tolerance/8
    return x2


# Implementing False Position Method
def regula_falsi(function, x0, x1, tolerance):

    try:
        function(x0)
    except TypeError:
        raise InvalidBracket("expects a bracket, but receives an invalid bracket")

    step = 1
    condition = True
    while condition:
        x2 = x0 - (x1 - x0) * function(x0) / (function(x1) - function(x0))

        if function(x0) * function(x2) < 0:
            x1 = x2
        else:
            x0 = x2

        step = step + 1
        condition = abs(function(x2)) > tolerance

    return x2


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

    # initializing the values
    x0g = 0.58
    x2g = 0.8
    tol = 1e-6
    print(f"Secant: {secant_method(f, x0g, x2g, tol, max_iteration=90)}")

    x0 = 0.58
    x1 = 0.8
    tol = 1e-6
    print(f"Regula Falsi: {regula_falsi(f, x0, x1, tol)}")

# This code is contributed by Dimitri Bolt.
