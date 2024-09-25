import numpy as np
from autodiff import autodiff, gradient, abs, max
from autodiff import Variable
from gradient_descent import find_global_minimum


@autodiff
def h(x: int | Variable) -> Variable:
    return abs(x - 1)


y, dy = h(-3)


@gradient
def f(x: int | Variable):
    return max(5 * x ** 2 + 3 * x, 2 * x ** 2 - 5 * x + 7, (x - 1) ** 2 + 5)
    # return max(1.0, 2 * x ** 2 - 5 * x + 7, (x - 1) ** 2 + 5)



y, grad = f(1)

x0 = 0
learning_rate = 0.1
tol = 1e-10
# x, y = find_global_minimum(f, x0, learning_rate, tol)
# print(f'Global minimum: {x}, Function value: {y}')
