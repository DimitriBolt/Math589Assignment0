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

x, y = find_global_minimum(f, x0, learning_rate, tol)

print(f'Global minimum: {x}, Function value: {y}')

#Check if x and y are good
y_ref = 5.0625


# assert(y <= y_ref + tol)


@gradient
def g(x, y):
    return max(5 * x ** 2 + y ** 2 + 3 * x + 5 * y, 3 * (x - 1) ** 2 + 3 * (y - 2) ** 2 + 3 * x * y)


# test g(1, 2)
# var_list = [1, 2]
# var_tuple = (1, 2)
# var_g = g(1, 2)
# var_g_list = g(var_list)
# var_g_tuple = g(var_tuple) # !! Not logical: x0 is tuple, but g is called with 2 arguments

y, grad = g(1, 2)

x0 = (0, 0)
x, y = find_global_minimum(g, x0, learning_rate, tol)

print(f'Global minimum: {x}, Function value: {y}')

# Check if x and y are good
y_ref = 6.317166615825466
assert (y <= y_ref + tol)

pass
