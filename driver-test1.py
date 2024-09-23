from autodiff import autodiff, gradient, abs, max
from gradient_descent import find_global_minimum

@autodiff
def h(x):
    return abs(x-1)

y, dy = h(-3)

@gradient
def f(x):
    var = 5*x**2+3*x, 2*x**2-5*x + 7, (x-1)**2+5
    return max(var)

y, grad = f(1)
