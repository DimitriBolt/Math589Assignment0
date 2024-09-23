from autodiff import autodiff, gradient, abs, max
from gradient_descent import find_global_minimum


@autodiff # First we go to autodiff.py and run the autodiff function
def h(x):
    return abs(x - 1)


y, dy = h(-3)
