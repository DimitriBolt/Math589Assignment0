from typing import Any

import numpy as np

from autodiff import autodiff, gradient, abs, max
from autodiff import Variable

from gradient_descent import find_global_minimum


@autodiff
def h(x: int | Variable) -> Variable:
    return abs(x - 1)


y, dy = h(-3)


# Test maximum function


@gradient
def f(x: int | Variable):
    return max(5*x**2+3*x, 2*x**2-5*x + 7, (x-1)**2+5)


y, grad = f(1)
