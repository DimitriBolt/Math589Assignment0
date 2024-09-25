#----------------------------------------------------------------
# File:     autodiff.py
#----------------------------------------------------------------
#
# Author:   Marek Rychlik (rychlik@arizona.edu)
# Date:     Sat Sep  7 08:11:18 2024
# Copying:  (C) Marek Rychlik, 2020. All rights reserved.
# 
#----------------------------------------------------------------
# A simple autodifferentiation package
import numpy as np
from multimethod import multimethod


class Variable:
    def __init__(self, value, derivative=1.0):
        self.value: float = value
        self.derivative: float = derivative

    def __add__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value + other.value, self.derivative + other.derivative)
        else:
            return Variable(self.value + other, self.derivative)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value - other.value, self.derivative - other.derivative)
        else:
            return Variable(self.value - other, self.derivative)

    def __rsub__(self, other):
        return Variable(other - self.value, -self.derivative)

    def __mul__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value * other.value, self.value * other.derivative + self.derivative * other.value)
        else:
            return Variable(self.value * other, self.derivative * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Variable):
            return Variable(self.value / other.value,
                            (self.derivative * other.value - self.value * other.derivative) / (other.value ** 2))
        else:
            return Variable(self.value / other, self.derivative / other)

    def __rtruediv__(self, other):
        return Variable(other / self.value, -other * self.derivative / (self.value ** 2))

    def __pow__(self, power):
        return Variable(self.value ** power, power * self.value ** (power - 1) * self.derivative)

    def sin(self):
        return Variable(np.sin(self.value), np.cos(self.value) * self.derivative)

    def cos(self):
        return Variable(np.cos(self.value), -np.sin(self.value) * self.derivative)

    def exp(self):
        return Variable(np.exp(self.value), np.exp(self.value) * self.derivative)

    def sqrt(self):
        return Variable(np.sqrt(self.value), self.derivative / (2 * np.sqrt(self.value)))

    def to_pair(self):
        return self.value, self.derivative

    def abs(self):
        """Absolute value operation for Variable."""
        return Variable(np.abs(self.value), np.sign(self.value) * self.derivative)

    def max(self, other):
        pass
        """Maximum value operation for Variable."""
        var: Variable = 1 / 2 * ((self + other) + abs(self - other))
        return var


@multimethod
def sqrt(x: float | int):
    return np.sqrt(x)


@multimethod
def sqrt(x: Variable):
    return x.sqrt()


@multimethod
def abs(x: Variable):
    return x.abs()


@multimethod
def max(*args: Variable):
    pass
    arg_list: list = list(args)
    if isinstance(arg_list[0], float):
        arg_list[0] = Variable(args[0], derivative=0.0)  # If the first argument is a float, convert it to a Variable
    pass
    if len(arg_list) == 1:
        return arg_list[0]
    else:
        var = max(*arg_list[1:])  # Recursive call
        max_ = arg_list[0].max(var)  # analog ordinary max-function between 2 Variables
    return max_


class Constant(Variable):
    def __init__(self, value):
        super().__init__(value, 0.0)


def autodiff(f: callable) -> callable:
    def g(x: float) -> tuple[float]:
        xx = Variable(x)
        yy = f(xx)
        return yy.to_pair()

    return g


def gateaux(f: callable) -> callable:
    def wrapper(*args, **kwargs):
        direction = kwargs.get('direction')
        del kwargs['direction']
        vargs = [Variable(arg, incr) for arg, incr in zip(args, direction)]
        val = f(*vargs, **kwargs)
        pass
        return val.to_pair()

    return wrapper


def standard_basis(n):
    for i in range(n):
        var = tuple(1 if j == i else 0 for j in range(n))
        yield tuple(1 if j == i else 0 for j in range(n))


def gradient(f: callable) -> callable:
    g: callable = gateaux(f)

    def wrapper(*args, **kwargs):
        n = len(args)
        partials = [g(*args, **kwargs, direction=v) for v in standard_basis(n)]
        val = partials[0][0]
        return val, tuple([p[1] for p in partials])

    return wrapper


if __name__ == '__main__':
    # Example usage:
    x = Variable(2.0)
    y = x ** 2 + 3 * x + 2

    print(f"Value of expression: {y.value}")
    print(f"Derivative of expression: {y.derivative}")


    # Function of 1 variable that "knows how to differentiate itself"
    @autodiff
    def f(x):
        y = x ** 2 + 3 * x + 2
        return y


    y, dy = f(2.0)
    print(f"Value of function: {y}")
    print(f"Derivative of function: {dy}")


    # Function of 2 variables that "knows how to find its directional derivative"
    @gateaux
    def g(x, y):
        z = x ** 2 + 3 * x * y
        return z


    z, dz = g(1, 2, direction=(1, 2))
    print(f"Value: {z}, Gateaux derivative in direction {(1, 2)}: {dz}")


    # Function of 2 variables that "knows how to find its gradient"
    @gradient
    def g(x, y):
        z = x ** 2 + 3 * x * y
        return z


    val, grad = g(1, 2)
    print(f"Value: {val}, Gradient: {grad}")

    # We need to add abs function for the Variable class
