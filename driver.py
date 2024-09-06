# Solve the equation sin(x) = 1-x

import solver589 as solver
from math import sin, cos, asin

tol = 1e-6
max_iter = 90

# A function
f = lambda x: sin(x) - (1 - x)
a = 0
b = 1.57
assert (f(a) * f(b) < 0)

x = solver.bisection_method(f, a, b, tol)
print(f"Bisection Method: {x}")

x0 = 0.58

g = lambda x: 1 - sin(x)
x = solver.fixed_point(g, x0, tol, max_iter)
print(f"Fixed point: {x}")

# A function plus derivative (a "jet" of a function)
df = lambda x: (sin(x) - (1 - x), cos(x) + 1)

x = solver.newton_raphson(df, x0, tol, max_iter)
print(f"Newton-Raphson: {x}")

x1 = 0.8
x = solver.secant_method(f, x0, x1, tol, max_iter)
print(f"Secant: {x}")

x = solver.regula_falsi(f, a, b, tol)
print(f"Regula Falsi: {x}")

## THE EXPECTED OUTPUT
# Bisection Method: 0.5109736299514771
# Fixed point: 0.5109738364898875
# Newton-Raphson: 0.5109734293885692
# Secant: 0.5109734293872639
# Regula Falsi: 0.510973429388569
