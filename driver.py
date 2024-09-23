from autodiff import autodiff, gradient, abs, max
from gradient_descent import find_global_minimum

@autodiff
def h(x):
    return abs(x-1)

y, dy = h(-3)

@gradient
def f(x):
    return max(5*x**2+3*x, 2*x**2-5*x + 7, (x-1)**2+5)

y, grad = f(1)


x0 = 0
learning_rate = 0.1
tol = 1e-10

x, y = find_global_minimum(f, x0, learning_rate, tol)

print(f'Global minimum: {x}, Function value: {y}')

#Check if x and y are good
y_ref = 5.0625
assert(y <= y_ref + tol)


@gradient
def g(x, y):
    return max(5*x**2+y**2 + 3*x + 5*y, 3*(x-1)**2 + 3*(y-2)**2 + 3*x*y)

y, grad = g(1,2)


x0 = (0,0)
x, y = find_global_minimum(g, x0, learning_rate, tol)

print(f'Global minimum: {x}, Function value: {y}')

# Check if x and y are good
y_ref = 6.317166615825466
assert(y <= y_ref + tol)