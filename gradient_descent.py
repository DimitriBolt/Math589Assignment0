def find_global_minimum(f, x0, learning_rate=0.1, tol=1e-6):
    """
    Find global minimum of a function f which admits the following syntax:
       y, grad = f(x)
    where y is the value of the function at x and grad is its gradient at x.
    The grad is a tuple of the same length as x.

    Parameters:
    f (callable):          A strictly convex function
    x0 (float|tuple|list): Initial approximation
    learning_rate (float): The learning rate. In applications, this parameter needs to be tuned.
                           In particular, to handle the functions in this problem, you will need to
                           gradually decrease η to meet the desired accuracy.
    tol (float):           Stopping condition constant. Stopping criterion should be:
                           |f(x_{n+1}) - f(x_n)| < tol or gradient norm of f(x_n) < tol or both.

    Returns:
    - (float|tuple): An approximate minimum
    - (float): The value of f at the minimum

    Raises:
    - Exception: If maximum iterations are exceeded.
    - TypeError: If x0 is not a float, int, list, or tuple.
    """
    import math

    # Convert x0 to a list if it's a tuple
    if isinstance(x0, (float, int)):
        x = x0
    elif isinstance(x0, (list, tuple)):
        x = list(x0)
    else:
        raise TypeError("x0 must be a float, int, list, or tuple")

    # Initialize y and grad
    try:
        y, grad = f(x)
    except Exception as e:
        raise Exception(f"Error evaluating function f at initial point x0: {e}")

    max_iter = 1000000  # To avoid infinite loops
    iter_count = 0

    while iter_count < max_iter:
        # Update x
        if isinstance(x, (float, int)):
            x_new = x - learning_rate * grad[0]  # grad is a tuple of one element
        else:
            x_new = [xi - learning_rate * gi for xi, gi in zip(x, grad)]

        # Compute new y and grad
        try:
            y_new, grad_new = f(x_new)
        except Exception as e:
            raise Exception(f"Error evaluating function f at x = {x_new}: {e}")

        # Compute change in y
        y_diff = abs(y_new - y)

        # Compute gradient norm
        grad_norm = math.sqrt(sum(g**2 for g in grad_new))

        # Check convergence
        if y_diff < tol:  # or grad_norm < tol:
            x = x_new
            y = y_new
            break  # Converged
        else:
            x = x_new
            y = y_new
            grad = grad_new
            iter_count += 1

            # Optionally decrease the learning rate
            learning_rate = learning_rate / (1 + 0.01 * iter_count)

    else:
        raise Exception("Maximum iterations exceeded")

    # Return the final x and y
    return x, y
