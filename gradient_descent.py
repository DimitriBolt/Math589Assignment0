def find_global_minimum(f, x0, learning_rate=0.1, tol=1e-6):
    """
    Find global minimum of a function f which admits the following syntax:
       y, grad = f(x)
    where y is the value of the function at x and grad is its gradient at x.
    The grad is a tuple of the same length as x.

    Parameters:
    f (callable):          A strictly convex function that accepts either one or two positional arguments.
    x0 (float|tuple|list): Initial approximation. If x0 is a float or int, f takes one argument.
                           If x0 is a tuple or list of length 2, f takes two arguments.
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

    # Determine if x0 is scalar or vector
    if isinstance(x0, (float, int)):
        x = x0
        is_scalar = True
    elif isinstance(x0, (list, tuple)) and len(x0) == 2:
        x = list(x0)
        is_scalar = False
    else:
        raise TypeError("x0 must be a float, int, or a list/tuple of length 2")

    max_iter = 500_000  # To avoid infinite loops
    iter_count = 0

    # Initialize y and grad
    try:
        if is_scalar:
            y, grad = f(x)
        else:
            y, grad = f(x[0], x[1])
    except Exception as e:
        raise Exception(f"Error evaluating function f at initial point x0: {e}")

    while iter_count < max_iter:
        # Update x
        if is_scalar:
            x_new = x - learning_rate * grad[0]
        else:
            x_new = [xi - learning_rate * gi for xi, gi in zip(x, grad)]

        # Compute new y and grad
        try:
            if is_scalar:
                y_new, grad_new = f(x_new)
            else:
                y_new, grad_new = f(x_new[0], x_new[1])
        except Exception as e:
            raise Exception(f"Error evaluating function f at x = {x_new}: {e}")

        # Compute change in y
        y_diff = abs(y_new - y)

        # Compute gradient norm
        grad_norm = math.sqrt(sum(g ** 2 for g in grad_new))

        # Check convergence
        if y_diff < tol or grad_norm < tol:
        # if abs(x_new - x) < tol or grad_norm < tol:
            x = x_new
            y = y_new
            break  # Converged
        else:
            x = x_new
            y = y_new
            grad = grad_new
            iter_count += 1

            # Optionally decrease the learning rate
            # learning_rate = learning_rate / (1 + 0.005 * iter_count)
            learning_rate = learning_rate / (1 + 0.00003)

    else:
        raise Exception("Maximum iterations exceeded")

    # Return the final x and y
    if is_scalar:
        return x, y
    else:
        return tuple(x), y
