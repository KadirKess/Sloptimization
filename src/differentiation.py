import numpy as np


def partial(f, x, i=0, dx=1e-8):
    """Centered finite-difference partial derivative of f along axis i at x.

    Error is O(dx^2) in theory; dominated by float rounding for dx too small.
    """
    h = np.zeros(x.size)
    h[i] = dx
    return (f(x + h) - f(x - h)) / (2 * dx)


def gradient(f, x, dx=1e-8):
    """Finite-difference gradient of f at x. Calls f 2*n times for n = x.size."""
    grad = np.zeros(x.size)
    for i in range(x.size):
        grad[i] = partial(f, x, i, dx)
    return grad
