import math
import numpy as np


def quadratic1_(x, gamma):
    return gamma * (x**2) + x + 1


def cubic1_(x, gamma):
    return x**3 + gamma * x**2 + x + 1


def multitrous1_(x, gamma):
    return 20 * np.cos(x**2) + (gamma * x**2)


def quadratic2_(x, gamma):
    return quadratic1_(x[0], gamma[0]) + quadratic1_(x[1], gamma[1])


def cubic2_(x, gamma=10):
    return cubic1_(x[0], gamma) + cubic1_(x[1], gamma)


def multitrous2_(x, gamma=4):
    return multitrous1_(x[0], 1) + multitrous1_(x[1], gamma)


def create_system(dim, cond=10, seed=100):
    """Build a symmetric positive-definite A and a vector b.

    A is generated as U^T U with U upper-triangular. The diagonal of U is set
    so the condition number of A is approximately `cond` (the two pinned
    diagonal entries fix the extreme eigenvalues).
    """
    np.random.seed(seed)
    A = 0.1 * np.random.uniform(-math.sqrt(cond), math.sqrt(cond), size=(dim, dim))
    A = np.triu(A)
    A = (
        A
        - np.diag(np.diag(A))
        + np.diag(np.random.uniform(1.0, math.sqrt(cond), size=(dim)))
    )
    A[0, 0] = 1.0
    A[1, 1] = math.sqrt(cond)
    b = 1.0 * np.random.randint(-10, 10, size=(dim))
    A = A.T @ A
    return A, b


def quadraticn_(x, A, b):
    """Generic n-dimensional convex quadratic: x -> (1/2) x^T A x - b^T x.

    Note: the notebook original closes over module-level A, b. Here they are
    passed explicitly so the function is reusable.
    """
    return (x.T @ A @ x) / 2 - b.T @ x


def Rosenbrock(x, gamma=100):
    """Rosenbrock banana: (x0 - 1)^2 + gamma (x0^2 - x1)^2. Min at (1, 1)."""
    return (x[0] - 1) ** 2 + gamma * (x[0] ** 2 - x[1]) ** 2
