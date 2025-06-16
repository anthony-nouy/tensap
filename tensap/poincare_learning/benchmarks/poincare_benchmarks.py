

import numpy as np
import tensap


def _get_shift_scaling(multivariate_random_variable):
    X = multivariate_random_variable
    shift = np.zeros(X.ndim())
    scaling = np.ones(X.ndim())
    for i in range(X.ndim()):
        poly = X.random_variables[i].orthonormal_polynomials()
        if hasattr(poly, "shift"):
            shift[i] = poly.shift
        if hasattr(poly, "scaling"):
            scaling[i] = poly.scaling
    return shift, scaling


def _build_sin_squared_norm(d=8, c=1., R=None):

    if R is None:
        R = np.eye(d)

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun(x):
        z = c**2 * np.einsum('ki,ki->k', x, x @ R).reshape(-1, 1)
        return np.sin(z)

    def fun_jac(x):
        z = c**2 * np.einsum('ki,ki->k', x, x @ R).reshape(-1, 1)
        out = 2 * c**2 * np.cos(z) * (x @ R)
        return out[:, None, :]

    return fun, fun_jac, X


def _build_cos_squared_norm(d=8, c=1., R=None):

    if R is None:
        R = np.eye(d)

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun(x):
        z = c**2 * np.einsum('ki,ki->k', x, x @ R).reshape(-1, 1)
        return np.cos(z)

    def fun_jac(x):
        z = c**2 * np.einsum('ki,ki->k', x, x @ R).reshape(-1, 1)
        out = -2 * c**2 * np.sin(z) * (x @ R)
        return out[:, None, :]

    return fun, fun_jac, X


def _build_sum_cos_sin_squared_norm(d=8, c=1., R1=None, R2=None):

    if R1 is None:
        R1 = np.eye(d)
    if R2 is None:
        R2 = np.eye(d)

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun(x):
        z1 = c**2 * np.einsum('ki,ki->k', x, x @ R1).reshape(-1, 1)
        z2 = c**2 * np.einsum('ki,ki->k', x, x @ R2).reshape(-1, 1)
        return np.cos(z1) + np.sin(z2)

    def fun_jac(x):
        z1 = c**2 * np.einsum('ki,ki->k', x, x @ R1).reshape(-1, 1)
        z2 = c**2 * np.einsum('ki,ki->k', x, x @ R2).reshape(-1, 1)
        du1 = - 2 * c**2 * np.sin(z1) * (x @ R1)
        du2 = 2 * c**2 * np.cos(z2) * (x @ R2)
        out = du1 + du2
        return out[:, None, :]

    return fun, fun_jac, X


def _build_exp_mean_sin_exp_cos(d=8, c=1.):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def g(x):
        y = np.cos(c * x)
        z = np.sin(c * x) * np.exp(y)
        return np.mean(z, axis=1).reshape(-1, 1)

    def jac_g(x):
        out = (np.cos(c * x) - np.sin(c * x)**2) * np.exp(np.cos(c * x)) * c / d
        return out[:, None, :]

    def fun(x):
        return np.exp(g(x))

    def fun_jac(x):
        return fun(x)[:, None, :] * jac_g(x)

    return fun, fun_jac, X


def build_benchmark(case, **kwargs):
    """
    Generate different functions used to benchmark the package.

    Parameters
    ----------
    case : str
        The name of the function. Can be 'sin_squared_norm', 'cos_squared_norm'.
    **kwargs
        Parameters of the function.

    Raises
    ------
    NotImplementedError
        If the function is not implemented.

    Returns
    -------
    fun : callable
        The asked function.
    fun_jac : callable
        The jacobian of the function
    X : tensap.RandomVector
        Input random variables.

    """

    if case == "sin_squared_norm":
        fun, fun_jac, X = _build_sin_squared_norm(**kwargs)
    elif case == "cos_squared_norm":
        fun, fun_jac, X = _build_cos_squared_norm(**kwargs)
    elif case == "sum_cos_sin_squared_norm":
        fun, fun_jac, X = _build_sum_cos_sin_squared_norm(**kwargs)
    elif case == "exp_mean_sin_exp_cos":
        fun, fun_jac, X = _build_exp_mean_sin_exp_cos(**kwargs)
    else:
        raise NotImplementedError("Function not implemented.")

    return fun, fun_jac, X
