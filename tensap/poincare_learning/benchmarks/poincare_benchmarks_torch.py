

import numpy as np
import torch
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


def _fun_torch_wrapper(fun_torch):
    vfun_torch = torch.vmap(fun_torch)
    vfun_jac_torch = torch.vmap(torch.func.jacrev(fun_torch))

    def fun(x):
        return np.array(vfun_torch(torch.asarray(x)))

    def fun_jac(x):
        return np.array(vfun_jac_torch(torch.asarray(x)))

    return fun, fun_jac


def _build_borehole():

    X = np.empty(8, dtype=object)
    X[0] = tensap.NormalRandomVariable(0.1, 0.0161812)
    X[1] = tensap.NormalRandomVariable(0, 1)
    X[2] = tensap.UniformRandomVariable(63070, 115600)
    X[3] = tensap.UniformRandomVariable(990, 1110)
    X[4] = tensap.UniformRandomVariable(63.1, 116)
    X[5] = tensap.UniformRandomVariable(700, 820)
    X[6] = tensap.UniformRandomVariable(1120, 1680)
    X[7] = tensap.UniformRandomVariable(9855, 12045)
    X = tensap.RandomVector(X)

    shift, scaling = _get_shift_scaling(X)
    X = X.get_standard_random_vector()
    shift, scaling = torch.asarray(shift), torch.asarray(scaling)

    def fun_torch(x):
        x = x * scaling + shift
        out = (
            2
            * torch.pi
            * x[2]
            * (x[3] - x[5])
            / (
                torch.log(torch.exp(7.71 + 1.0056 * x[1]) / x[0])
                * (
                    1
                    + 2
                    * x[6]
                    * x[2]
                    / torch.log(torch.exp(7.71 + 1.0056 * x[1]) / x[0])
                    / x[0] ** 2
                    / x[7]
                    + x[2] / x[4]
                )
            )
        )
        return out

    return fun_torch, X


def _build_ishigami(d=3, a=7, b=0.1):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-np.pi, np.pi), d)

    def fun_torch(x):
        return (
            torch.sin(x[0])
            + a * torch.sin(x[1]) ** 2
            + b * x[2] ** 4 * torch.sin(x[0])
        )

    return fun_torch, X


def _build_sin_of_a_sum(d=3, c=None):

    X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

    if c is None:
        c = torch.asarray(np.ones(1, d))

    def fun_torch(x):
        return torch.sin(torch.matmul(c, x))

    return fun_torch, X


def _build_canonical_rank_2(d=3):

    X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

    def fun_torch(x):
        return x[0] * x[1] * x[2] + x[0] ** 2 + x[1]

    return fun_torch, X


def _build_mixture(d=6):

    X = tensap.RandomVector(tensap.UniformRandomVariable(), d)

    def fun_torch(x):
        return (
            torch.sin(x[0] + x[3]) * torch.exp(x[4]) * x[5]
            + torch.sin(x[2] * x[3]) * x[5]
        )

    return fun_torch, X


def _build_field(d=6):

    X = tensap.RandomVector(tensap.UniformRandomVariable(0, 1), d)

    def fun_torch(x):
        return (
            1
            + torch.cos(x[0]) * x[1]
            + torch.sin(x[0]) * x[2]
            + torch.exp(x[0]) * x[3]
            + 1 / (x[0] + 1) * x[4]
            + 1 / (2 * x[0] + 3) * x[5]
        )

    return fun_torch, X


def _build_henon_heiles(d=3):

    X = tensap.RandomVector(tensap.NormalRandomVariable(), d)

    def fun_torch(x):
        return (
            0.5 * torch.sum(x ** 2)
            + 0.2 * torch.sum(x[:-1] * x[1:] ** 2 - x[:-1] ** 3)
            + 0.2 ** 2 / 16 * torch.sum((x[:-1] ** 2 + x[1:] ** 2) ** 2)
        )

    return fun_torch, X


def _build_sin_squared_norm(d=8, c=1., R=None):

    if R is None:
        R = np.eye(d)

    R_torch = torch.asarray(R)

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        return torch.sin(c**2 * x.T @ R_torch @ x)

    return fun_torch, X


def _build_cos_squared_norm(d=8, c=1., R=None):

    if R is None:
        R = np.eye(d)

    R_torch = torch.asarray(R)

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        return torch.cos(c**2 * (x.T @ R_torch @ x))

    return fun_torch, X


def _build_sum_cos_sin_squared_norm(d=8, c=1., R1=None, R2=None):

    if R1 is None:
        R1 = np.eye(d)
    if R2 is None:
        R2 = np.eye(d)

    R1_torch = torch.asarray(R1)
    R2_torch = torch.asarray(R2)

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        u1 = torch.cos(c**2 * (x.T @ R1_torch @ x))
        u2 = torch.sin(c**2 * (x.T @ R2_torch @ x))
        return u1 + u2

    return fun_torch, X


def _build_exp_mean_sin_exp_cos(d=8, c=1.):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        y = torch.cos(c * x)
        z = torch.sin(c * x) * torch.exp(y)
        return torch.exp(torch.mean(z))

    return fun_torch, X


def _build_cos_low_rank(d=8):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        z1 = torch.flatten(torch.outer(x[:d // 2], x[:d // 4]))
        z2 = torch.flatten(torch.outer(x[-d // 2:], x[-d // 4:]))
        z = z1.T @ z2
        return torch.cos(z)

    return fun_torch, X


def _build_schaffer(d=7):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        y = x[:-1]**2 + x[1:]**2
        z = (torch.sin(100 * y**0.5)**2 - 0.5) / (1 + 10 * y)**2
        return 3 + torch.sum(z)

    return fun_torch, X


def _build_ackley(d=7):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        y = x * 32.768
        z0 = 20 + torch.exp(torch.ones(1))
        z1 = torch.mean(torch.sum(y**2))
        z2 = torch.mean(torch.sum(torch.cos(2 * torch.pi * y)))
        return z0 - 20 * torch.exp(- 0.2 * torch.sqrt(z1)) - torch.exp(z2)

    return fun_torch, X


def _build_robot_arm(d=8):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    def fun_torch(x):
        y1 = (x[:d // 2] + 1) / 2
        y2 = torch.cumsum(torch.pi * x[d // 2:], 0)
        u = torch.sum(y1 * torch.cos(y2))
        v = torch.sum(y1 * torch.sin(y2))
        return torch.sqrt(u**2 + v**2)

    return fun_torch, X


def _build_piston(d=6):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    X = np.empty(7, dtype=object)
    X[0] = tensap.UniformRandomVariable(30, 60)
    X[1] = tensap.UniformRandomVariable(0.005, 0.002)
    X[2] = tensap.UniformRandomVariable(0.002, 0.01)
    X[3] = tensap.UniformRandomVariable(1000, 5000)
    X[4] = tensap.UniformRandomVariable(90000, 110000)
    X[5] = tensap.UniformRandomVariable(290, 296)
    X[6] = tensap.UniformRandomVariable(340, 360)
    X = tensap.RandomVector(X)

    shift, scaling = _get_shift_scaling(X)
    X = X.get_standard_random_vector()
    shift, scaling = torch.asarray(shift), torch.asarray(scaling)

    def fun_torch(x):
        x = x * scaling + shift
        aux = x[4] * x[2] / x[6]
        A = x[1] * x[4] + 19.62 * x[0] - x[2] * x[3] / x[1]
        V = x[1] / (2 * x[3]) * (A**2 + 4 * x[3] * aux * x[5] - A)**0.5
        out = 2 * torch.pi * (x[0] / (x[3] + x[1] * aux * x[5] / V**2))**0.5
        return out

    return fun_torch, X


def _build_cos_tree(d=8):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)
    ind_lst = [np.arange(i * d // 4, (i + 1) * d // 4) for i in range(4)]

    def fun_torch(x):
        z0 = torch.cos(torch.sum(x[ind_lst[0]]**2))
        z1 = torch.cos(torch.sum(x[ind_lst[1]]**2))
        z2 = torch.cos(torch.sum(x[ind_lst[2]]**2))
        z3 = torch.cos(torch.sum(x[ind_lst[3]]**2))
        y1 = torch.cos(2 * np.pi * (z0 + z1))
        y2 = torch.cos(2 * np.pi * (z2 * z3))
        out = y1 + y2
        return out

    return fun_torch, X


def _build_quartic_sin_collective(d=9, mat_lst=[]):

    X = tensap.RandomVector(tensap.UniformRandomVariable(-1, 1), d)

    if len(mat_lst) == 0:
        mat_lst = [np.eye(d - 1)]

    mat_lst_torch = []
    for M in mat_lst:
        mat_lst_torch.append(torch.asarray(M))

    def fun_torch(x):
        z = 0.
        for i, M in enumerate(mat_lst_torch):
            zi = (x[:-1].T @ M @ x[:-1])
            zi = zi**2
            c = (torch.pi * (i + 1)) / (2 * len(mat_lst))
            zi *= torch.sin(c * x[-1])
            z += zi
        out = z
        return out

    return fun_torch, X


def build_benchmark_torch(case, **kwargs):
    """
    Generate different functions used to benchmark the package.

    Parameters
    ----------
    case : str
        The name of the function. Can be 'borehole', 'ishigami,
        'sin_of_asum', 'sin_squared_norm', 'canonical_rank_2', 'mixture',
        'field', 'henon_heiles'.
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

    torch.set_default_device('cpu')

    if case == "borehole":
        fun_torch, X = _build_borehole(**kwargs)

    elif case == "ishigami":
        fun_torch, X = _build_ishigami(**kwargs)

    elif case == "sin_of_asum":
        fun_torch, X = _build_sin_of_a_sum(**kwargs)

    elif case == "canonical_rank_2":
        fun_torch, X = _build_canonical_rank_2(**kwargs)

    elif case == "mixture":
        fun_torch, X = _build_mixture(**kwargs)

    elif case == "field":
        fun_torch, X = _build_field(**kwargs)

    elif case == "henon_heiles":
        fun_torch, X = _build_henon_heiles(**kwargs)

    elif case == "sin_squared_norm":
        fun_torch, X = _build_sin_squared_norm(**kwargs)

    elif case == "cos_squared_norm":
        fun_torch, X = _build_cos_squared_norm(**kwargs)

    elif case == "sum_cos_sin_squared_norm":
        fun_torch, X = _build_sum_cos_sin_squared_norm(**kwargs)

    elif case == "exp_mean_sin_exp_cos":
        fun_torch, X = _build_exp_mean_sin_exp_cos(**kwargs)

    elif case == "cos_low_rank":
        fun_torch, X = _build_cos_low_rank(**kwargs)

    elif case == "schaffer":
        fun_torch, X = _build_schaffer(**kwargs)

    elif case == "ackley":
        fun_torch, X = _build_ackley(**kwargs)

    elif case == "robot_arm":
        fun_torch, X = _build_robot_arm(**kwargs)

    elif case == "piston":
        fun_torch, X = _build_piston(**kwargs)

    elif case == "cos_tree":
        fun_torch, X = _build_cos_tree(**kwargs)

    elif case == "quartic_sin_collective":
        fun_torch, X = _build_quartic_sin_collective(**kwargs)

    else:
        raise NotImplementedError("Function not implemented.")

    fun, fun_jac = _fun_torch_wrapper(fun_torch)

    return fun, fun_jac, X
