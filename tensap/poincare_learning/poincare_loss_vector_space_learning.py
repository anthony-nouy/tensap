

import numpy as np
import scipy
from tensap.poincare_learning.poincare_loss_vector_space import poincare_loss_vector_space, poincare_loss_vector_space_gradient, _eval_jac_g, _eval_HG_X, _eval_SGinv_X


def _iteration_qn(G, jac_u, jac_basis, R=None, **cg_kwargs):
    """
    Perform one iteration of the quasi Newton algorithm described in 
    Bigoni et al. 2022.

    Parameters
    ----------
    G : numpy.ndarray
        The coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    R : numpy.ndarray, optional
        The inner product matrix with respect to which G is orthonormal.
        The default is None, corresponding to identity matrix.
    **cg_kwargs : dict
        Key word arguments for scipy.sparse.linalg.cg to solve S(G)x=b

    Returns
    -------
    Gnext : numpy.ndarray
        The updated coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).

    """
    N, K, d = jac_basis.shape
    if R is None:
        R = np.eye(K)
    Gmat = G.reshape(K, -1, order='F')
    jac_g = _eval_jac_g(Gmat, jac_basis)
    b = _eval_HG_X(Gmat, Gmat, jac_u, jac_basis, jac_g)
    Gaux = _eval_SGinv_X(Gmat, b, jac_u, jac_basis, jac_g, **cg_kwargs)
    M = Gaux.T @ R @ Gaux
    Gnext = Gaux @ np.linalg.inv(np.linalg.cholesky(M).T)
    return Gnext


def poincare_minimize_qn(G0, jac_u, jac_basis, R=None, maxiter_qn=100, tol_qn=1e-5, verbosity=2, **cg_kwargs):
    """
    Perform one iteration of the quasi Newton algorithm described in 
    Bigoni et al. 2022.

    Parameters
    ----------
    G0 : numpy.ndarray
        Initialization point of the qn algorithm.
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    R : numpy.ndarray, optional
        The inner product matrix with respect to which G is orthonormal.
        The default is None, corresponding to identity matrix.
    maxiter_qn : int, optional
        Maximal number of iteration of the QN algorithm.
        The default is 100.
    tol_qn : float, optional
        Tolerance for QN algorithm.
        The default is 1e-5
    verbosity : int, optional
        Verbosity parameter.
    **cg_kwargs : dict
        Key word arguments for scipy.sparse.linalg.cg to solve S(G)x=b
        at each iteration.

    Returns
    -------
    out : numpy.ndarray
        Result of the QN algorithm.
        Has same shape as G0.

    """
    N, K, d = jac_basis.shape
    if R is None:
        R = np.eye(K)
    G0mat = G0.reshape(K, -1, order='F')
    M0 = G0mat.T @ R @ G0mat
    G0mat = G0mat @ np.linalg.inv(np.linalg.cholesky(M0).T)
    Gnow = G0mat[:]
    i = -1
    delta = np.inf
    if verbosity >= 1:
        print("Optimizing Poincare loss with QN from Bigoni et al.")
    while i < maxiter_qn and delta >= tol_qn:
        i = i+1
        Gnext = _iteration_qn(Gnow, jac_u, jac_basis, R, **cg_kwargs)
        # delta = np.linalg.norm(Gnext - Gnow)
        delta = 1 - np.linalg.svd(Gnext.T @ R @ Gnow)[1].min()
        Gnow[:] = Gnext[:]
        if verbosity >= 2:
            err = poincare_loss_vector_space(Gnow, jac_u, jac_basis)
            print(f"| Iter:{i} loss:{err:.3e} step_size:{delta:.3e}")
    out = Gnow.reshape(G0.shape, order='F')
    return out


def poincare_minimize_pymanopt(G0, jac_u, jac_basis, use_precond=False, precond_kwargs={}, optimizer_kwargs={}, ls_kwargs={}):
    """
    Minimize the Poincare loss using a conjugate gradient algorithm on the Grassmann manifold Grass(K, m).

    Parameters
    ----------
    G0 : numpy.ndarray
        Initialization point of the qn algorithm.
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    use_precond : bool, optional
        If True, use the precond from the quasi Newton algorithm escribed in Bigoni et al. 2022, meaning taking S(G) as approximate Hessian.
        The default is False.
    optimizer_kwargs : dict, optional
        Key word arguments of the pymanopt ConjugateGradient optimizer.
        See pymanopt documentation for more details.
        The default is {}.
    precond_kwargs : dict, optional
        Key word arguments for scipy.sparse.linalg.cg to inverse S(G)
        as a preconditioner.
        See scipy.sparse.linalg.cg documentation for more details.
        The default is {}.
    ls_kwargs : dict, optional
        Key word arguments for the pymanopt line search algorithm 
        BackTrackingLineSearcher
        See pymanopt documentation for more details.
        The default is {}.

    Returns
    -------
    optim_result : pymanopt.optimizers.optimizer.OptimizerResult
        The result of the conjugate gradient algorithm
        See pymanopt documentation for more details.

    """
    K = jac_basis.shape[1]
    if G0.ndim == 1:
        G0mat = G0.Reshape((K, G0.shape[0] // K))
    else:
        G0mat = G0
    m = G0mat.shape[1]
    problem, optimizer = _build_pymanopt_problem(jac_u, jac_basis, m, use_precond, optimizer_kwargs, precond_kwargs, ls_kwargs)
    optim_result = optimizer.run(problem, initial_point=G0)
    return optim_result


def _build_pymanopt_problem(jac_u, jac_basis, m, use_precond=False, optimizer_kwargs={}, precond_kwargs={}, ls_kwargs={}):
    """
    Build instances of pymanopt classes Problem and ConjugateGradient
    which can be used to minimize the Poincare loss using a conjugate 
    gradient algorithm on the Grassmann manifold Grass(K, m).

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    m : int
        Number of features to learn.
    use_precond : bool, optional
        If True, use the precond from the quasi Newton algorithm escribed in Bigoni et al. 2022, meaning taking S(G) as approximate Hessian.
        The default is False.
    optimizer_kwargs : dict, optional
        Key word arguments of the pymanopt ConjugateGradient optimizer.
        See pymanopt documentation for more details.
        The default is {}.
    precond_kwargs : dict, optional
        Key word arguments for scipy.sparse.linalg.cg to inverse S(G)
        as a preconditioner.
        See scipy.sparse.linalg.cg documentation for more details.
        The default is {}.
    ls_kwargs : dict, optional
        Key word arguments for the pymanopt line search algorithm 
        BackTrackingLineSearcher
        See pymanopt documentation for more details.
        The default is {}.

    Returns
    -------
    problem : pymanopt.Problem
        The Optimization problem on the Grassman manifold.
    optimizer : pymanopt.optimizers.conjugate_gradient.ConjugateGradient
        The conjugate gradient optimizer.

    """
    import pymanopt
    K = jac_basis.shape[1]
    manifold = pymanopt.manifold.grassmann.Grassmann(K, m)

    @pymanopt.function.numpy(manifold)
    def cost(G): return poincare_loss_vector_space(G, jac_u, jac_basis)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(G): return poincare_loss_vector_space_gradient(G, jac_u, jac_basis)

    precond = None
    if use_precond:
        def precond(G, x): return _eval_SGinv_X(G, x, jac_u, jac_basis, **precond_kwargs)

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient, preconditioner=precond)
    line_search = pymanopt.optimizers.line_search.BackTrackingLineSearcher(**ls_kwargs)
    optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient(**optimizer_kwargs, line_searcher=line_search)

    return problem, optimizer

