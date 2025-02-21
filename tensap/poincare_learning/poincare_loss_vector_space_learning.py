

import numpy as np
import scipy
from tensap.poincare_learning.utils._loss_vector_space import _eval_HG_X, _eval_SGinv_X, _eval_jac_g, poincare_loss_vector_space, poincare_loss_vector_space_gradient, _eval_surrogate_matrices, poincare_loss_surrogate_vector_space


def _iteration_qn(jac_u, jac_basis, G, R=None, cg_kwargs={}):
    """
    Perform one iteration of the quasi Newton algorithm described in 
    Bigoni et al. 2022.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    G : numpy.ndarray
        The coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).
    R : numpy.ndarray, optional
        The inner product matrix with respect to which G is orthonormal.
        The default is None, corresponding to identity matrix.
    cg_kwargs : dict
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
    Gaux = _eval_SGinv_X(Gmat, b, jac_u, jac_basis, jac_g, cg_kwargs)
    M = Gaux.T @ R @ Gaux
    Gnext = Gaux @ np.linalg.inv(np.linalg.cholesky(M).T)
    return Gnext


def _minimize_qn_(jac_u, jac_basis, G0, R=None, maxiter_qn=100, tol_qn=1e-5, verbosity=2, cg_kwargs={}):
    """
    Perform the quasi Newton algorithm described in Bigoni et al. 2022, 
    starting at a single initial point.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    G0 : numpy.ndarray
        Initialization point of the algorithm.
        Has shape (K, m).
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
    cg_kwargs : dict
        Key word arguments for scipy.sparse.linalg.cg to solve S(G)x=b
        at each iteration.

    Returns
    -------
    out : numpy.ndarray
        Result of the QN algorithm.
        Has shape .

    """
    N, K, d = jac_basis.shape
    if R is None:
        R = np.eye(K)
    M0 = G0.T @ R @ G0
    G_now = G0 @ np.linalg.inv(np.linalg.cholesky(M0).T)
    i = -1
    delta = np.inf
    if verbosity >= 1:
        print("Optimizing Poincare loss with QN from Bigoni et al.")
    while i < maxiter_qn and delta >= tol_qn:
        i = i+1
        G_next = _iteration_qn(jac_u, jac_basis, G_now, R, cg_kwargs)
        # delta = np.linalg.norm(Gnext - Gnow)
        delta = 1 - np.linalg.svd(G_next.T @ R @ G_now)[1].min()
        G_now[:] = G_next[:]
        if verbosity >= 2:
            err = poincare_loss_vector_space(G_now, jac_u, jac_basis)
            print(f"| Iter:{i} loss:{err:.3e} step_size:{delta:.3e}")
    
    G = G_now
    loss = poincare_loss_vector_space(G, jac_u, jac_basis)

    return G, loss


def _minimize_qn(jac_u, jac_basis, G0=None, m=None, n_try=None, R=None, maxiter_qn=100, tol_qn=1e-5, verbosity=2, cg_kwargs={}):
    """
    Perform the quasi Newton algorithm described in Bigoni et al. 2022, 
    starting at potentially multiple initial points.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    G0 : numpy.ndarray, optional
        Initialization point of the algorithm.
        If None, the algorithm takes n_try random initial points.
        Has shape (n_try, K, m) or (K, m).
        The default is None.
    m : int, optional
        Only used if G0 is None.
        The default is None.
    n_try : in, optional
        Only used if G0 is None
        The default is None.
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
    cg_kwargs : dict
        Key word arguments for scipy.sparse.linalg.cg to solve S(G)x=b
        at each iteration.

    Returns
    -------
    G : numpy.ndarray
        Result of the QN algorithm.
        Has shape (n_try, K, m) or (K, m).

    """
    K = jac_basis.shape[1]

    if R is None:
        R = np.eye(K)

    if G0 is None:
        assert not(m is None)
        G0 = np.random.normal(size=(n_try, K, m))

    elif G0.ndim == 2:
        G0 = G0[None, :, :]
        
    l, _, m = G0.shape
    loss = np.inf * np.ones(l)
    G = 0 * G0

    for i in range(l):
        G[i], loss[i] = _minimize_qn_(jac_u, jac_basis, G0[i], R, maxiter_qn, tol_qn, verbosity, cg_kwargs)
        
    if l == 1:
        loss = loss[0]
        G = G[0]

    return G, loss


def _minimize_pymanopt(jac_u, jac_basis, G0=None, m=None, n_try=None, use_precond=True, precond_kwargs={}, optimizer_kwargs={}, ls_kwargs={}):
    """
    Minimize the Poincare loss using a conjugate gradient algorithm on 
    the Grassmann manifold Grass(K, m).

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    G0 : numpy.ndarray, optional
        Initialization point of the algorithm.
        If None, the algorithm takes n_try random initial points.
        Has shape (n_try, K, m) or (K, m).
        The default is None.
    m : int, optional
        Only used if G0 is None.
        The default is None.
    n_try : in, optional
        Only used if G0 is None
        The default is None.
    use_precond : bool, optional
        If True, use the precond from the quasi Newton algorithm escribed in 
        Bigoni et al. 2022, meaning taking S(G) as approximate Hessian.
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
    G : numpy.ndarray0 = G0[None, :, :]
        
        Minimizers for each initial point.
        Has shape (n_try, K, m) of (K, m).
    loss : numpy.ndarray
        Minimal costs for initial point.

    """
    K = jac_basis.shape[1]

    if G0 is None:
        assert not(m is None)
        G0 = np.random.normal(size=(n_try, K, m))

    elif G0.ndim == 2:
        G0 = G0[None, :, :]

    l, _, m = G0.shape
    problem, optimizer = _build_pymanopt_problem(jac_u, jac_basis, m, use_precond, optimizer_kwargs, precond_kwargs, ls_kwargs)
    loss = np.inf * np.ones(l)
    G = 0 * G0

    for i in range(l):
        optim_result = optimizer.run(problem, initial_point=G0[i])
        loss[i] = optim_result.cost
        G[i] = optim_result.point

    if l == 1:
        loss = loss[0]
        G = G[0]
        
    return G, loss


def _build_pymanopt_problem(jac_u, jac_basis, m, use_precond=True, optimizer_kwargs={}, precond_kwargs={}, ls_kwargs={}):
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
    manifold = pymanopt.manifolds.grassmann.Grassmann(K, m)

    @pymanopt.function.numpy(manifold)
    def cost(G): return poincare_loss_vector_space(G, jac_u, jac_basis)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(G): return poincare_loss_vector_space_gradient(G, jac_u, jac_basis)

    precond = None
    if use_precond:
        def precond(G, x): return _eval_SGinv_X(G, x, jac_u, jac_basis, None, precond_kwargs)

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient, preconditioner=precond)
    line_search = pymanopt.optimizers.line_search.BackTrackingLineSearcher(**ls_kwargs)
    optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient(**optimizer_kwargs, line_searcher=line_search)

    return problem, optimizer


def _minimize_surrogate(jac_u, jac_basis, G0=None, R=None, m=1):
    """
    Compute the minimizer of the surrogate proposed in Nouy et al. 2025.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    G0 : numpy.ndarray, optional
        The coefficients of the fixed features maps.
        If None, it corresponds to the one feature setting.
        Has shap (K, j). 
        The default is None.
    R : numpy.ndarray, optional
        The inner product matrix with respect to which G0 is orthonormal.
        The default is None, corresponding to identity matrix.
    m : int, optional
        Number of singular vectors to take as features. 
        Recall that exact recovery only holds for m <= dim(u(x)).
        The default is 1.

    Returns
    -------
    G : numpy.ndarray
        Coefficients in the basis of feature maps.
        Has shape (K, m).
    loss : float
        The Poincare loss of the minimizer of the surrogate.
    surrogate : float
        The minimum of the surrogate.
    """

    K = jac_basis.shape[1]
    
    if R is None:
        R = np.eye(K)
    
    # Orthonormalize G0 if necessary
    if not(G0 is None):
        M = G0.T @ R @ G0
        if np.linalg.norm(M - np.eye(M.shape[0])) > 1e-6:
            G0 = G0 @ np.linalg.inv(np.linalg.cholesky(M).T)
    
    A, B, C = _eval_surrogate_matrices(jac_u, jac_basis, G0, R)
    eigvals, eigvec = scipy.linalg.eigh(B - A + C, R)
    G = eigvec[:,:m]
    surrogate = eigvals.min()

    if not(G0 is None):
        G = np.hstack([G0, G])
    
    loss = poincare_loss_vector_space(G, jac_u, jac_basis)

    return G, loss, surrogate


def _minimize_surrogate_greedy(jac_u, jac_basis, m_max, R=None, optimize_poincare=True, tol=1e-7, verbose=2, pmo_kwargs={}):
    """
    Greedy algorithm to learn multiple features, as proposed in 
    Nouy et al. 2025.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    m_max : int
        Maximum number of features to learn.
    R : numpy.ndarray, optional
        The inner product matrix with respect to which G is orthonormal.
        The default is None, corresponding to identity matrix.
    optimize_poincare : bool, optional
        If True, perform a Poincare loss minimization after each 
        surrogate minimization. The surrogate minimizer then
        serves as an initializer to the minimization problem
        on the Poincare loss.
        The default is True.
    tol : float, optional
        The greedy algorithm stops if the Poincare loss is smaller.
        The default is 1e-7
    verbose : int, optional
        Verbosity level.
        The default is 2.
    pmo_kwargs
        Key word arguments for the minimization algorithm with pymanopt.
        For more details see _minimize_pymanopt.

    Returns
    -------
    G : numpy.ndarray
        Coefficients in the basis of feature maps.
        Has shape (K, m)
    losses : numpy.ndarray
        Poincare losses of the minimizers of the surrogates, 
        at each iterations
        Has shape (m, )
    losses_optimized : numpy.ndarray
        Poincare losses minimized starting from the minimizers of 
        the surrogates, at each iterations.
        Has shape (m, )
    surrogates : numpy.ndarray
        Values of the surrogates at each iterations.
        Has shape (m, )

    """
    N, K, d = jac_basis.shape
    if R is None:
        R = np.eye(K)
    losses = -np.ones(m_max)
    losses_optimized = -np.ones(m_max)
    surrogates = -np.ones(m_max)

    if verbose >=1: print(f"Greedy iteration {1}")

    # Learn first feature from surrogate
    G, losses[0], surrogates[0] = _minimize_surrogate(jac_u, jac_basis, None, R)
    
    # Run minimization of Poincare loss if necessary
    if optimize_poincare:
        G, losses_optimized[0] = _minimize_pymanopt(jac_u, jac_basis, G, **pmo_kwargs)
        if not(R is None):
            G = G @ np.linalg.inv(np.linalg.cholesky(G.T @ R @ G).T)
    else:
        losses_optimized[0] = losses[0]

    if verbose >=2: 
            print(f"=Surrogate   : {surrogates[0]:.3e}")
            print(f"=Loss        : {losses[0]:.3e}")
            if optimize_poincare:
                print(f"=Loss optim  : {losses_optimized[0]:.3e}")

    j = 1

    while j < m_max and losses_optimized[j-1] > tol:

        if verbose >=1: print(f"Greedy iteration {j+1}")

        # Learn the j-th feature from surrogate and previous features
        G, losses[j], surrogates[j] = _minimize_surrogate(jac_u, jac_basis, G, R)

        # Run minimization of Poincare loss on all features if necessary
        if optimize_poincare:
            G, losses_optimized[j] = _minimize_pymanopt(jac_u, jac_basis, G, **pmo_kwargs)
            if not(R is None):
                G = G @ np.linalg.inv(np.linalg.cholesky(G.T @ R @ G).T)
        else:
            losses_optimized[j] = losses[j]

        if verbose >=2: 
            print(f"=Surrogate   : {surrogates[j]:.3e}")
            print(f"=Loss        : {losses[j]:.3e}")
            if optimize_poincare:
                print(f"=Loss optim  : {losses_optimized[j]:.3e}")

        j += 1

    # shorten the results if greedy stopped early
    losses = losses[:j]
    losses_optimized = losses_optimized[:j]
    surrogates = surrogates[:j]

    return G, losses, losses_optimized, surrogates

