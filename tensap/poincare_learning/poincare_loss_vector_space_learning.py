

import numpy as np
import scipy
import logging
from tensap.poincare_learning.utils._loss_vector_space import _eval_HG_X, _eval_SGinv_X, \
    _eval_SG_full, _eval_HessG_full, _eval_jac_g, poincare_loss_vector_space, \
    poincare_loss_vector_space_gradient, _eval_surrogate_matrices


def _minimize_active_subspace(jac_u, jac_basis=None, m=1):
    """
    Compute the minimizer of the Poincare loss for linear features.
    This corresponds to the active subspace method.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray, optional
        Has shape (N, d, d) or (d, d).
        The defatul is None
    m : int, optional
        Number of singular vectors to take as features.
        The default is 1.

    Returns
    -------
    G : numpy.ndarray
        Coefficients in the basis of feature maps.
        Has shape (d, m).
    """

    # if no basis provided, take identity
    if jac_basis is None:
        jac_basis == np.eye(jac_u.shape[-1])

    # check that basis only contains linear functions
    if jac_basis.ndim == 3:
        jac_basis = jac_basis[0]

    # create the matrix H
    jb_jac_u = np.einsum('ij,lkj->lki', jac_basis, jac_u)
    H = np.einsum('lki,lkj->ij', jb_jac_u, jb_jac_u)

    # compute eigen decomposition and keep largest eigen values
    _, eigvec = scipy.linalg.eigh(H, jac_basis @ jac_basis.T)

    # orthonormalize
    G = np.linalg.svd(eigvec[:, -m:], full_matrices=False)[0]

    return G


def _initialization(jac_u, jac_basis, m, init_method='active_subspace', n_try=1, R=None, seed=None):
    """
    Compute coefficients of the initial feature map according to the choosen method.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray, optional
        Has shape (N, d, d) or (d, d).
        The defatul is None
    m : int, optional
        Number of singular vectors to take as features.
        The default is 1.
    init_method : string, optional
        Only used if G0 is None.
        Initialization method, must be one of
        'random', 'random_linear', 'surrogate', 'surrogate_greedy', 'active_subspace'.
        Note that 'active_subspace' assumes that the first d basis functions are linear.
        The default is 'active_subspace'.
    n_try : int, optional
        Number of initial points to compute
        Only used if G0 is None and init_method is 'random' or 'random_linear'.
        The default is 1.

    Returns
    -------
    G0 : numpy.ndarray
        Initial coefficients.
        Has shape (K, m) of (n_try, K, m).

    """
    N, K, d = jac_basis.shape

    if init_method == 'surrogate':
        G0 = _minimize_surrogate(jac_u, jac_basis, R=R, m=m)[0]

    elif init_method == 'surrogate_greedy':
        G0 = _minimize_surrogate_greedy(jac_u, jac_basis, m, R=R, optimize_poincare=False)[0]

    elif init_method == 'random':
        G0 = np.random.RandomState(seed).normal(size=(n_try, K, m))

    elif init_method == 'random_linear':
        G0 = np.zeros((n_try, K, m))
        G0[:, :d, :] = np.random.RandomState(seed).normal(size=(n_try, d, m))

    elif init_method == 'active_subspace':
        G0 = np.zeros((K, m))
        G0[:d, :] = _minimize_active_subspace(jac_u, jac_basis[0, :d, :], m=m)

    else:
        raise ValueError('Initialization method not valid')

    return G0


def _iteration_qn(jac_u, jac_basis, G, R=None, precond_method='sigma', precond_kwargs={}):
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
    precond_method : str, optional
        Method to use as approximation of the hessian.
        Should be one of
        - None : identity matrix
        - "sigma" : sigma matrix from Bigoni et al. 2022 inverted by lstsq.
        - "sigma_cg" : sigma matrix from Bigoni et al. 2022 inverted by CG .
        - "hessian" : hessian matrix inverted by lstsq.
    precond_kwargs : dict
        Key word arguments the precond method.
        The default is dict()

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

    if precond_method is None:
        dG = b.reshape(G.shape, order='F')

    elif precond_method[:3] != "cg_":
        if precond_method == "sigma":
            P = _eval_SG_full(G, jac_u, jac_basis)

        elif precond_method == "hessian":
            P = _eval_HessG_full(G, jac_u, jac_basis)
            (G, jac_u, jac_basis)

        dG, _, _, _ = np.linalg.lstsq(P, b.reshape(-1, order='F'))

    elif precond_method == "cg_sigma":
        dG = _eval_SGinv_X(Gmat, b, jac_u, jac_basis, jac_g, cg_kwargs=precond_kwargs)

    else:
        Warning("Precond method not implemented", precond_method)

    Gaux = Gmat + dG.reshape(K, -1, order='F')
    M = Gaux.T @ R @ Gaux
    Gnext = Gaux @ np.linalg.inv(np.linalg.cholesky(M).T)
    Gnext = Gnext.reshape(G.shape)
    return Gnext


def _minimize_qn_(jac_u, jac_basis, G0, R=None, maxiter=100, tol=1e-5, precond_method='sigma',
                  precond_kwargs={}):
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
    maxiter : int, optional
        Maximal number of iteration of the QN algorithm.
        The default is 100.
    tol : float, optional
        Tolerance for QN algorithm.
        The default is 1e-5
    precond_method : str, optional
        Method to use as approximation of the hessian.
        Should be one of
        - None : identity matrix
        - "sigma" : sigma matrix from Bigoni et al. 2022 inverted by lstsq.
        - "sigma_cg" : sigma matrix from Bigoni et al. 2022 inverted by CG .
        - "hessian" : hessian matrix inverted by lstsq.
    precond_kwargs : dict
        Key word arguments the precond method.
        The default is dict()
    Returns
    -------
    out : numpy.ndarray
        Result of the QN algorithm.
        Has shape (K, m) or (K*m, ).

    """
    N, K, d = jac_basis.shape
    if R is None:
        R = np.eye(K)
    M0 = G0.T @ R @ G0
    G_now = G0 @ np.linalg.inv(np.linalg.cholesky(M0).T)
    i = -1
    delta = np.inf
    logging.info("Optimizing Poincare loss with QN and precond method " + str(precond_method))
    while i < maxiter and delta >= tol:
        i = i + 1
        G_next = _iteration_qn(jac_u, jac_basis, G_now, R, precond_method, precond_kwargs)
        # delta = np.linalg.norm(Gnext - Gnow)
        delta = 1 - np.linalg.svd(G_next.T @ R @ G_now)[1].min()
        G_now[:] = G_next[:]
        err = poincare_loss_vector_space(G_now, jac_u, jac_basis)
        logging.info(f"| Iter:{i} loss:{err:.3e} step_size:{delta:.3e}")

    G = G_now
    loss = poincare_loss_vector_space(G, jac_u, jac_basis)

    return G, loss


def _minimize_qn(jac_u, jac_basis, G0=None, m=None, n_try=None, init_method="active_subspace",
                 R=None, maxiter=100, tol=1e-5, precond_method='sigma', precond_kwargs={},
                 seed=None):
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
    n_try : int, optional
        Number of initial points to compute
        Only used if G0 is None and init_method is 'random' or 'random_linear'.
        The default is 1.
    init_method : string, optional
        Only used if G0 is None.
        Initialization method, must be one of
        'random', 'random_linear', 'surrogate', 'surrogate_greedy', 'active_subspace'.
        Note that 'active_subspace' assumes that the first d basis functions are linear.
        The default is 'active_subspace'.
    R : numpy.ndarray, optional
        The inner product matrix with respect to which G is orthonormal.
        The default is None, corresponding to identity matrix.
    maxiter : int, optional
        Maximal number of iteration of the QN algorithm.
        The default is 100.
    tol : float, optional
        Tolerance for QN algorithm.
        The default is 1e-5
    precond_method : str, optional
        Method to use as approximation of the hessian.
        Should be one of
        - None : identity matrix
        - "sigma" : sigma matrix from Bigoni et al. 2022 inverted by lstsq.
        - "sigma_cg" : sigma matrix from Bigoni et al. 2022 inverted by CG .
        - "hessian" : hessian matrix inverted by lstsq.
    precond_kwargs : dict
        Key word arguments the precond method.
        The default is dict()
    seed : int, optional
        The seed of the random number generator for random initialization.
        Only used if init_method is random or random_linear.
        The default is None.
    Returns
    -------
    G : numpy.ndarray
        Result of the QN algorithm.
        Has shape (n_try, K, m) or (K, m).
    loss : float
        Loss associated to the result.
    """
    K = jac_basis.shape[1]

    if R is None:
        R = np.eye(K)

    if G0 is None:
        assert not (m is None)
        G0 = _initialization(jac_u, jac_basis, m, init_method, n_try, R, seed)

    if G0.ndim == 2:
        G0 = G0[None, :, :]

    l, _, m = G0.shape
    loss = np.inf * np.ones(l)
    G = 0 * G0

    for i in range(l):
        logging.info(f"Minimizing Poincare loss with QN on {l} initializations")
        G[i], loss[i] = _minimize_qn_(
            jac_u, jac_basis, G0[i], R, maxiter, tol, precond_method, precond_kwargs)

    if l == 1:
        loss = loss[0]
        G = G[0]

    return G, loss


def _minimize_pymanopt(jac_u, jac_basis, G0=None, m=None, init_method='active_subspace',
                       n_try=1, R=None, use_precond=False, precond_kwargs={},
                       optimizer_kwargs={}, ls_kwargs={}, seed=None):
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
        If not None, ignores init_method, m, n_try and R.
        Has shape (n_try, K, m) or (K, m).
        The default is None.
    m : int, optional
        Only used if G0 is None.
        The default is None.
    init_method : string, optional
        Only used if G0 is None.
        Initialization method, must be one of
        'random', 'random_linear', 'surrogate', 'surrogate_greedy', 'active_subspace'.
        Note that 'active_subspace' assumes that the first d basis functions are linear.
        The default is 'active_subspace'.
    n_try : int, optional
        Number of initial points to compute
        Only used if G0 is None and init_method is 'random' or 'random_linear'.
        The default is 1.
    R : numpy.ndarray, optional
        Matrix wrt which G should be orthonormal.
        Only used if init_method is one of 'surrogate' or 'surrogate_greedy'.
        The default is None, corresponding to identity matrix.
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
    seed : int, optional
        The seed of the random number generator for random initialization.
        Only used if init_method is random or random_linear.
        The default is None.
    Returns
    -------
    G : numpy.ndarray
        Minimizers for each initial point.
        Has shape (n_try, K, m) or (K, m).
    loss : numpy.ndarray
        Minimal costs for initial point.
    optim_results : OptimizerResult or list of OptimizerResult
        The detailed optimization results
    """
    _, K, d = jac_basis.shape

    if G0 is None:
        G0 = _initialization(jac_u, jac_basis, m, init_method, n_try, R, seed)

    if G0.ndim == 1:
        G0 = G0[None, :, None]

    elif G0.ndim == 2:
        G0 = G0[None, :, :]

    l, _, m = G0.shape
    problem, optimizer = _build_pymanopt_problem(
        jac_u, jac_basis, m, use_precond, optimizer_kwargs, precond_kwargs, ls_kwargs)
    loss = np.inf * np.ones(l)
    G = 0 * G0

    optim_results = []
    for i in range(l):
        logging.info(f"Minimizing Poincare loss with pymanopt on {l} initializations")
        res = optimizer.run(problem, initial_point=G0[i])
        optim_results.append(res)
        loss[i] = res.cost
        G[i] = res.point

    if l == 1:
        optim_results = optim_results[0]
        loss = loss[0]
        G = G[0]

    return G, loss, optim_results


def _build_pymanopt_problem(jac_u, jac_basis, m, use_precond=False, optimizer_kwargs={},
                            precond_kwargs={}, ls_kwargs={}):
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
        If True, use the precond from the quasi Newton algorithm escribed in Bigoni et al. 2022,
        meaning taking S(G) as approximate Hessian.
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
    def cost(G):
        return poincare_loss_vector_space(G, jac_u, jac_basis)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(G):
        return poincare_loss_vector_space_gradient(G, jac_u, jac_basis)

    if use_precond:
        if K * m <= 2000:  # TODO: parse as argument
            def precond(G, x):
                S = _eval_SG_full(G, jac_u, jac_basis)
                out, _, _, _ = np.linalg.lstsq(S, x.reshape(-1, order='F'))
                return out.reshape((K, m), order='F')
        else:
            def precond(G, x):
                return _eval_SGinv_X(G, x, jac_u, jac_basis, None, precond_kwargs)

    else:
        precond = None

    problem = pymanopt.Problem(
        manifold, cost, euclidean_gradient=euclidean_gradient, preconditioner=precond)
    line_search = pymanopt.optimizers.line_search.BackTrackingLineSearcher(**ls_kwargs)
    optimizer = pymanopt.optimizers.conjugate_gradient.ConjugateGradient(
        **optimizer_kwargs, line_searcher=line_search)

    return problem, optimizer


def _minimize_surrogate(jac_u, jac_basis, G0=None, R=None, m=1):
    """
    Compute the minimizer of the surrogate proposed in Nouy and Pasco 2025.

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
    if not (G0 is None):
        M = G0.T @ R @ G0
        if np.linalg.norm(M - np.eye(M.shape[0])) > 1e-6:
            G0 = G0 @ np.linalg.inv(np.linalg.cholesky(M).T)

    A, B, C = _eval_surrogate_matrices(jac_u, jac_basis, G0, R)
    eigvals, eigvec = scipy.linalg.eigh(B - A + C, R)
    G = eigvec[:, :m]
    surrogate = eigvals.min()

    if not (G0 is None):
        G = np.hstack([G0, G])

    loss = poincare_loss_vector_space(G, jac_u, jac_basis)

    return G, loss, surrogate


def _minimize_surrogate_greedy(jac_u, jac_basis, m_max, R=None, optimize_poincare=False, tol=1e-7,
                               pmo_kwargs={}):
    """
    Greedy algorithm to learn multiple features, as proposed in
    Nouy and Pasco 2025.

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
        The default is False.
    tol : float, optional
        The greedy algorithm stops if the Poincare loss is smaller.
        The default is 1e-7
    pmo_kwargs
        Key word arguments for the minimization algorithm with pymanopt.
        For more details see _minimize_pymanopt.

    Returns
    -------
    G : numpy.ndarray
        Coefficients in the basis of feature maps.
        Has shape (K, m)
    losses_optimized : numpy.ndarray
        Poincare losses minimized starting from the minimizers of
        the surrogates, at each iterations.
        Has shape (m, )
    losses : numpy.ndarray
        Poincare losses of the minimizers of the surrogates,
        at each iterations
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

    logging.info(f"Starting surrogate greedy iterations with optimize_poincare={optimize_poincare}")
    logging.info(f"Greedy iteration {1}")

    # Learn first feature from surrogate
    G, losses[0], surrogates[0] = _minimize_surrogate(jac_u, jac_basis, None, R)

    # Run minimization of Poincare loss if necessary
    if optimize_poincare:
        G, losses_optimized[0], _ = _minimize_pymanopt(jac_u, jac_basis, G, **pmo_kwargs)
        if not (R is None):
            G = G @ np.linalg.inv(np.linalg.cholesky(G.T @ R @ G).T)
    else:
        losses_optimized[0] = losses[0]

    logging.info(f"Surrogate   : {surrogates[0]:.3e}")
    logging.info(f"Loss        : {losses[0]:.3e}")
    logging.info(f"Loss optim  : {losses_optimized[0]:.3e}")

    j = 1

    while j < m_max and losses_optimized[j - 1] > tol:

        logging.info(f"Greedy iteration {j + 1}")

        # Learn the j-th feature from surrogate and previous features
        G, losses[j], surrogates[j] = _minimize_surrogate(jac_u, jac_basis, G, R)

        # Run minimization of Poincare loss on all features if necessary
        if optimize_poincare:
            G, losses_optimized[j], _ = _minimize_pymanopt(jac_u, jac_basis, G, **pmo_kwargs)
            if not (R is None):
                G = G @ np.linalg.inv(np.linalg.cholesky(G.T @ R @ G).T)
        else:
            losses_optimized[j] = losses[j]

        logging.info(f"Surrogate   : {surrogates[j]:.3e}")
        logging.info(f"Loss        : {losses[j]:.3e}")
        logging.info(f"Loss optim  : {losses_optimized[j]:.3e}")

        j += 1

    # shorten the results if greedy stopped early
    losses = losses[:j]
    losses_optimized = losses_optimized[:j]
    surrogates = surrogates[:j]

    return G, losses_optimized, losses, surrogates
