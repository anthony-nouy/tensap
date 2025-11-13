

import numpy as np
import scipy
from tensap.poincare_learning.utils._loss import poincare_loss, poincare_loss_surrogate
from tensap.approximation.bases.functional_bases import FunctionalBases
from tensap.approximation.bases.polynomial_functional_basis import PolynomialFunctionalBasis
from tensap.approximation.bases.sparse_tensor_product_functional_basis import \
    SparseTensorProductFunctionalBasis
from tensap.tools.multi_indices import MultiIndices


def _build_ortho_poly_basis(X, p, m):
    """
    Build an orthonormal polynomial basis with bounded p_norm of the degree,
    where the constant polynomial has been removed.

    Parameters
    ----------
    X : RandomVector
        The random vector wrt which the basis should be orthonormal.
    p : float or numpy.inf
        The p-norm used for the degrees, either a positive real scalar or numpy.inf.
    m : float
        The bound of the norm, a positive real scalar.

    Returns
    -------
    basis : SparseTensorProductFunctionalBasis
        The orthonormal basis.

    """
    bases = FunctionalBases([
        PolynomialFunctionalBasis(poly, range(m + 1))
        for poly in X.orthonormal_polynomials()
    ])
    I0 = MultiIndices.with_bounded_norm(X.ndim(), p, m)
    I0 = I0.remove_indices(0)
    basis = SparseTensorProductFunctionalBasis(bases, I0)
    return basis


def _eval_jac_g(G, jac_basis):
    """
    Compute evaluations of jac_g from evaluation of jac_basis, where g = G.T @ basis.

    Parameters
    ----------
    G : numpy.ndarray
        The coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).
    jac_basis : numpy.ndarray
        N Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).

    Returns
    -------
    jac_g : numpy.ndarray
        Evaluations of jac_g.
        Has shape (N, m, d)

    """
    K = jac_basis.shape[1]
    if G.ndim == 1:
        Gmat = G.reshape((K, G.shape[0] // K), order='F')  # column-major ordering
    else:
        Gmat = G
    jac_g = np.einsum("ji, kjl", Gmat, jac_basis)
    jac_g = np.moveaxis(jac_g, 0, 1)
    return jac_g


def poincare_loss_vector_space(G, jac_u, jac_basis, jac_g=None):
    """
    Evaluate the Poincare based loss for a feature map from a vector
    space of nonlinear feature maps, as described in Bigoni et al. 2022.
    The feature map writes G.T @ basis.

    Parameters
    ----------
    G : numpy.ndarray
        The coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        N Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        N Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map,
        whose quality is assessed by the Poincare based loss.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided,
        it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    out : float
        Poincare based loss.
    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    out = poincare_loss(jac_u, jac_g)
    return out


def poincare_loss_vector_space_gradient(G, jac_u, jac_basis, jac_g=None):
    """
    Compute the gradient of the Poincare-based loss at the feature map
    of G.T @ basis with respect to G.
    The formula used is a bit different from Bigoni et al. 2022

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    gradient : numpy.ndarray
        The gradient of the loss.
        Has same shame as G.

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    N, m, d = jac_g.shape
    K = jac_basis.shape[1]
    gradients = np.zeros((N, K, m))
    for k, jb, jg, ju in zip(np.arange(N), jac_basis, jac_g, jac_u):
        ug, sg, vgh = np.linalg.svd(jg)
        ug, vgh = ug[:, :m], vgh[:m, :]
        res1 = jb @ (np.eye(d) - vgh.T @ vgh) @ ju.T
        res2 = ug @ np.diag(1 / sg) @ vgh @ ju.T
        gradients[k] = - 2 * res1 @ res2.T
    gradient = gradients.mean(axis=0)
    gradient = gradient.reshape(G.shape, order="F")
    return gradient


def _eval_SG_X(G, X, jac_u, jac_basis, jac_g=None):
    """
    Compute the matrix-vector multiplication Sigma(G)X as described in
    Bigoni et al. 2022, where X is a column-major vectorized matrix of
    same size as G. In other words, it computes Sigma(G) @ X.flatten('F').

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d) or (N, d)
    jac_basis : numpy.ndarray
        Has shape (N, K, d)
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    Sx : numpy.ndarray
        Has same shape as X.

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    Xmat = X.reshape((K, m), order='F')  # column-major ordering
    Sx = np.zeros(Xmat.shape)
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        jgju = jg @ ju.T
        GBG = jg @ jg.T
        GAG = jgju @ jgju.T
        Y = np.linalg.solve(GBG.T, Xmat.T)
        Sx += jb @ jb.T @ np.linalg.solve(GBG.T, GAG.T @ Y).T / N
    Sx = Sx.reshape(X.shape, order='F')  # column-major ordering
    return Sx


def _eval_HG_X(G, X, jac_u, jac_basis, jac_g=None):
    """
    Compute the matrix-vector multiplication H(G)X as described in
    Bigoni et al. 2022, where
    X is a column-major vectorized matrix of same size as G.
    In other words, it computes H(G) @ X.flatten('F').

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    Hx : numpy.ndarray
        Has same shape as X.

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    Xmat = X.reshape((K, m), order='F')  # column-major ordering
    Hx = np.zeros(Xmat.shape)
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        jbju = jb @ ju.T
        GBG = jg @ jg.T
        Y = np.linalg.solve(GBG.T, Xmat.T)
        Hx += jbju @ jbju.T @ Y.T / N
    Hx = Hx.reshape(X.shape, order='F')  # column-major ordering
    return Hx


def _eval_SG_diag(G, jac_u, jac_basis, jac_g=None):
    """
    Compute the diagnoal of the matrix Sigma(G) from Bigoni et al. 2022.
    It is used for preconditioning the conjugate gradient used to apply
    the inverse of Sigma(G).

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    diag : numpy.ndarray
        Has shape (K*m, ).

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    diag = np.zeros(K * m)
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        jgju = jg @ ju.T
        GBG = jg @ jg.T
        GAG = jgju @ jgju.T
        M = np.linalg.solve(GBG.T, GAG.T).T
        M = np.linalg.solve(GBG, M)
        diag1 = np.diag(M)
        diag2 = np.linalg.norm(jb, axis=1)
        diag += np.kron(diag1, diag2) / N
    return diag


def _eval_SGinv_X(G, X, jac_u, jac_basis, jac_g=None, cg_kwargs={}):
    """
    Apply the inverse of the matrix Sigma(G) from Bigoni et al. 2022
    to a vector X.
    The conjugate gradient method is used with Jacobi preconditioning.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).
    cg_kwargs : dict
        Key word arguments for scipy.sparse.linalg.cg.

    Returns
    -------
    out : numpy.ndarray
        Has same shape as X.

    """
    N, K, d = jac_basis.shape
    Km = np.prod(G.shape)
    Xvec = X.reshape(Km, order='F')

    def matvec(Y):
        return _eval_SG_X(G, Y, jac_u, jac_basis, jac_g)

    sigma = scipy.sparse.linalg.LinearOperator((Km, Km), matvec=matvec)
    diag = _eval_SG_diag(G, jac_u, jac_basis)
    M = scipy.sparse.diags(1 / diag)
    out, info = scipy.sparse.linalg.cg(sigma, Xvec, M=M, **cg_kwargs)
    out = out.reshape(X.shape, order='F')
    return out


def _eval_SG_full(G, jac_u, jac_basis, jac_g=None):
    """
    Build the full matrix Sigma(G) from Bigoni et al. 2022.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    S : numpy.ndarray
        Has shape (K*m, K*m).

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    S = np.zeros((K * m, K * m))
    for ju, jb, jg in zip(jac_u, jac_basis, jac_g):
        B = jb @ jb.T
        jgju = jg @ ju.T
        GBG = jg @ jg.T
        GAG = jgju @ jgju.T
        GBG_inv = np.linalg.inv(GBG)
        S += np.kron(GBG_inv @ GAG @ GBG_inv, B) / N
    return S


def _eval_SG_HG_full(G, jac_u, jac_basis, jac_g=None):
    """
    Build the full matrices Sigma(G) and H(G) from Bigoni et al. 2022.
    This should only be used for debugging or testing purpose.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    S : numpy.ndarray
        Has shape (K*m, K*m).
    H : numpy.ndarray
        Has shape (K*m, K*m).

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    S = np.zeros((K * m, K * m))
    H = np.zeros((K * m, K * m))
    for ju, jb, jg in zip(jac_u, jac_basis, jac_g):
        jbju = jb @ ju.T
        A = jbju @ jbju.T
        B = jb @ jb.T
        jgju = jg @ ju.T
        GBG = jg @ jg.T
        GAG = jgju @ jgju.T
        GBG_inv = np.linalg.inv(GBG)
        S += np.kron(GBG_inv @ GAG @ GBG_inv, B) / N
        H += np.kron(GBG_inv, A) / N
    return S, H


def _eval_HessG_X(G, X, jac_u, jac_basis, jac_g=None):
    """
    Compute the matrix-vector multiplication Hess(G) @ X where
    X is a column-major vectorized matrix of same size as G.
    In other words, it computes Hess(G) @ X.flatten('F').

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) of (K*m, ).
    X : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray.
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    hess : numpy.ndarray
        Has same shape as X.

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    if X.ndim == 1:
        Xmat = X.reshape((K, m), order='F')  # column-major ordering
    else:
        Xmat = X
    hess = 0 * Xmat
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        grad_u = ju.T
        djg = Xmat.T @ jb
        jg_inv = scipy.linalg.pinv(jg)
        p = np.eye(d) - jg_inv @ jg
        res1 = 2 * jg_inv.T @ (djg.T @ jg_inv.T - jg_inv @ djg @ p) @ grad_u @ grad_u.T @ p
        res2 = 2 * jg_inv.T @ grad_u @ grad_u.T @ (jg_inv @ djg @ p + p @ djg.T @ jg_inv.T)
        hess += jb @ (res1 + res2).T / N

    hess = hess.reshape(X.shape, order='F')  # column-major ordering
    return hess


def _eval_HessG_diag(G, jac_u, jac_basis, jac_g=None):
    """
    Compute the diagonal of the matrix Hess(G).
    It is used for preconditioning the conjugate gradient used to
    apply the inverse of Hess(G).

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    diag : numpy.ndarray
        Has shape (K*m, ).

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    diag_mat = np.zeros((K, m))
    for jb, ju, jg in zip(jac_basis, jac_u, jac_g):
        grad_u = ju.T
        jg_inv = scipy.linalg.pinv(jg)
        p = np.eye(jb.shape[1]) - jg_inv @ jg
        d1 = (jb @ p @ grad_u @ grad_u.T @ jg_inv) * (jb @ jg_inv)
        d2 = (jb @ p @ grad_u)**2 @ np.ones((ju.shape[0], 1)) @ np.diag(jg_inv.T @ jg_inv)
        d3 = (jb @ jg_inv) * (jb @ p @ grad_u @ grad_u.T @ jg_inv)
        d4 = (jb @ p)**2 @ np.ones((d, 1)) @ np.diag(jg_inv.T @ grad_u @ grad_u.T @ jg_inv)
        diag_mat += d1 - d2.reshape(1, -1) + d3 + d4.reshape(1, -1)
    diag_mat = 2 * diag_mat / N
    diag = diag_mat.flatten(order="F")
    return diag


def _eval_HessG_full(G, jac_u, jac_basis, jac_g=None):
    """
    Build the full matrices Hess(G).
    This should only be used for debugging or testing purpose.

    Parameters
    ----------
    G : numpy.ndarray
        Has shape (K, m) or (K, m).
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    Hess : numpy.ndarray
        Has shape (K*m, K*m).

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    K = jac_basis.shape[1]
    N, m, d = jac_g.shape
    hess = np.zeros((K * m, K * m))
    for i in range(K * m):
        X = np.zeros(K * m)
        X[i] = 1
        hess[i, :] = _eval_HessG_X(G, X, jac_u, jac_basis, jac_g)
    return hess


def poincare_loss_surrogate_vector_space(G, jac_u, jac_basis, G0=None, jac_g=None, jac_g0=None):
    """

    Evaluate the covex surrogate to the Poincare based loss for a feature map from a vector space
    of nonlinear feature maps, from samples. The feature map writes G.T @ basis.
    For more details see Nouy and Pasco 2025.

    Parameters
    ----------
    G : numpy.ndarray
        The coefficients of the feature map in the basis.
        Has shape (K, m) or (K*m, ).
    jac_u : numpy.ndarray
        N Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        N Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, K, d).
    G0 : numpy.ndarray
        The coefficients of fixed feature maps in the basis.
        Has shape (K, m) or (K*m, ).
    jac_g0 : numpy.ndarray, optional
        Samples of the jacobian of the fixed feature maps.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample. If not provided, it is computed
        from G0 and jac_basis.
        Has shape (N, m, d).
    jac_g : numpy.ndarray, optional
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        If not provided, it is computed from G and jac_basis.
        Has shape (N, m, d).

    Returns
    -------
    out : float
        Surrogate to Poincare based loss.

    """
    if jac_g is None:
        jac_g = _eval_jac_g(G, jac_basis)
    if jac_g0 is None and not (G0 is None):
        jac_g0 = _eval_jac_g(G0, jac_basis)
    out = poincare_loss_surrogate(jac_u, jac_g, jac_g0)
    return out


def _eval_surrogate_matrices(jac_u, jac_basis, G0=None, R=None):
    """
    Build the matrices for to the convex surrogate to Poincare loss.
    Handles both the one feature and the multiple features cases where
    G0 is the coefficients matrix of the fixed features.
    Three matrices computed A, B and C are such that H = B-A with H
    from Nouy and Pasco 2025, and such that the largest generalized eigen value of B-A+C
    with respect to R has G0 as eigen vectors, while its smallest eigen value is the same
    as H with same eigen vector.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Has shape (N, n, d).
    jac_basis : numpy.ndarray
        Has shape (N, K, d).
    G0 : numpy.ndarray, optional
        The coefficients of the fixed j features maps.
        If None, it corresponds to the one feature setting.
        Has shap (K, j).
        The default is None.
    R : numpy.ndarray, optional
        The inner product matrix with respect to which G0 is orthonormal.
        The default is None.

    Returns
    -------
    A : numpy.ndarray
        Has shape (K, K).

    B : numpy.ndarray
        Has shape (K, K).

    C : numpy.ndarray
        Has shape (K, K).

    """
    K, d = jac_basis.shape[1:]
    A = np.zeros((K, K))
    B = np.zeros((K, K))
    C = np.zeros((K, K))
    P_g0 = np.zeros((d, d))

    if R is None:
        R = np.eye(K)

    for jb, ju in zip(jac_basis, jac_u):

        if not (G0 is None):
            jg0 = G0.T @ jb
            P_g0 = jg0.T @ scipy.linalg.pinv(jg0.T)

        P_g0_perp = np.eye(d) - P_g0
        v0, s0, _ = np.linalg.svd(P_g0_perp @ ju.T, full_matrices=False)
        v0 = v0[:, s0 > 1e-12 * np.linalg.norm(s0)]  # truncate near zero singular values
        v1 = jb @ v0
        Ax = v1 @ v1.T
        Bx = jb @ P_g0_perp @ jb.T

        w = s0.max()**2
        A += w * Ax / jac_u.shape[0]
        B += w * Bx / jac_u.shape[0]

    if not (G0 is None):
        R12inv = np.linalg.inv(np.linalg.cholesky(R))
        Minv = np.linalg.inv(G0.T @ R @ G0)
        C = R @ G0 @ Minv @ G0.T @ R
        C = C * np.linalg.norm(R12inv @ B @ R12inv.T, 2)

    return A, B, C
