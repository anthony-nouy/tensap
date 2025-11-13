

import numpy as np
import scipy


def poincare_loss(jac_u, jac_g):
    """
    Evaluate the Poincare based loss for a general feature map
    from samples. For more details see Bigoni et al. 2022.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d) or (N, d).
    jac_g : numpy.ndarray
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        Has shape (N, m, d).

    Returns
    -------
    out : float
        Poincare loss.

    """
    N = jac_u.shape[0]
    out = 0
    for ju, jg in zip(jac_u, jac_g):
        proj = scipy.linalg.pinv(jg) @ jg
        out += np.linalg.norm(ju.T - proj @ ju.T) ** 2 / N
    return out


def poincare_loss_augmented(jac_u, jac_g, alpha=1):
    """
    Evaluate the Poincare based loss with dimension augmentation, for a
    general feature map using only samples.
    For more details see Verdiere et al. 2023.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d) or (N, d).
    jac_g : numpy.ndarray
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        Has shape (N, m, d).
    alpha : float
        Standard deviation of the noise in the augmented dimensions.
        One can see it as a regularization parameter.
    Returns
    -------
    out : float
        Poincare loss.

    """
    N, d = jac_u.shape[0], jac_u.shape[-1]
    out = np.linalg.norm(jac_u)**2 / N
    for ju, jg in zip(jac_u, jac_g):
        jgju = jg @ ju.T
        gram = alpha**2 * np.eye(d) + jg @ jg.T
        out -= np.linalg.norm(jgju.T @ np.linalg.solve(gram, jgju))**2 / N
    return out


def poincare_loss_surrogate(jac_u, jac_g, jac_g0=None):
    """
    Evaluate the covex surrogate to the Poincare based loss for
    a general feature map from samples.
    For more details see Nouy and Pasco 2025.

    Parameters
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d) or (N, d).
    jac_g : numpy.ndarray
        Samples of the jacobian of the feature map.
        jac_g[k,i,j] is dg_i / dx_j evaluated at the k-th sample.
        Has shape (N, m, d).
    jac_g0 : numpy.ndarray, optional
        Samples of the jacobian of the previously learnt feature maps.
        jac_g0[k,i,j] is dg0_i / dx_j evaluated at the k-th sample.
        Has shape (N, j, d). The default is None.

    Returns
    -------
    out : float
        Poincare loss.

    """
    if jac_u.ndim == 2:
        jac_u = jac_u[:, None, :]
    N, n, d = jac_u.shape
    out = 0
    if jac_g0 is None:
        jac_g0 = 0 * jac_g
    for ju, jg, jg0 in zip(jac_u, jac_g, jac_g0):
        proj_g0 = scipy.linalg.pinv(jg0) @ jg0
        v0 = ju.T - proj_g0 @ ju.T
        w0 = jg.T - proj_g0 @ jg.T
        proj_v0 = v0 @ scipy.linalg.pinv(v0)
        res = np.linalg.norm(w0 - proj_v0 @ w0) ** 2 / N
        c = np.linalg.svd(v0)[1].max()**2
        out += c * res
    return out
