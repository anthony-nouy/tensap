
import numpy as np
from tensap.poincare_learning.utils._loss_vector_space import poincare_loss_vector_space, \
    poincare_loss_vector_space_gradient, _eval_SG_X, _eval_HG_X, _eval_SG_diag, _eval_SGinv_X, \
    _eval_SG_full, _eval_SG_HG_full, _eval_HessG_X, _eval_HessG_diag, _eval_HessG_full, \
    poincare_loss_surrogate_vector_space, _eval_surrogate_matrices
from tensap.poincare_learning.poincare_loss_vector_space_learning import \
    _minimize_active_subspace, _minimize_qn, _minimize_pymanopt, _minimize_surrogate, \
    _minimize_surrogate_greedy
from tensap.approximation.bases.sub_functional_basis import SubFunctionalBasis
from copy import deepcopy


class PoincareLossVectorSpace:
    """
    Class PoincareLossVectorSpace.
    Class implementing the Poincare based loss from Bigoni et al. 2022
    for the specific case of nonlinear features from a finite dimensional
    vector space of nonlinear functions spanned by some basis.
    In other words g = G.T @ basis.
    The class also implement the surrogates from Nouy and Pasco 2025 for
    one or multiple features.

    Attributes
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d) or (N, d).
    jac_basis : numpy.ndarray
        Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, K, d).
    basis : instance of class with eval method
        Object such that g(x) = G.T @ basis.eval(x)
    R : numpy.ndarray, optional
        The inner product matrix wrt which the coefficient matrix G
        should be orthonormal, i.e G.T @ R @ G = Im
        The default is None, corresponding to identity matrix.
    """

    def __init__(self, jac_u, jac_basis, basis=None, R=None):
        assert jac_u.ndim > 1
        assert jac_u.shape[0] == jac_basis.shape[0]
        assert jac_u.shape[-1] == jac_basis.shape[-1]

        self.jac_u = jac_u
        self.jac_basis = jac_basis
        self.basis = basis
        self.R = R

        if jac_u.ndim == 2:
            self.jac_u = self.jac_u[:, None, :]

    def eval(self, G, jac_g=None):
        return poincare_loss_vector_space(G, self.jac_u, self.jac_basis, jac_g)

    def eval_gradient(self, G, jac_g=None):
        return poincare_loss_vector_space_gradient(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SG_X(self, G, X, jac_g=None):
        return _eval_SG_X(G, X, self.jac_u, self.jac_basis, jac_g)

    def eval_HG_X(self, G, X, jac_g=None):
        return _eval_HG_X(G, X, self.jac_u, self.jac_basis, jac_g)

    def eval_SG_diag(self, G, jac_g=None):
        return _eval_SG_diag(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SG_full(self, G, jac_g=None):
        return _eval_SG_full(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SG_HG_full(self, G, jac_g=None):
        return _eval_SG_HG_full(G, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_X(self, G, X, jac_g=None):
        return _eval_HessG_X(G, X, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_diag(self, G, jac_g=None):
        return _eval_HessG_diag(G, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_full(self, G, jac_g=None):
        return _eval_HessG_full(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SGinv_X(self, G, X, jac_g=None, cg_kwargs={}):
        return _eval_SGinv_X(G, X, self.jac_u, self.jac_basis, jac_g, cg_kwargs)

    def eval_surrogate(self, G, jac_g=None, G0=None, jac_g0=None):
        return poincare_loss_surrogate_vector_space(
            G, self.jac_u, self.jac_basis, G0, jac_g, jac_g0)

    def eval_surrogate_matrices(self, G0=None):
        return _eval_surrogate_matrices(self.jac_u, self.jac_basis, G0, self.R)

    def minimize_active_subspace(self, m=1):
        return _minimize_active_subspace(self.jac_u, self.jac_basis, m)

    def minimize_qn(self, G0=None, m=None, n_try=1, init_method="active_subspace", maxiter=100,
                    tol=1e-5, precond_method='sigma', precond_kwargs={}, seed=None):
        return _minimize_qn(self.jac_u, self.jac_basis, G0, m, n_try, init_method, self.R, maxiter,
                            tol, precond_method, precond_kwargs, seed)

    def minimize_pymanopt(self, G0=None, m=None, init_method='active_subspace', n_try=1,
                          use_precond=True, precond_kwargs={}, optimizer_kwargs={},
                          ls_kwargs={}, seed=None):
        return _minimize_pymanopt(self.jac_u, self.jac_basis, G0, m, init_method, n_try, self.R,
                                  use_precond, precond_kwargs, optimizer_kwargs, ls_kwargs, seed)

    def minimize_surrogate(self, G0=None, m=1):
        return _minimize_surrogate(self.jac_u, self.jac_basis, G0, self.R, m)

    def minimize_surrogate_greedy(self, m_max, optimize_poincare=False, tol=1e-7, pmo_kwargs={}):
        return _minimize_surrogate_greedy(
            self.jac_u, self.jac_basis, m_max, self.R, optimize_poincare, tol, pmo_kwargs)

    def project_basis(self, sub_basis):
        """
        Create a new instance of same type, but where the sub basis of
        the original basis. This has impact on the attributes `basis`,
        `jac_basis` and `R`.

        Parameters
        ----------
        sub_basis : numpy.ndarray
            Coeffiein
            Has shape (K, p).

        Returns
        -------
        projected_loss : type(self)
            Array of shape (K, p), where K is the number of elements in
            `self.basis`, which defines a set of p basis functions in the
            space generated by `self.basis`.

        """
        projected_loss = deepcopy(self)

        if not (self.basis is None):
            projected_loss.basis = SubFunctionalBasis(self.basis, sub_basis)

        if not (self.R is None):
            projected_loss.R = sub_basis.T @ self.R @ sub_basis

        projected_loss.jac_basis = np.einsum('kil,ij->kjl', self.jac_basis, sub_basis)

        return projected_loss


class PoincareLossVectorSpaceTruncated(PoincareLossVectorSpace):
    """
    Class PoincareLossVectorSpaceTruncated.
    This subclass of PoincareLossVectorSpace, which just includes
    an additional projection step of jac_u[k,:,:] onto its dominant singular modes.
    The existing methods of PoincareLossVectorSpace are then applied on these projected jacobians.

    Attributes
    ----------
    jac_u : numpy.ndarray
        Samples of the jacobian of the function to approximate.
        jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
        Has shape (N, n, d) or (N, d).
    jac_basis : numpy.ndarray
        Samples of the jacobian of the basis spanning the space of feature maps.
        jac_basis[k,i,j] is dbasis_i / dx_j evaluated at the k-th sample.
        Has shape (N, K, d).
    basis : instance of class with eval method
        Object such that g(x) = G.T @ basis.eval(x)
    R : numpy.ndarray, optional
        The inner product matrix wrt which the coefficient matrix G
        should be orthonormal, i.e G.T @ R @ G = Im
        The default is None, corresponding to identity matrix.
    m : int, optional
        The number of dominant modes to keep.
        If m = d, then there is no difference with the PoincareLossVectorSpace class.
        The default is None, corresponding to m = d.
    """

    def __init__(self, jac_u, jac_basis, basis=None, R=None, m=None):

        assert jac_u.ndim > 1
        assert jac_u.shape[0] == jac_basis.shape[0]
        assert jac_u.shape[-1] == jac_basis.shape[-1]

        self.jac_u_full = jac_u

        if jac_u.ndim == 2:
            jac_u = jac_u[:, None, :]

        if m is None:
            m = jac_u.shape[2]

        self.m = m

        if m < jac_u.shape[1]:
            jac_u_truncated = self._truncate(m)

        else:
            jac_u_truncated = jac_u.copy()

        super().__init__(jac_u_truncated, jac_basis, basis, R)

    def truncate(self, m):

        if m != self.m:
            self.jac_u = self._truncate(m)
            self.m = m

    def _truncate(self, m):

        jac_u = self.jac_u_full
        jac_u_truncated = 0 * jac_u

        for i, ju in enumerate(jac_u):
            V = np.linalg.svd(ju.T)[0][:, :m]
            jac_u_truncated[i] = ju @ V @ V.T

        return jac_u_truncated
