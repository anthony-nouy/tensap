

from tensap.poincare_learning.utils._loss_vector_space import poincare_loss_vector_space, poincare_loss_vector_space_gradient, _eval_SG_X, _eval_HG_X, _eval_SG_diag, _eval_SGinv_X, _eval_SG_HG_full, _eval_HessG_X, _eval_HessG_diag, _eval_HessG_full, poincare_loss_surrogate_vector_space, _eval_surrogate_matrices
from tensap.poincare_learning.poincare_loss_vector_space_learning import _minimize_qn, _minimize_pymanopt, _minimize_surrogate, _minimize_surrogate_greedy


class PoincareLossVectorSpace:
    """
    Class PoincareLossVectorSpace.
    Class implementing the Poincare based loss from Bigoni et al. 2022
    for the specific case of nonlinear features from a finite dimensional
    vector space of nonlinear functions spanned by some basis.
    In other words g = G.T @ basis.
    The class also implement the surrogates from Nouy et al. 2025 for
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
        assert jac_u.ndim >1
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

    def eval_SG_HG_full(self, G, jac_g=None):
        return _eval_SG_HG_full(G, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_X(self, G, X, jac_g=None):
        return _eval_HessG_X(G, X, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_diag(self, G, jac_g=None):
        return _eval_HessG_diag(G, self.jac_u, self.jac_basis, jac_g)

    def eval_HessG_full(self, G, jac_g=None):
        return _eval_HessG_full(G, self.jac_u, self.jac_basis, jac_g)

    def eval_SGinv_X(self, G, X, jac_g=None, **cg_kwargs):
        return _eval_SGinv_X(G, X, self.jac_u, self.jac_basis, jac_g, **cg_kwargs)

    def eval_surrogate(self, G, jac_g=None, G0=None, jac_g0=None):
        return poincare_loss_surrogate_vector_space(G, self.jac_u, self.jac_basis, G0, jac_g, jac_g0)
    
    def eval_surrogate_matrices(self, G0=None):
        return _eval_surrogate_matrices(self.jac_u, self.jac_basis, G0, self.R)
    
    def minimize_qn(self, G0=None, maxiter_qn=100, tol_qn=1e-5, verbosity=2, **cg_kwargs):
        return _minimize_qn(G0, self.jac_u, self.jac_basis, self.R, maxiter_qn=100, tol_qn=1e-5, verbosity=2, **cg_kwargs)
    
    def minimize_pymanopt(self, G0=None, use_precond=True, 
    precond_kwargs={}, optimizer_kwargs={}, ls_kwargs={}):
        return _minimize_pymanopt(G0, self.jac_u, self.jac_basis, use_precond, precond_kwargs, optimizer_kwargs, ls_kwargs)

    def minimize_surrogate(self, G0=None, m=1):
        return _minimize_surrogate(self.jac_u, self.jac_basis, G0, self.R, m=1)
    
    def minimize_surrogate_greedy(self, m_max, optimize_poincare=True, tol=1e-7, verbose=2, **pmo_kwargs):
        return _minimize_surrogate_greedy(self.jac_u, self.jac_basis, m_max, self.R, optimize_poincare, tol, verbose, **pmo_kwargs)

