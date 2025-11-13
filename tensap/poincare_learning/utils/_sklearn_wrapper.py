

import numpy as np
from sklearn.base import BaseEstimator
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace, \
    PoincareLossVectorSpaceTruncated
from tensap.approximation.bases.sub_functional_basis import SubFunctionalBasis
import logging


class PolynomialFeatureEstimator(BaseEstimator):
    """
    Class PolynomialFeatureEstimator. Wraps a Poincare Inequality based feature learning
    method into a sklearn estimator, which allows to use sklearns learning
    algorithms such as GridSearchCV.
    The features learnt have the form g(x) = G^T Phi(x) where Phi:R^d -> R^K is an orthonormal
    polynomial basis wrt the L2 inner product, whose degree multi-indices have
    bounded p_norm. G has shape (K, m) and is orthonormal, ie G^T G = I_m.

    Attributes
    ----------
    random_vector : RandomVector
        Instance of RandomVector with respect to which the polynomial basis
    p_norm : float
        The p-norm used for the degrees, either a positive real scalar or numpy.inf.
    max_p_norm : float
        The bound of the norm, a positive real scalar.
    innerp : string
        The orthogonality condition wrt which the matrix G should be orthonormal
        during the learning procedure. Then G is re-orthonormalized wrt
        euclidean inner product after learning, so that G^T Phi(X) is standardised.
        Should be one of 'l2' or 'h1_0'.
    fit_method : string, optional
        The method used to learn the feature map, ie to learn G.
        Can be one of 'pymanopt', 'qn', 'surrogate' or 'surrogate_greedy'.
        The default is 'pymanopt'.
    fit_parameters : dict, optional
        Parameters fot the learning of the feature map, parsed into the learning method
        determined by fit_method.
        The default is {}.

    """

    def __init__(self, random_vector, p_norm, max_p_norm, innerp, fit_method='pymanopt',
                 fit_parameters={}):

        self.random_vector = random_vector
        self.p_norm = p_norm
        self.max_p_norm = max_p_norm
        self.innerp = innerp
        self.fit_method = fit_method
        self.fit_parameters = fit_parameters

        self.__build_basis()
        self.optim_history = {}

    def __build_basis(self):

        # build orthonormal basis
        self.basis = _build_ortho_poly_basis(self.random_vector, self.p_norm, self.max_p_norm)
        self.G = np.zeros((self.basis.cardinal(), 1))
        self.g = SubFunctionalBasis(self.basis, self.G)

        # Compute the inner pdocut matrix if necessary
        if self.innerp == "l2":
            R = None
        elif self.innerp == "h1_0":
            R = self.basis.gram_matrix_h1_0()
        else:
            logging.warning("Orthogonality condition for G not valid, using identity.")
            R = None

        self.R = R

    def fit(self, X, jac_u):
        """
        Fit the coefficients G of the feature map in the fixed basis.

        Parameters
        ----------
        X : numpy.ndarray
            Sample points.
            Has shape (N, d).
        jac_u : numpy.ndarray
            Evaluations of jac_u at sample points.
            jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
            Has shape (N, n, d).
        Returns
        -------
        self : PoincareEstimator
            The fitted estimator.

        """
        self.__build_basis()

        # assume that vector valued always means collective setting
        if jac_u.ndim == 3 and jac_u.shape[1] > 1:
            ploss = PoincareLossVectorSpaceTruncated(
                jac_u, self.basis.eval_jacobian(X), self.basis, self.R,
                self.fit_parameters.get('m'))

        else:
            ploss = PoincareLossVectorSpace(
                jac_u, self.basis.eval_jacobian(X), self.basis, self.R)

        if self.fit_method == 'pymanopt':

            # Non preconditionned search on maybe multiple init
            G, losses, optim_results = ploss.minimize_pymanopt(
                use_precond=True, **self.fit_parameters)

            # if several initial points
            if G.ndim == 3:
                ind = losses.argmin()
                G = G[ind]
                optim_results = optim_results[ind]

            optim_log = optim_results.log.get('iterations')
            self.optim_history['method'] = ['pCG' for i in range(len(optim_log['cost']))]
            self.optim_history['cost'] = optim_log['cost']
            self.optim_history['gradient_norm'] = optim_log['gradient_norm']

        elif self.fit_method == 'surrogate':
            G, losses = ploss.minimize_surrogate(**self.fit_parameters)[:2]

        elif self.fit_method == 'surrogate_greedy':
            G, losses = ploss.minimize_surrogate_greedy(**self.fit_parameters)[:2]

        else:
            raise NotImplementedError('Method not implemented')

        # Re-orthonormalize G wrt euclidean inner product
        G = np.linalg.svd(G, full_matrices=False)[0]
        self.G = G
        self.g = SubFunctionalBasis(self.basis, G)

        return self

    def score(self, X, jac_u):
        """
        Compute the negative Poincare loss from samples.

        Parameters
        ----------
        X : numpy.ndarray
            Sample points.
            Has shape (N, d).
        jac_u : numpy.ndarray
            Evaluations of jac_u at sample points.
            jac_u[k,i,j] is du_i / dx_j evaluated at the k-th sample.
            Has shape (N, n, d).

        Returns
        -------
        out : float
            Negative Poincare loss.

        """
        ploss = PoincareLossVectorSpace(
            jac_u, self.basis.eval_jacobian(X), self.basis, self.R)

        # negative Poincare loss
        out = - ploss.eval(self.G)

        return out
