

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace
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

    def __init__(self, random_vector, p_norm, max_p_norm, innerp, fit_method='pymanopt', fit_parameters={}):
        
        self.random_vector = random_vector
        self.p_norm = p_norm  
        self.max_p_norm = max_p_norm
        self.innerp = innerp
        self.fit_method = fit_method
        self.fit_parameters = fit_parameters

        # build orthonormal basis
        self.basis = _build_ortho_poly_basis(random_vector, p_norm, max_p_norm)
        self.G = np.zeros((self.basis.cardinal(), 1))
        self.g = SubFunctionalBasis(self.basis, self.G)

        # Compute the inner pdocut matrix if necessary
        if self.innerp == "l2":
            R = None
        elif self.innerp == "h1_0":
            R = self.basis.gram_matrix_h1_0()
        else:
            logging.warning("Warning : orthogonality condition for G not valid, identity matrix is taken by default.")
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

        # Learning the feature map by minizming Poincare loss with pymanopt
        ploss = PoincareLossVectorSpace(
            jac_u, self.basis.eval_jacobian(X), self.basis, self.R)
        
        if self.fit_method == 'pymanopt':
            minimizer = ploss.minimize_pymanopt
        elif self.fit_method == 'qn':
            minimizer = ploss.minimize_qn
        elif self.fit_method == 'surrogate':
            minimizer = ploss.minimize_surrogate
        elif self.fit_method == 'surrogate':
            minimizer = ploss.minimize_surrogate
        else:
            raise NotImplementedError('Method not implemented')
            
        G, losses = minimizer(**self.fit_parameters)[:2]

        # Re-orthonormalize G wrt euclidean inner product
        G = np.linalg.svd(G, full_matrices=False)[0]
        self.G = G
        self.g = SubFunctionalBasis(self.basis, G)

        return self
    
    
    def score(self, X, jac_u):
        
        ploss = PoincareLossVectorSpace(
            jac_u, self.basis.eval_jacobian(X), self.basis, self.R)

        return ploss.eval(self.G)

