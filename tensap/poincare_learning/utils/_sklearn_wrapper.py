

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.kernel_ridge import KernelRidge
from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import build_benchmark_torch
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace


class PoincareEstimator(RegressorMixin, BaseEstimator):
    """
    Class PoincareEstimator. Wraps a Poincare Inequality based approximation
    method into a sklearn estimator, which allows to use sklearns learning
    algorithms such as GridSearchCV.
    The approximation method is of the form f(g(x)) with g the so-called 
    feature map, and f the so-called profile or regression function.
    Here g have the form g(x) = G^T Phi(x) where Phi:R^d -> R^K is an orthonormal
    polynomial basis wrt the L2 inner product, whose degree multi-indices have
    bounded p_norm. G has shape (K, m) and is orthonormal, ie G^T G = I_m. 
    f is a Kernel Ridge Regression function.

    Attributes
    ----------
    p_norm : float
        The p-norm used for the degrees, either a positive real scalar or numpy.inf.
    max_p_norm : float
        The bound of the norm, a positive real scalar.
    innerp : string
        The orthogonality condition wrt which the matrix G should be orthonormal
        during the learning procedure. Note that G is re-orthonormalized wrt
        euclidean inner product after learning, in order to stabilize the regression f.
        Must be one of 'l2' or 'h1_0'.
    alpha : float
        Parameter in the kernel ridge regression. For more details see sklearn documentation
    gamma : float
        Parameter in the kernel ridge regression. For more details see sklearn documentation
    benchmark_name : string
        Name of the benchmark considered. It is used to obtained the random vector X
        wrt which the basis Phi should be orthonormal.
    benchmark_kwargs : dict, optional
        Additional key word arguments for the benchmark, such as input dim, etc.
        The default is {}
    fit_method : string, optional
        The method used to learn the feature map, ie to learn G.
        Can be one of 'pymanopt', 'qn', 'surrogate' or 'surrogate_greedy'.
        The default is 'pymanopt'.
    fit_parameters : dict, optional
        Parameters fot the learning of the feature map, parsed into the learning method 
        determined by fit_method.
        The default is {}.

    """

    def __init__(self, p_norm, max_p_norm, innerp, alpha, gamma, benchmark_name, benchmark_kwargs={}, fit_method='pymanopt', fit_parameters={}):
        
        self.p_norm = p_norm  
        self.max_p_norm = max_p_norm
        self.innerp = innerp
        self.alpha = alpha
        self.gamma = gamma
        self.benchmark_name = benchmark_name
        self.benchmark_kwargs = benchmark_kwargs
        self.fit_method = fit_method
        self.fit_parameters = fit_parameters

        self.G = None

        # get the random vector input associated to the benchmark
        _, _, rand_vec = build_benchmark_torch(benchmark_name, **benchmark_kwargs)
        self.rand_vec = rand_vec

        # build orthonormal basis
        self.basis = _build_ortho_poly_basis(rand_vec, p_norm, max_p_norm)

        # initialise kernel ridge reressor
        self.regressor = KernelRidge(kernel="rbf", alpha=alpha, gamma=gamma)

        # Compute the inner pdocut matrix if necessary
        if self.innerp == "l2":
            R = None
        elif self.innerp == "h1_0":
            R = self.basis.gram_matrix_h1_0()
        else:
            print("Warning : orthogonality condition for G not valid, identity matrix is taken by default.")
            R = None
        
        self.R = R


    def fit(self, X, y, jac):
        """
        Fit the coefficients G of the feature map in the fixed basis and the regression function.

        Parameters
        ----------
        X : numpy.ndarray
            Sample points.
            Has shape (N, d).
        y : numpy.ndarray
            Evaluations of u at sample points.
            Has shape (N, n) of (N, n).
        jac : numpy.ndarray
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
            jac, self.basis.eval_jacobian(X), self.basis, self.R)
        G = self._fit_features(ploss)

        # Re-orthonormalize G wrt euclidean inner product
        G = np.linalg.svd(G, full_matrices=False)[0]
        self.G = G

        # learning the kernel ridge regressor on the features
        Z = self.basis.eval(X) @ G
        self.regressor.set_params(alpha=self.alpha, gamma=self.gamma)
        self.regressor.fit(Z, y)
        
        return self


    def predict(self, X):

        Z = self.basis.eval(X) @ self.G
        Y = self.regressor.predict(Z)

        return Y
    
    
    def _fit_features(self, ploss):

        if self.fit_method == 'pymanopt':
            minimizer = ploss.minimize_pymanopt
        elif self.fit_method == 'qn':
            minimizer = ploss.minimize_qn
        elif self.fit_method == 'surrogate':
            minimizer = ploss.minimize_surrogate
        else:
            raise NotImplementedError('Method not implemented')
        
        G, losses = minimizer(**self.fit_parameters)[:2]
        G = G[losses.argmin()]

        return G
    