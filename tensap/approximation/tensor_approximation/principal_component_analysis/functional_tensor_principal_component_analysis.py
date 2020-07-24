'''
Module functional_tensor_principal_component_analysis.

Copyright (c) 2020, Anthony Nouy, Erwan Grelier
This file is part of tensap (tensor approximation package).

tensap is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

tensap is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with tensap.  If not, see <https://www.gnu.org/licenses/>.

'''

from copy import deepcopy
import numpy as np
import tensap


class FunctionalTensorPrincipalComponentAnalysis:
    '''
    Class FunctionalTensorPrincipalComponentAnalysis: principal component
    analysis of multivariate functions based on
    TensorPrincipalComponentAnalysis for algebraic tensors.

    Attributes
    ----------
    tol : int or float or list or numpy.ndarray
        An array containing the prescribed relative precisions (the length
        depends on the format).
        If len(tol)==1, use the same value for all alpha.
        Set tol = inf to prescribe the rank.
    max_rank : int or list or numpy.ndarray
        An array containing the maximum alpha-ranks (the length depends on the
        format).
        If len(max_rank)==1, use the same value for all alpha.
        Set max_rank = np.inf to prescribe the precision.
    bases : tensap.FunctionalBases
        The functional bases used for the projection of the function.
    grid : tensap.FullTensorGrid
         FullTensorGrid for the projection of the function on the functional
         bases.
    display : bool
        Boolean specifying the verbosity of the methods.
    pca_sampling_factor : int
        A factor to determine the number of samples N for the estimation of
        the principal components (1 by default):
            - if prescribed precision, N = pca_sampling_factor*N_alpha,
            - if prescribed rank, N = pca_sampling_factor*t.
    pca_adaptive_sampling : bool
        Adaptive sampling to determine the principal components with prescribed
        precision.
    projection_type : str
        The type of projection. The default is 'interpolation'.

    '''

    def __init__(self):
        '''
        Constructor for the class FunctionalTensorPrincipalComponentAnalysis.

        Returns
        -------
        None.

        '''
        self.tol = 1e-8
        self.max_rank = np.inf
        self.bases = None
        self.grid = None
        self.display = True
        self.pca_sampling_factor = 1
        self.pca_adaptive_sampling = False
        self.projection_type = 'interpolation'

    def hopca(self, fun):
        '''
        Return the set of alpha-principal components of a tensor, for all alpha
        in {0,1,...,d-1}.

        For prescribed precision, set FPCA.max_rank = np.inf and FPCA.tol to
        the desired precision (possibly an array of length d).

        For prescribed rank, set FPCA.tol = np.inf and FPCA.max_rank to the
        desired rank (possibly an array of length d).

        See also the documentation of the class
        FunctionalTensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : function
            A function of d variables.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        f_pc : list
            List of the alpha-principal components of the function.
        outputs : list
            List containing the outputs of the method
            alpha_principal_components.

        '''
        solver, t_fun, shape, tpca = self._prepare(fun)
        t_pc, outputs = tpca.hopca(t_fun, shape)

        P = solver._projection_operators()
        t_pc = [np.matmul(p, x) for p, x in zip(P, t_pc)]
        f_pc = [tensap.SubFunctionalBasis(b, a) for b, a in
                zip(solver.bases.bases, t_pc)]
        return f_pc, outputs

    def tucker_approximation(self, fun):
        '''
        Approximation of a function of d variables in Tucker format based on a
        Principal Component Analysis.

        For a prescribed precision, set TPCA.max_rank = np.inf and TPCA.tol to
        the desired precision (possibly an array of length d).

        For a prescribed rank, set TPCA.tol = np.inf and TPCA.max_rank to the
        desired rank (possibly an array of length d).

        See also the documentation of the class
        FunctionalTensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : function
            A function of d variables.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        tensap.FunctionalTensor
            A function in tree based format with a trivial tree.
        dict
            Dictionnary containing the outputs of the method.

        '''
        solver, t_fun, shape, tpca = self._prepare(fun)
        tensor, output = tpca.tucker_approximation(t_fun, shape)
        return solver._project(tensor), output

    def tt_approximation(self, fun):
        '''
        Approximation of a function of d variables in tensor train format based
        on a Principal Component Analysis.

        For a prescribed precision, set TPCA.max_rank = np.inf and TPCA.tol to
        the desired precision (possibly an array of length d).

        For a prescribed rank, set TPCA.tol = np.inf and TPCA.max_rank to the
        desired rank (possibly an array of length d).

        See also the documentation of the class
        FunctionalTensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : function
            A function of d variables.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        tensap.FunctionalTensor
            A function in tree based format with a linear tree.
        dict
            Dictionnary containing the outputs of the method.

        '''
        solver, t_fun, shape, tpca = self._prepare(fun)
        tensor, output = tpca.tt_approximation(t_fun, shape)
        return solver._project(tensor), output

    def tree_based_approximation(self, fun, tree, is_active_node=None):
        '''
        Approximation of a function of d variables in tree based tensor format
        based on a Principal Component Analysis.

        For a prescribed precision, set TPCA.max_rank = np.inf and TPCA.tol to
        the desired precision (possibly an array of length d-1).

        For a prescribed rank, set TPCA.tol = np.inf and TPCA.max_rank to the
        desired rank (possibly an array of length d-1).

        See also the documentation of the class
        FunctionalTensorPrincipalComponentAnalysis.

        Parameters
        ----------
        fun : fun or tensap.Function
            Function of d variables i_1, ..., i_d which returns the entries of
            the tensor.
        shape : list or numpy.ndarray
            The shape of the tensor.
        tree : tensap.DimensionTree
            The required dimension tree.
        is_active_node : list or numpy.ndarray, optional
            An array of booleans indicating which nodes of the tree are active.
            The default is None, settings all the nodes active.

        Raises
        ------
        ValueError
            If the provided tolerance and max ranks are not correct.

        Returns
        -------
        tensap.FunctionalTensor
            A function in tree based format.
        dict
            Dictionnary containing the outputs of the method.

        '''
        solver, t_fun, shape, tpca = self._prepare(fun)
        tensor, output = tpca.tree_based_approximation(t_fun, shape, tree,
                                                       is_active_node)
        return solver._project(tensor), output

    def _prepare(self, fun):
        '''
        Prepare the principal component analysis.

        Parameters
        ----------
        fun : function
            A function of d variables.

        Raises
        ------
        ValueError
            If the attribute projection_type is wrong.

        Returns
        -------
        tensap.FunctionalTensorPrincipalComponentAnalysis
            A FunctionalTensorPrincipalComponentAnalysis object with updated
            attributes.
        function
            A function used by the methods of
            tensap.TensorPrincipalComponentAnalysis.
        numpy.ndarray
            The cardinals of the functional bases.
        tensap.TensorPrincipalComponentAnalysis
            A tensap.TensorPrincipalComponentAnalysis used to perform the
            principal component analysis.

        '''
        solver = deepcopy(self)
        solver.bases = tensap.FunctionalBases(self.bases)

        # Create the tensor product grid
        if solver.projection_type == 'interpolation':
            if solver.grid is None:
                solver.grid = solver.bases.interpolation_points()
            else:
                solver.grid = solver.bases.interpolation_points(solver.grid)
        else:
            raise ValueError('Wrong projection_type attribute.')
        solver.grid = tensap.FullTensorGrid(solver.grid)

        # Create the function which provides the values of the function on the
        # grid
        def t_fun(i):
            return fun(solver.grid.eval_at_indices(i))

        shape = solver.bases.cardinals()

        # Create a TensorPrincipalComponentAnalysis with the same values of
        # properties as the FunctionalPrincipalComponentAnalysis
        tpca = tensap.TensorPrincipalComponentAnalysis()
        tpca.display = solver.display
        tpca.pca_sampling_factor = solver.pca_sampling_factor
        tpca.pca_adaptive_sampling = solver.pca_adaptive_sampling
        tpca.tol = solver.tol
        tpca.max_rank = solver.max_rank

        return solver, t_fun, shape, tpca

    def _projection_operators(self):
        '''
        Compute projection operators.

        The matrix P[nu] represents the operator which associates to the values
        of a function of the variable x_nu on a grid the coefficients of the
        projection on the basis of functions of the variable x_nu.

        Raises
        ------
        ValueError
            If the attribute projection_type is wrong.

        Returns
        -------
        P : list
            The projection operators for each dimension.

        '''
        if self.projection_type == 'interpolation':
            P = self.bases.eval(self.grid.grids)
            P = [np.linalg.pinv(x) for x in P]
        else:
            raise ValueError('Wrong projection_type attribute.')
        return P

    def _project(self, t):
        '''
        Compute the projection of a tensor on the functional bases.

        The method takes an AlgebraicTensor t whose entries are the values of
        the function on a product grid, and returns a FunctionalTensor obtained
        by applying the projections obtained by the method
        _projectionOperators.

        Parameters
        ----------
        t : tensap.Tensor
            The tensor used for the projection.

        Returns
        -------
        tensap.FunctionalTensor
            The obtained projection.

        '''
        tensor = deepcopy(t)
        P = self._projection_operators()
        for nu in range(tensor.order):
            alpha = tensor.tree.dim2ind[nu]
            if tensor.is_active_node[alpha-1]:
                data = np.matmul(P[nu], tensor.tensors[alpha-1].data)
                tensor.tensors[alpha-1] = tensap.FullTensor(data, 2, data.shape)
            else:
                pa = tensor.tree.parent(alpha)
                ch = tensor.tree.child_number(alpha)
                tensor.tensors[pa-1] = \
                    tensor.tensors[pa-1].tensor_matrix_product(P[nu], ch-1)
        return tensap.FunctionalTensor(tensor, self.bases)
