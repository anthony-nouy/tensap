'''
Module functional_tensor.

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


class FunctionalTensor(tensap.Function):
    '''
    Class FunctionalTensor.

    Attributes
    ----------
    tensor : Tensor or tensap.FunctionalTensor, optional
        The tensor of the FunctionalTensor. The default is None.
    bases : list or tensap.FunctionalBases, optional
        The functional bases of the FunctionalTensor. The default is None.
    fdims : list or numpy.ndarray, optional
        The dimensions corresponding to the bases. The default is None.

    '''

    def __init__(self, tensor=None, bases=None, fdims=None):
        '''
        Constructor for the FunctionalTensor.

        Parameters
        ----------
        tensor : Tensor or tensap.FunctionalTensor, optional
            The tensor of the FunctionalTensor. The default is None.
        bases : list or tensap.FunctionalBases, optional
            The functional bases of the FunctionalTensor. The default is None.
        fdims : list or numpy.ndarray, optional
            The dimensions corresponding to the bases. The default is None.

        Raises
        ------
        ValueError
            If the provided objects are not of the expected types.

        Returns
        -------
        None.

        '''
        tensap.Function.__init__(self)

        if isinstance(tensor, FunctionalTensor):
            tensor = tensor.tensor

        self.tensor = tensor

        if bases is not None and \
                not isinstance(bases, (tensap.FunctionalBases, list)):
            raise ValueError('Must provide a FunctionalBases object, or a ' +
                             'cell of bases evaluations.')
        self.bases = bases
        if fdims is None:
            if tensor.order != len(bases):
                raise ValueError('Bases must contain as many bases as the ' +
                                 'order of the tensor, with possible empty ' +
                                 'elements.')
            fdims = range(tensor.order)
        else:
            if len(fdims) != len(bases):
                raise ValueError('The number of functional dimensions must ' +
                                 'correspond to the number of bases in bases.')
        self.fdims = np.array(fdims)
        if isinstance(bases, list):
            self.evaluated_bases = True
        else:
            self.evaluated_bases = False
            self.measure = self.bases.measure
            self.dim = np.sum(self.bases.ndim())

    def __plus__(self, y):
        if isinstance(y, FunctionalTensor) and self.bases == y.bases:
            tensor = self.tensor + y.tensor
        else:
            raise NotImplementedError('Method not implemented.')
        return FunctionalTensor(tensor, self.bases, self.fdims)

    def is_random(self):
        '''
        Determine if self is random.

        Returns
        -------
        bool
            Boolean indicating if if self is random.

        '''
        return isinstance(self.bases.measure, tensap.ProbabilityMeasure)

    def mean(self, *measure):
        '''
        Compute the expectation of the random variable self(measure) if
        measure is provided, and of self(self.bases.measure) otherwise.

        Parameters
        ----------
        *measure : tensap.RandomVector, optional
            The measure used for the computation of the mean. If not provided,
            indicates to use self.bases.measure.

        Returns
        -------
        float or Tensor
            The mean of the function.

        '''
        bases_eval = self.bases.mean(None, *measure)
        return self.tensor.tensor_vector_product(bases_eval,
                                                 self.fdims).tolist()

    def expectation(self, *measure):
        '''
        Compute the expectation of the random variable self(measure) if
        measure is provided, and of self(self.bases.measure) otherwise.

        Parameters
        ----------
        *measure : tensap.RandomVector, optional
            The measure used for the computation of the expectation. If not
            provided, indicates to use self.bases.measure.

        Returns
        -------
        float or Tensor
            The expectation of the function.

        '''
        return self.mean(*measure)

    def variance(self, *measure):
        '''
        Compute the variance of the random variable self(measure) if
        measure is provided, and of self(self.bases.measure) otherwise.

        Parameters
        ----------
        *measure : tensap.RandomVector, optional
            The measure used for the computation of the variance. If not
            provided, indicates to use self.bases.measure.

        Returns
        -------
        var : float or Tensor
            The variance of the function.

        '''
        mean = self.expectation(*measure)
        if np.isscalar(mean):
            var = self.dot_product_expectation(self, None, *measure) - mean**2
        else:
            raise NotImplementedError('Method not implemented.')
        return var

    def std(self, *measure):
        ''''
        Compute the standard deviation of the random variable self(measure) if
        measure is provided, and of self(self.bases.measure) otherwise.

        Parameters
        ----------
        *measure : tensap.RandomVector, optional
            The measure used for the computation of the standard deviation. If
            not provided, indicates to use self.bases.measure.

        Returns
        -------
        v : float or Tensor
            The standard deviation of the function.

        '''
        return np.sqrt(self.variance(*measure))

    def dot_product_expectation(self, f_2, fdims=None, *measure):
        '''
        Computes the expectation of self(X)f_2(X) where X is the random vector
        associated with self.bases if measure is not provided, and measure
        otherwise.

        For tensor-valued functions of X (len(X)<self.order), fdims specifies
        the dimensions of self and f_2 corresponding to theRandomVector X.

        Parameters
        ----------
        f_2 : tensap.FunctionalTensor
            The second functional tensor of the product.
        fdims : list of numpy.ndarray, optional
            Specifies the dimensions of self and f_2 corresponding to
            theRandomVector X. The default is None, indicating all the
            dimensions
        *measure : tensap.RandomVector, optional
            The measure used for the computation of the product. If not
            provided, indicates to use self.bases.measure.

        Raises
        ------
        ValueError
            If the two tensors do not have the same order and fdims is not
            specified.
        NotImplementedError
            If the bases of self and f_2 are not equal.

        Returns
        -------
        float or Tensor
            The expectation of self(X)f_2(X).

        '''
        if fdims is None:
            if self.tensor.order == f_2.tensor.order:
                fdims = range(self.tensor.order)
            else:
                raise ValueError('Tensors u and v do not have the same ' +
                                 'order, must specify fdims.')
        if self.bases == f_2.bases:
            gram_matrix = self.bases.gram_matrix(fdims, *measure)
        else:
            raise NotImplementedError('Method not implemented.')
        tensor = self.tensor.tensor_matrix_product(gram_matrix, fdims)
        return tensor.dot(f_2.tensor)

    def norm(self, *measure):
        '''
        Return the L^2 norm of self(X), with X = measure if provided, and
        X = self.bases.measure otherwise.

        If self.evaluatedBases is true, without additional information, return
        the canonical norm of self.tensor.

        Parameters
        ----------
        *measure : tensap.RandomVector, optional
            The measure used for the computation of the norm. If not provided,
            indicates to use self.bases.measure.

        Returns
        -------
        float or Tensor
            The L^2 norm of self(X).

        '''
        if not self.evaluated_bases:
            gram_matrix = self.bases.gram_matrix(range(self.tensor.order),
                                                 *measure)
        else:
            gram_matrix = [np.eye(x.shape[1]) for x in self.bases]
        tensor = self.tensor.tensor_matrix_product(gram_matrix,
                                                   range(self.tensor.order))
        return np.sqrt(tensor.dot(self.tensor))

    def conditional_expectation(self, dims, *args):
        '''
        Compute the conditional expectation of self with respect to the random
        variables dims (a subset of [1, ..., d]).

        The expectation with respect to other variables (in the complementary
        set of dims) is taken with respect to the probability measure given by
        a tensap.RandomVector if provided as an additional argument, or with
        respect to the probability measure associated with the corresponding
        bases of self.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The dimensions of the random variables with respect to which the
            conditional expectation is to be computed.
        *args : tuple
            Tuple containing a tensap.randomVector giving the probability
            measure of the variables other than the ones in dims. If not
            provided, the measure is infered from self.bases.measure.

        Returns
        -------
        tensap.FunctionalTensor
            The conditional expectation of self with respect to the random
            variables dims, as a len(dims)-order tensor.

        '''
        dims = np.atleast_1d(dims)
        if np.all([isinstance(x, bool) for x in dims]):
            dims = np.nonzero(dims)[0]

        d = self.tensor.order
        if np.size(dims) == 0:
            return self.expectation(*args)

        dims = np.sort(dims)
        assert np.size(self.fdims) == d and \
            np.array_equal(self.fdims, range(d)), \
            ('Method not implemented for self.fdims different from ' +
             'range(d).')

        dims_C = np.setdiff1d(range(len(self.bases)), dims)
        if dims_C.size == 0:
            return deepcopy(self)

        H = self.bases.mean(dims_C, *args)
        t = self.tensor.tensor_vector_product(H, dims_C)

        bases = self.bases.keep_bases(dims)
        # TODO Take into account the mapping when implemented

        out = FunctionalTensor(t, bases)
        if self.measure is not None:
            out.measure = self.measure.marginal(dims)

        return out

    def variance_conditional_expectation(self, alpha):
        '''
        Compute the variance of the conditional expectation of self in
        dimensions in alpha.

        Parameters
        ----------
        alpha : list or numpy.ndarray
            Array containing the dimensions (either explicitely or using
            booleans) in which the variance of the conditional expectation is
            computed.

        Returns
        -------
        v : numpy.ndarray
            The variance of the conditional expectation of self in
            dimensions in alpha.

        '''
        alpha = np.atleast_2d(alpha)
        m = self.expectation()
        v = np.zeros(alpha.shape[0])
        for i in range(alpha.shape[0]):
            u = alpha[i, :]
            if np.all([isinstance(x, bool) for x in u]):
                u = np.nonzero(u)[0]
            if u.size == 0:
                v[i] = 0
            else:
                mu = self.conditional_expectation(u)
                v[i] = mu.dot_product_expectation(mu) - m**2
        return v

    def eval(self, x, *dims):
        '''
        Evaluate self at the points x.

        If dims is provided, compute the partial evaluations of self at points
        x in dimensions in dims.

        Parameters
        ----------
        x : list or numpy.ndarray or None
            The points at which the function is to be evaluated. If x is None
            and self.evaluated_bases, evaluates the function using the
            evaluations of the bases.
        *dims : list or numpy.ndarray, optional
            The dimensions of the partial evaluation. If not provided,
            evaluate the function in all dimensions.

        Returns
        -------
        numpu.ndarray or Tensor
            The evaluations of self at the points x.

        '''
        if self.evaluated_bases:
            bases_eval = self.bases
        else:
            bases_eval = self.bases.eval(x, *dims)
        return self.eval_with_bases_evals(bases_eval, *dims)

    def __mul__(self, f_2):
        if isinstance(f_2, (FunctionalTensor, tensap.FunctionalTensor)):
            b = self.bases.kron(f_2.bases)
            t = self.tensor.kron(f_2.tensor)
            out = FunctionalTensor(t, b)
            if isinstance(out.tensor, tensap.TreeBasedTensor) and \
                    out.tensor.ranks[out.tensor.tree.root-1] > 1:
                if self.tensor.ranks[self.tensor.tree.root-1] != \
                        f_2.tensor.ranks[f_2.tensor.tree.root-1]:
                    raise ValueError('Wrong tensor shapes.')
                else:
                    root = out.tensor.tree.root
                    c = out.tensor.tensors[root-1]
                    n = self.tensor.ranks[root-1]
                    s = [':']*c.order
                    s[-1] = np.arange(n**2, step=n)
                    c = c.sub_tensor(*s)
                    out.tensor.tensors[root-1] = c
                    out.tensor.ranks[root-1] = n
        else:
            out = deepcopy(self)
            out.tensor = out.tensor * f_2
        return out

    def parameter_gradient_eval(self, alpha, x=None, *args):
        '''
        Compute the gradient of the function with respect to its alpha-th
        parameter, evaluated at some points.

        Parameters
        ----------
        alpha : int
            The number of the parameter with respect to which compute the
            gradient of self.
        x : list or numpy.ndarray, optional
            The points at which the gradient is to be evaluated. The default is
            None, indicating to use self.bases if self.evaluated_bases is True.

        Raises
        ------
        ValueError
            If no input points are provided.

        Returns
        -------
        grad : Tensor
            The gradient of the function with respect to its alpha-th
            parameter, evaluated at some points.

        '''
        if self.evaluated_bases:
            bases_eval = self.bases
        elif x is not None:
            bases_eval = self.bases.eval(x)
        else:
            raise ValueError('Must provide the evaluation points or the ' +
                             'bases evaluations.')

        dims = np.arange(self.tensor.order)
        if isinstance(self.tensor, tensap.TreeBasedTensor):
            # Compute fH, the TimesMatrixEvalDiag of f with bases_eval in all
            # the dimensions except the ones associated with alpha (if alpha
            # is a leaf node) or with the inactive children of alpha (if
            # alpha is an internal node). The tensor fH is used to compute
            # the gradient of f with respect to f.tensor.tensors[alpha-1].
            tree = self.tensor.tree
            if tree.is_leaf[alpha-1]:
                dims = dims[tree.dim2ind != alpha]
            else:
                children = tree.children(alpha)
                ind = tensap.fast_intersect(
                    tree.dim2ind,
                    children[np.logical_not(
                        self.tensor.is_active_node[children-1])])
                dims = dims[np.logical_not(np.isin(tree.dim2ind, ind))]

            if np.all(self.tensor.is_active_node):
                fH = self.tensor.tensor_matrix_product([bases_eval[x] for x
                                                        in dims], dims)
            else:
                remaining_dims = np.arange(self.tensor.order)
                tensors = np.array(self.tensor.tensors)
                dim2ind = np.array(tree.dim2ind)

                for leaf in tensap.fast_intersect(tree.dim2ind[dims],
                                                  self.tensor.active_nodes):
                    dims = tensap.fast_setdiff(
                        dims, np.nonzero(tree.dim2ind == leaf)[0][0])
                    tensors[leaf-1] = self.tensor.tensors[leaf-1].\
                        tensor_matrix_product(bases_eval[
                            np.nonzero(tree.dim2ind == leaf)[0][0]], 0)

                for pa in np.unique(tree.parent(tensap.fast_setdiff(
                        tree.dim2ind[dims], self.tensor.active_nodes))):
                    ind = tensap.fast_intersect(tree.dim2ind[dims],
                                                tree.children(pa))
                    ind = tensap.fast_setdiff(ind, self.tensor.active_nodes)
                    dims_loc = np.array([np.nonzero(x == tree.dim2ind)[0][0]
                                         for x in ind])
                    if len(ind) > 1:
                        tensors[pa-1] = self.tensor.tensors[pa-1].\
                            tensor_matrix_product_eval_diag([bases_eval[x] for
                                                             x in dims_loc],
                                                            tree.child_number(
                                                                ind)-1)
                        remaining_dims = tensap.fast_setdiff(remaining_dims,
                                                             dims_loc[1:])
                        if np.all(np.logical_not(self.tensor.is_active_node[
                                tree.children(pa)-1])):
                            dim2ind[dims_loc[0]] = tree.parent(
                                tree.dim2ind[dims_loc[0]])
                        else:
                            dims = tensap.fast_setdiff(dims, dims_loc[0])
                        dim2ind[dims_loc[1:]] = 0
                        perm = np.concatenate((
                            [tree.child_number(ind[0])-1],
                            tensap.fast_setdiff(np.arange(tensors[pa-1].order),
                                                tree.child_number(ind[0])-1)))
                        tensors[pa-1] = tensors[pa-1].itranspose(perm)
                    elif len(ind) == 1:
                        dims = dims[dims != dims_loc]
                        tensors[pa-1] = self.tensor.tensors[pa-1].\
                            tensor_matrix_product([bases_eval[x] for
                                                   x in dims_loc],
                                                  tree.child_number(ind)-1)
                        dim2ind[dims_loc] = tree.dim2ind[dims_loc]

                keep_ind = tensap.fast_setdiff(np.arange(tree.nb_nodes),
                                               tree.dim2ind[dims]-1)
                adj_mat = tree.adjacency_matrix[np.ix_(keep_ind, keep_ind)]
                dim2ind = dim2ind[dim2ind != 0]

                ind = np.zeros(tree.nb_nodes)
                ind[tensap.fast_setdiff(np.arange(tree.nb_nodes),
                                        keep_ind)] = 1
                ind = np.cumsum(ind).astype(int)
                dim2ind -= ind[dim2ind-1]
                alpha = alpha - ind[alpha-1]

                tree = tensap.DimensionTree(dim2ind, adj_mat)
                fH = tensap.TreeBasedTensor(tensors[keep_ind], tree)
                fH = fH.remove_unique_children()
                bases_eval = [bases_eval[x] for x in remaining_dims]
        else:
            if alpha <= self.tensor.order:
                dims = np.delete(dims, alpha-1)
            fH = self.tensor.tensor_matrix_product([bases_eval[x] for
                                                    x in dims], dims)

        grad = fH.parameter_gradient_eval_diag(alpha, bases_eval)
        if isinstance(self.tensor, tensap.TreeBasedTensor) and \
                not tree.is_leaf[alpha-1]:
            # If the order of the children has been modified in grad, compute
            # the inverse permutation.
            ch = tree.children(alpha)
            perm_1 = np.argsort(np.concatenate((
                np.atleast_1d(ch[fH.is_active_node[ch-1]]),
                np.atleast_1d(ch[np.logical_not(fH.is_active_node[ch-1])]))))

            if alpha == tree.root:
                perm_2 = []
            else:
                perm_2 = [fH.tensors[alpha-1].order]

            if alpha != tree.root and self.tensor.ranks[tree.root-1] > 1:
                perm_3 = [grad.order-1]
            else:
                perm_3 = []

            grad = grad.transpose(np.concatenate(([0], perm_1+1,
                                                  perm_2, perm_3)).astype(int))

        return grad

    def parameter_gradient_eval_dmrg(self, alpha, x=None, dmrg_type='dmrg',
                                     *args):
        if self.evaluated_bases:
            bases_eval = self.bases
        elif x is not None:
            bases_eval = self.bases.eval(x)
        else:
            raise ValueError('Must provide the evaluation points or the ' +
                             'bases evaluations.')

        dims = np.arange(self.tensor.order)
        if isinstance(self.tensor, tensap.TreeBasedTensor):
            # Compute fH, the TimesMatrixEvalDiag of f with bases_eval in all
            # the dimensions except the ones associated with alpha (if alpha
            # is a leaf node) or with the inactive children of alpha (if
            # alpha is an internal node). The tensor fH is used to compute
            # the gradient of f with respect to f.tensor.tensors[alpha-1].
            tree = self.tensor.tree
            if tree.is_leaf[alpha-1]:
                dims = dims[tree.dim2ind != alpha]
            else:
                children = tree.children(alpha)
                ind = tensap.fast_intersect(
                    tree.dim2ind,
                    children[np.logical_not(
                        self.tensor.is_active_node[children-1])])
                dims = dims[np.logical_not(np.isin(tree.dim2ind, ind))]

            fH = self.tensor.tensor_matrix_product([bases_eval[x] for x
                                                    in dims], dims)
        else:
            if alpha <= self.tensor.order:
                dims = np.delete(dims, alpha-1)
            fH = self.tensor.tensor_matrix_product([bases_eval[x] for
                                                    x in dims], dims)

        grad, g_alpha, g_gamma = \
            fH.parameter_gradient_eval_diag_dmrg(alpha, bases_eval)

        if isinstance(self.tensor, tensap.TreeBasedTensor):
            # If the order of the children has been modified in grad, compute
            # the inverse permutation.
            ch = tree.children(alpha)

            if ch.size == 0:
                perm_1 = np.array([0])
            else:
                perm_1 = np.argsort(np.concatenate((
                    np.atleast_1d(ch[fH.is_active_node[ch-1]]),
                    np.atleast_1d(
                        ch[np.logical_not(fH.is_active_node[ch-1])]))))
            gamma = tree.parent(alpha)
            ch = tensap.fast_setdiff(tree.children(gamma), alpha)
            perm_1b = np.argsort(np.concatenate((
                    np.atleast_1d(ch[fH.is_active_node[ch-1]]),
                    np.atleast_1d(
                        ch[np.logical_not(fH.is_active_node[ch-1])]))))

            if dmrg_type == 'dmrg':
                perm_1 = np.concatenate((perm_1, perm_1.size+perm_1b))
                perm_2 = []
                if alpha != tree.root and gamma != tree.root:
                    perm_2 = [fH.tensors[alpha-1].order +
                              fH.tensors[gamma-1].order-2]
                perm_3 = []
                if alpha != tree.root and self.tensor.ranks[tree.root-1] > 1:
                    perm_3 = [grad.order-1]
                grad = grad.transpose(np.concatenate(
                    ([0], perm_1+1, perm_2, perm_3)).astype(int))
            elif dmrg_type == 'dmrg_low_rank':
                g_alpha = g_alpha.transpose(np.concatenate(([0], perm_1+1)))
                perm_2 = []
                if gamma != tree.root:
                    # TODO Checks
                    perm_2 = [fH.tensors[gamma-1].order-1]
                perm_3 = []
                if alpha != tree.root and self.tensor.ranks[tree.root-1] > 1:
                    perm_3 = [grad.order-1]
                g_gamma = g_gamma.transpose(np.concatenate(
                    ([0], perm_1b+1, perm_2, perm_3)).astype(int))

                grad = [g_alpha, g_gamma]
            else:
                raise ValueError('Wrong DMRG type.')
        return grad

    def eval_derivative(self, n, x, *dims):
        '''
        Evaluate the n-th order derivative of self at the points x.

        If dims is provided, compute the partial evaluations of the n-th order
        derivative of self at points x n dimensions in dims.

        Parameters
        ----------
        n : int
            The order of derivation.
        x : list or numpy.ndarray
            The points at which the function is to be evaluated.
        *dims : list or numpy.ndarray, optional
            The dimensions of the partial evaluation. If not provided,
            evaluate the function in all dimensions.

        Returns
        -------
        numpu.ndarray or Tensor
            The evaluations of the n-th derivative of self at the points x.

        '''
        bases_eval = self.bases.eval_derivative(n, x, *dims)
        return self.eval_with_bases_evals(bases_eval, *dims)

    def derivative(self, n):
        '''
        Compute the n-th order derivative of self.

        Parameters
        ----------
        n : int
            The order of derivation.

        Returns
        -------
        df : tensap.FunctionalTensor
            The n-th order derivative of self.

        '''
        df = deepcopy(self)
        df.bases = self.bases.derivative(n)
        return df

    def eval_on_grid(self, x, dims=None):
        '''
        Compute evaluations of self at points x.

        Parameters
        ----------
        x : list
            List such that x[k] contains the grid associated with the (k+1)-th
            variable.
        dims : list or numpy.ndarray, optional
            Array indicating the dimensions associated with x. The default is
            None, indicating all the dimensions.

        Returns
        -------
        out : FunctionalTensor or tensap.Tensor
            The evaluations of self at points x.

        '''
        if dims is None:
            dims = np.arange(self.bases.length())
        H = self.bases.eval(x, dims)

        if np.size(dims) == self.tensor.order:
            out = self.tensor.tensor_matrix_product(H, dims)
        else:
            out = deepcopy(self)
            out.tensor = self.tensor.tensor_matrix_product(H, dims)
            out.bases = out.bases.remove_bases(dims)
            out.fdims = out.fdims[np.setdiff1d(
                np.range(np.size(out.fdims)), dims)]
            if np.size(out.fdims) == 0:
                out = out.tensor
        return out

    def random(self, *args, **kwargs):
        return self.random_dims(range(len(self.fdims)), *args, **kwargs)

    def random_dims(self, dims, *args, nargout=1):
        '''
        Evaluate the function in dimensions dims using n points drawn randomly
        according to measure if provided, or to
        self.bases.measure.marginal(dims) otherwise.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The dimensions of the bases to be evaluated.
        n : int, optional
            The number of random evaluations. The default is 1.
        measure : tensap.ProbabilityMeasure, optional
            The probability measure used for the generation of the input
            points. The default is None, indicating to use
            self.measure.marginal(dims).

        Returns
        -------
        bases_eval : list or numpy.ndarray
            Random evaluations of the function.
        x : numpy.ndarray
            The input points, grouped by basis.

        '''
        bases_eval, x = self.bases.random_dims(dims, *args, nargout=2)
        if nargout == 1:
            return self.eval_with_bases_evals(bases_eval, dims)
        return self.eval_with_bases_evals(bases_eval, dims), x

    def get_random_vector(self):
        '''
        Return the RandomVector associated with self.bases.

        Returns
        -------
        tensap.RandomVector
            The RandomVector associated with self.bases.

        '''
        return self.bases.get_random_vector()

    def eval_with_bases_evals(self, bases_eval, dims=None):
        '''
        Evaluate the function self, given evaluations of self.bases.

        Parameters
        ----------
        bases_eval : list or numpy.ndarray
            The evaluations of self.bases.
        dims : list or numpy.ndarray, optional
            The dimensions of the evaluation. The default is None, indicating
            all the dimensions.

        Returns
        -------
        out : numpy.ndarray or Tensor
            The (partially) evaluated function.

        '''
        if dims is None:
            dims = range(len(self.bases))
        if len(dims) == 1 and not isinstance(bases_eval, list):
            bases_eval = [bases_eval]
        if len(dims) == self.tensor.order:
            out = self.tensor.tensor_matrix_product_eval_diag(
                bases_eval).numpy()
        else:
            out = deepcopy(self)
            fdims_eval = out.fdims[dims]
            out.tensor = out.tensor.tensor_matrix_product_eval_diag(bases_eval,
                                                                    fdims_eval)
            fdims_eval.sort()
            old_dims = np.setdiff1d(range(self.tensor.order), fdims_eval[2:])
            if out.tensor.shape[fdims_eval[0]] == 1:
                out.tensor = out.tensor.squeeze(fdims_eval[0])
                old_dims = np.delete(old_dims, fdims_eval[0])
            out.bases = out.bases.remove_bases(dims)
            out.fdims = np.delete(out.fdims, dims)
            out.fdims = np.nonzero(np.isin(old_dims, out.fdims))[0]
            if out.fdims.size == 0:
                out = out.tensor
                if out.order == 1:
                    out = out.numpy()
        return out

    def storage(self):
        '''
        Return the storage requirement of the FunctionalTensor.

        Returns
        -------
        int
            The storage requirement of the FunctionalTensor.

        '''
        return self.tensor.storage()
