'''
Module canonical_tensor.

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


class CanonicalTensor:
    '''
    Class CanonicalTensor.

    Attributes
    ----------
    core : tensap.DiagonalTensor
        The core of the canonical tensor.
    space : numpy.ndarray
        The tensor space.
    order : int
        The order of the tensor.
    shape : numpy.ndarray
        The shape of the tensor.
    is_orth : bool
        Boolean indicating, if is_orth = True, the dimension mu for which
        the mu-matricization of the tensor is orthogonal.

    '''

    def __init__(self, space, core):
        '''
        Constructor for the class CanonicalTensor

        Parameters
        ----------
        space : numpy.ndarray or list
        The tensor space.
        core : tensap.DiagonalTensor or list or numpy.ndarray
        The core of the canonical tensor.

        Returns
        -------
        None.

        '''
        assert isinstance(space, (list, np.ndarray)), \
            'The input space must be a list or a numpy.ndarray.'

        assert isinstance(core, (list, np.ndarray, tensap.DiagonalTensor)), \
            ('The input core must be a list or a numpy.ndarray or a ' +
             'tensap.DiagonalTensor.')

        if not isinstance(core, tensap.DiagonalTensor):
            core = tensap.DiagonalTensor(core, len(space))

        self.core = core
        self.space = space
        self.order = core.order
        self.shape = np.array([x.shape[0] for x in self.space])
        self.is_orth = False

    def __add__(self, arg):
        core = np.concatenate((self.core.data, arg.core.data))
        space = [np.hstack((x, y)) for x, y in zip(list(self.space),
                                                   list(arg.space))]
        return CanonicalTensor(space, core)

    def __radd__(self, arg):
        return self + arg

    def __sub__(self, arg):
        return self + (-arg)

    def __rsub__(self, arg):
        return -self + arg

    def __neg__(self):
        return CanonicalTensor(list(self.space), -self.core.data)

    def __repr__(self):
        return ('<{} CanonicalTensor:{n}' +
                '{t}order = {},{n}' +
                '{t}shape = {},{n}' +
                '{t}is_orth = {}>').format('x'.join(map(str, self.shape)),
                                           self.order,
                                           self.shape,
                                           self.is_orth,
                                           t='\t', n='\n')

    @property
    def ndim(self):
        '''
        Compute the order of the tensor. Equivalent to self.order.

        Returns
        -------
        int
            The order of the tensor.

        '''
        return self.order

    def orth(self):
        '''
        Return an orthonormalized representation of the tensor.

        Returns
        -------
        out : CanonicalTensor
            The orthonormalized representation of the tensor.

        '''
        dims = range(self.order)
        qr = [np.linalg.qr(x) for x in self.space]
        space = [x[0] for x in qr]
        M = [x[1] for x in qr]
        core = self.core.tensor_matrix_product(M, dims)
        core = core.orth()[0]
        out = CanonicalTensor(space, self.core)
        out.core = core
        return out

    def full(self):
        '''
        Convert the object to a tensap.FullTensor.

        Returns
        -------
        tensap.FullTensor
            The canonical tensor as a tensap.FullTensor.

        '''
        return self.core.full().tensor_matrix_product(self.space)

    def numpy(self):
        '''
        Convert the CanonicalTensor to a numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            The CanonicalTensor as a numpy.ndarray.

        '''
        return self.full().data

    def tensor_matrix_product(self, matrices, dims=None):
        '''
        Contract a tensor with matrices.

        The second dimension of the matrix matrices[k] is contracted with the
        k-th dimension of self, with the indices k given in dims (if provided).

        Parameters
        ----------
        matrices : numpy.ndarray or list of numpy.ndarray
            The matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        CanonicalTensor
            The tensor after the contractions with the matrices.

        '''
        if dims is None:
            assert isinstance(matrices, (list, np.ndarray)), \
                'matrices should be a list or a numpy.ndarray.'
            assert len(matrices) == self.order, \
                'len(matrices) must be self.order.'
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)
            if not isinstance(matrices, list):
                matrices = [matrices]
            assert len(matrices) == dims.size, \
                'len(matrices) must be equal to dims.size.'

        space = list(self.space)
        for i, dim in enumerate(dims):
            space[dim] = np.matmul(matrices[i], space[dim])
        return CanonicalTensor(space, np.array(self.core.data))

    def tensor_matrix_product_eval_diag(self, matrices, dims=None):
        '''
        Evaluate the diagonal of a tensor obtained by contraction with
        matrices.

        Provides the diagonal of the tensor obtained by contracting the tensor
        with matrices H[k] along dimensions dims(k)+1, for k = 0, ...,
        dims.size-1.

        Parameters
        ----------
        matrices : list
            The matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        out : CanonicalTensor or tensap.FullTensor
            The diagonal of the contractions of the tensor with the matrices.

        '''
        return self.tensor_matrix_product(matrices, dims).eval_diag(dims)

    def eval_diag(self, dims=None):
        '''
        Extract the diagonal of the tensor.

        The tensor must be such that self.shape[mu] = n for all mu (in dims if
        provided).

        Parameters
        ----------
        dims : list of numpy.ndarray, optional
            The dimensions associated with the indices of the diagonal. The
            default is None,indicating that the indices refer to all the
            dimensions.

        Returns
        -------
        data : CanonicalTensor or tensap.FullTensor
            The evaluations of the diagonal of the tensor.

        '''
        if dims is None:
            is_none = True
            dims = np.arange(self.order)
        else:
            is_none = False
            dims = np.atleast_1d(dims)
            if dims.size == 1:
                print('Only one dimension: degenerate case for eval_diag, ' +
                      'returning the tensor itself.')
                return deepcopy(self)
            dims = np.sort(dims)
            new_dims = np.setdiff1d(range(self.order), dims[1:])

        out = self.space[dims[0]]
        for k in dims[1:]:
            out *= self.space[k]

        if is_none or dims.size == self.order:
            out = tensap.FullTensor(np.matmul(out, self.core.data))
        else:
            space = list(self.space)
            core = deepcopy(self.core)

            space[dims[0]] = out
            space = space[new_dims]
            core.shape = core.shape[new_dims]
            core.order = new_dims.size
            out = CanonicalTensor(space, core)
        return out

    def dot(self, tensor_2):
        '''
        Return the inner product of two tensors.

        Parameters
        ----------
        tensor2 : CanonicalTensor
            The second tensor of the inner products.

        Returns
        -------
        numpy.float
            The inner product of the two tensors.

        '''
        assert isinstance(tensor_2, tensap.CanonicalTensor), \
            'The second argument must be a tensap.CanonicalTensor.'
        matrices = [np.matmul(np.transpose(x), y) for
                    x, y in zip(self.space, tensor_2.space)]
        out = tensor_2.core.full().tensor_matrix_product(matrices)
        return self.core.full().dot(out)

    def norm(self, matrix=None):
        '''
        Compute the canonical norm of the CanonicalTensor.

        Returns
        -------
        numpy.float
            The norm of the tensor.

        '''
        if matrix is None:
            if self.is_orth:
                norm = np.linalg.norm(self.core.data)
            else:
                norm = np.sqrt(np.abs(self.dot(self)))
        else:
            raise NotImplementedError('Method not implemented.')
        return norm

    def storage(self):
        '''
        Return the storage complexity of the CanonicalTensor.

        Returns
        -------
        int
            The storage complexity of the CanonicalTensor.

        '''
        return self.core.data.size + np.sum([x.size for x in self.space])

    def sparse_storage(self):
        '''
        Return the sparse storage complexity of the CanonicalTensor.

        Returns
        -------
        int
            The sparse storage complexity of the CanonicalTensor.

        '''
        return np.count_nonzero(self.core.data) + \
            np.sum([np.count_nonzero(x) for x in self.space])

    def representation_rank(self):
        '''
        Return the representation rank of the tensor.

        Returns
        -------
        int
            The representation rank of the tensor.

        '''
        return self.core.data.size

    def parameter_gradient_eval_diag(self, mu, matrices=None):
        '''
        Compute the diagonal of the gradient of the tensor with respect to a
        given parameter.

        Parameters
        ----------
        mu : int
            Index of the parameter.
        matrices : list or numpy.array, optional
            Matrices with which to compute outer_product_eval_diag if alpha is
            associated with some dimensions. Useful for evaluation the gradient
            of a tensap.FunctionalTensor. The default is None, indicating
            identity matrices.

        Returns
        -------
        out : tensap.FullTensor
            The diagonal of the gradient of the tensor with respect to
            the parameter with index mu.

        '''
        rank = len(self.core.data)
        if mu == self.order + 1:
            N = self.space[0].shape[0]
            out = np.ones((N, rank))
            for nu in range(self.order):
                out *= self.space[nu]
            out = tensap.FullTensor(out)
        else:
            no_mu = np.setdiff1d(np.arange(1, self.order+1), mu)
            N = self.space[no_mu[0]-1].shape[0]
            f_mu = np.ones((N, rank))
            for nu in no_mu:
                f_mu *= self.space[nu-1]
            if matrices is not None:
                out = tensap.FullTensor(f_mu).outer_product_eval_diag(
                    tensap.FullTensor(matrices[mu-1]), 0, 0)
            else:
                out = tensap.FullTensor(f_mu).outer_product_eval_diag(
                    tensap.FullTensor(np.eye(self.shape[mu-1])), [], [], True)
        return out

    def tree_based_tensor(self, tree, is_active_node=None):
        '''
        Convert a CanonicalTensor to a tensap.TreeBasedTensor with given
        dimension tree and active nodes.

        Parameters
        ----------
        tree : tensap.DimensionTree
            The dimension tree.
        is_active_node : list or numpy.ndarray, optional
            Booleans indicating if the nodes are active. The default is None,
            settings all the nodes active.

        Returns
        -------
        out : tensap.TreeBasedTensor
            The canonical tensor converted into a tree-based tensor.

        '''

        rank = self.core.shape[0]
        tensors = np.empty(tree.nb_nodes, dtype=object)
        for nod in np.arange(1, tree.nb_nodes+1):
            ch = tree.children(nod)
            if tree.parent(nod) == 0:
                order = ch.size
                tensors[nod-1] = tensap.FullTensor.diag(self.core.data, order)
            else:
                order = ch.size + 1
                tensors[nod-1] = tensap.FullTensor.diag(np.ones(rank), order)

        tensors[tree.dim2ind-1] = self.space
        out = tensap.TreeBasedTensor(tensors, tree)
        if is_active_node is not None:
            out = out.inactivate_nodes(
                np.nonzero(np.logical_not(is_active_node))[0]+1)
        return out

    @staticmethod
    def create(generator, rank, shape):
        '''
        Create a FullTensor of rank rank and shape shape using a given
        generator.

        Parameters
        ----------
        generator : function
            Function generating a numpy.ndarray, given a shape.
        rank : int
            The rank of the tensor.
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        CanonicalTensor
            The created tensor.

        '''
        space = [generator([x, rank]) for x in shape]
        return CanonicalTensor(space, np.ones(rank))

    @staticmethod
    def zeros(rank, shape):
        '''
        Create a FullTensor of rank rank and shape shape with entries equal to
        0.

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        CanonicalTensor
            The created tensor.

        '''
        return CanonicalTensor.create(np.zeros, rank, shape)

    @staticmethod
    def ones(rank, shape):
        '''
        Create a FullTensor of rank rank and shape shape with entries equal to
        1.

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        CanonicalTensor
            The created tensor.

        '''
        return CanonicalTensor.create(np.ones, rank, shape)

    @staticmethod
    def rand(rank, shape):
        '''
        Create a FullTensor of rank rank and shape shape with i.i.d. entries
        drawn according to the uniform distribution on [0, 1].

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        CanonicalTensor
            The created tensor.

        '''
        return CanonicalTensor.create(lambda x: np.random.rand(*x), rank,
                                      shape)

    @staticmethod
    def randn(rank, shape):
        '''
        Create a FullTensor of rank rank and shape shape with i.i.d. entries
        drawn according to the standard gaussian distribution.

        Parameters
        ----------
        rank : int
            The rank of the tensor.
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        CanonicalTensor
            The created tensor.

        '''
        return CanonicalTensor.create(lambda x: np.random.randn(*x), rank,
                                      shape)
