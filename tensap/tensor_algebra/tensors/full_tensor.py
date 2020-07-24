'''
Module full_tensor.

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

# from string import ascii_lowercase
import numpy as np
import tensap


class FullTensor:
    '''
    Class FullTensor.

    Attributes
    ----------
    data : numpy.ndarray
        The content of the tensor.
    order : int
        The order of the tensor.
    shape : numpy.ndarray
        The shape of the tensor.
    is_orth : bool
        Boolean indicating if the representation of the tensor is orthogonal
        (i.e. one mu-matricization is orthogonal).
    orth_dim : bool
        Boolean indicating, if is_orth = True, the dimension mu for which
        the mu-matricization of the tensor is orthogonal.

    '''

    # Override numpy's operations with reversed operands
    __array_priority__ = 1

    def __init__(self, data, order=None, shape=None):
        '''
        Constructor for the class FullTensor.

        Parameters
        ----------
        data : numpy.ndarray or tensap.FullTensor or tensorflow.Tensor
            The content of the tensor.
        order : int, optional
            The order of the tensor. The default is None, indicating to infer
            it from data.
        shape : list or numpy.ndarray, optional
            The shape of the tensor. The default is None, the shape is deduced
            from data.

        Returns
        -------
        None.

        '''
        if hasattr(data, 'data'):
            self.data = np.array(data.data)
        else:
            self.data = np.array(data)
        if shape is not None:
            self.data = np.reshape(self.data, shape, order='F')

        ndim = np.ndim(self.data)
        if order is not None and ndim != order:
            for d in np.arange(ndim, order):
                self.data = np.expand_dims(self.data, d)

        self.is_orth = False
        self.orth_dim = None

    @property
    def order(self):
        '''
        Compute the order of the tensor.

        Returns
        -------
        int
            The order of the tensor.

        '''
        return np.ndim(self.data)

    @property
    def ndim(self):
        '''
        Compute the order of the tensor. Equivalent to self.order.

        Returns
        -------
        int
            The order of the tensor.

        '''
        return self.data.ndim

    @property
    def size(self):
        '''
        Compute the number of elements of the tensor.

        Returns
        -------
        numpy.ndarray
            The number of elements of the tensor.

        '''
        return np.array(self.data.shape)

    @property
    def shape(self):
        '''
        Compute the shape of the tensor

        Returns
        -------
        numpy.ndarray
            The shape of the tensor.

        '''
        return np.array(self.data.shape)

    def tree_based_tensor(self):
        '''
        Convert a FullTensor into a TreeBasedTensor.

        Returns
        -------
        TreeBasedTensor
            The FullTensor in tree-based tensor format.

        '''
        tree = tensap.DimensionTree.trivial(self.order)
        tensors = [FullTensor(self)]
        for dim in range(self.order):
            tensors.append(FullTensor(np.eye(self.shape[dim])))
        return tensap.TreeBasedTensor(tensors, tree)

    def sparse(self):
        '''
        Conversion of a FullTensor into a SparseTensor.

        Returns
        -------
        tensap.SparseTensor
            A SparseTensor representation of the FullTensor.

        '''
        dat = np.reshape(self.data, -1, order='F')
        ind = np.nonzero(dat)[0]
        indices = tensap.MultiIndices.ind2sub(self.shape, ind)
        return tensap.SparseTensor(dat, indices, self.shape)

    def __repr__(self):
        return ('<{} FullTensor:{n}' +
                '{t}order = {},{n}' +
                '{t}shape = {},{n}' +
                '{t}is_orth = {},{n}' +
                '{t}orth_dim = {}>').format('x'.join(map(str, self.shape)),
                                            self.order,
                                            self.shape,
                                            self.is_orth,
                                            self.orth_dim,
                                            t='\t', n='\n')

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __eq__(self, tensor2):
        return np.all(self.data == tensor2.data)

    def __add__(self, arg):
        if isinstance(arg, FullTensor):
            arg = arg.data
        return FullTensor(self.data + arg)

    def __radd__(self, arg):
        return self + arg

    def __sub__(self, arg):
        if isinstance(arg, FullTensor):
            arg = arg.data
        return FullTensor(self.data - arg)

    def __rsub__(self, arg):
        return -self + arg

    def __neg__(self):
        return FullTensor(-self.data)

    def __mul__(self, arg):
        if isinstance(arg, FullTensor):
            arg = arg.data
        return FullTensor(self.data * arg)

    def __rmul__(self, arg):
        return self * arg

    def __truediv__(self, arg):
        if isinstance(arg, FullTensor):
            arg = arg.data
        return FullTensor(self.data / arg)

    def __pow__(self, arg):
        assert np.isscalar(arg), 'The power must be a scalar.'
        return FullTensor(self.data ** arg)

    def hadamard_product(self, arg):
        '''
        Compute the Hadamard product of two tensors.

        Equivalent to self * arg.

        Parameters
        ----------
        arg : tensap.FullTensor or numpy.ndarray
            The second tensor of the Hadamard product.

        Returns
        -------
        FullTensor
            The tensor resulting from the Hadamard product.

        '''
        return self * arg

    def storage(self):
        '''
        Return the storage complexity of the FullTensor.

        Returns
        -------
        int
            The storage complexity of the FullTensor.

        '''
        return np.size(self.data)

    def sparse_storage(self):
        '''
        Return the sparse storage complexity of the FullTensor.

        Returns
        -------
        int
            The sparse storage complexity of the FullTensor.

        '''
        return np.count_nonzero(self.data)

    def numpy(self):
        '''
        Convert the FullTensor to a numpy.ndarray.

        Returns
        -------
        numpy.ndarray
            The FullTensor as a numpy.ndarray.

        '''
        return self.data

    def eval_at_indices(self, indices, dims=None):
        '''
        Evaluate the tensor at indices.

        If dims is None, return
        s(k) = x(indices(k, 1), indices(k, 2), ..., indices(k, d)),
        1 <= k <= self.shape[0].

        If dims is not None, return a partial evaluation: up to a permutation
        (placing the dimensions dims on the left), return
        s(k, i_1, ..., i_d') = x(indices(k, 1), indices(k, 2), ...,
        indices(k, M), i_1, ..., i_d'),
        1 <= k <= self.shape[0], with M = dims.size and d' = self.order - M.

        Parameters
        ----------
        indices : list of numpy.ndarray
            The indices of the tensor.
        dims : list of numpy.ndarray, optional
            The dimensions associated with the indices. The default is None,
            indicating that indices refers to all the dimensions.

        Returns
        -------
        evaluations : numpy.ndarray or FullTensor
            The evaluations of the tensor.

        '''
        indices = np.atleast_2d(indices)
        if dims is None:
            dims = np.arange(self.order)
        else:
            dims = np.atleast_1d(dims)
            if indices.shape[1] != dims.size:
                indices = np.transpose(indices)
            assert dims.size == indices.shape[1], \
                'Wrong size of multi-indices.'
            sort_ind = np.argsort(dims)
            dims = dims[sort_ind]
            indices = indices[:, sort_ind]
        assert dims.size == indices.shape[1], 'Wrong size of multi-indices.'

        if dims.size == self.order:
            data = self
            evaluations = np.array([data[tuple(i)] for i in indices.tolist()])
        elif dims.size == 1:
            ind = [':']*self.order
            ind[dims[0]] = np.ravel(indices).tolist()
            evaluations = self.sub_tensor(*ind)
        else:
            no_dims = tensap.fast_setdiff(np.arange(self.order), dims)
            indices = np.ravel_multi_index(np.transpose(indices),
                                           [self.shape[i] for i in dims])
            evaluations = self.matricize(dims).sub_tensor(indices, ':')
            evaluations = evaluations.reshape([indices.size] +
                                              [self.shape[i] for i in no_dims])
            left_dims = np.arange(dims[0])
            evaluations = evaluations.transpose(
                np.concatenate((np.arange(1, left_dims.size + 1), [0],
                                np.arange(left_dims.size + 1,
                                          self.order - dims.size + 1))))
        return evaluations

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
        data : numpy.ndarray
            The evaluations of the diagonal of the tensor.

        '''
        if dims is None:
            dims = np.arange(self.order)
        else:
            dims = np.atleast_1d(dims)

        if dims.size == 1:
            data = self
        else:
            assert np.all([self.shape[x] == self.shape[dims[0]] for
                           x in dims]),\
             'The shapes of the tensor in dimensions dims should be equal.'
            ind = np.repeat(np.reshape(np.arange(self.shape[dims[0]]),
                                       [-1, 1]), dims.size, 1)
            data = self.eval_at_indices(ind, dims)
        return data

    def reshape(self, shape):
        '''
        Reshape the tensor.

        Parameters
        ----------
        shape : list or numpy.ndarray
            The new shape of the tensor.

        Returns
        -------
        tensor : FullTensor
            The reshaped tensor.

        '''
        tensor = FullTensor(self)
        tensor.data = np.reshape(tensor.data, shape, order='F')
        return tensor

    def transpose(self, dims):
        '''
        Transpose (permute) the dimensions of the tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The new ordering of the dimensions.

        Returns
        -------
        tensor : FullTensor
            The transposed (permuted) tensor.

        '''
        tensor = FullTensor(self)
        tensor.data = np.transpose(tensor.data, dims)
        return tensor

    def itranspose(self, dims):
        '''
        Return the inverse transpose (permutation) of the dimensions of the
        tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray
            The original transpose (permutation) indices.

        Returns
        -------
        FullTensor
            The transposed (permuted) tensor.

        '''
        return self.transpose(np.argsort(dims))

    def squeeze(self, dims=None):
        '''
        Remove the singleton dimensions of the tensor.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            Dimensions to squeeze. The default is None, indicating all the
            singleton dimensions.

        Returns
        -------
        out : float or FullTensor
            The squeezed tensor.

        '''
        if dims is not None:
            dims = tuple(dims)

        out = FullTensor(np.squeeze(self.data, dims))
        if out.order == 0:
            out = out.data
        return out

    def dot(self, tensor2):
        '''
        Return the inner product of two tensors.

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor of the inner products.

        Returns
        -------
        numpy.float
            The inner product of the two tensors.

        '''
        return np.sum(np.multiply(self.data, tensor2.data))

    def norm(self):
        '''
        Compute the canonical norm of the FullTensor.

        Returns
        -------
        numpy.float
            The norm of the tensor.

        '''
        return np.linalg.norm(self.data)

    def full(self):
        '''
        Return the tensor.

        Returns
        -------
        FullTensor
            The tensor.

        '''
        return self

    def sub_tensor(self, *indices):
        '''
        Extract a subtensor of the tensor.

        The result is a tensor s of shape
        len(indices[0]), ..., len(indices[self.order-1]),
        such that
        s(k1,...,kd) = x(indices[0][k1], ..., indices[self.order-1][kd]).

        Example: x.subTensor([1, 2], ':', [2, 5, 6]) returns a tensor with
        shape [2, self.shape[1], 3].

        Parameters
        ----------
        *indices : list
            The indices to extract in each dimension. ':' indicates all the
            indices.

        Returns
        -------
        FullTensor
            The subtensor.

        '''
        data = self.data
        order = self.order
        for dim in range(self.order):
            ind_loc = np.atleast_1d(indices[dim])
            if ind_loc.size != 1 or ind_loc[0] != ':':
                data = np.take(data, ind_loc, axis=dim)
                if order != data.ndim:
                    data = np.expand_dims(data, dim)
                order = data.ndim
        return FullTensor(data)

    def cat(self, tensor2, dims=None):
        '''
        Concatenate the tensors.

        Concatenates self and tensor2 in a tensor z such that:
        z(i_1 ,..., i_d) = x(i_1, ..., i_d) if i_k <= sz[k]-1 for k in dims,
        z(i_1, ..., i_d) = y(i_1-sz[0], ..., i_d-sz[d-1]) if i_k >= sz[k]
        for k in dims,
        z(i_1, ..., i_d) = 0 otherwise, with sz = self.shape and
        dims = range(self.order) if not provided.

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor to be concatenaed.
        dims : list or numpy.ndarray, optional
            The dimensions of the concatenation. The default is None,
            indicating all the dimensions.
        Returns
        -------
        data : FullTensor
            The concatenated tensors.

        '''
        assert self.order == tensor2.order, \
            'The orders of the tensors must be equal.'

        tensor1 = FullTensor(self)
        order = self.order
        shape1 = np.atleast_1d(self.shape)
        shape2 = np.atleast_1d(tensor2.shape)

        if dims is None:
            dims = range(self.order)

        dims = np.atleast_1d(dims)
        dims_not = tensap.fast_setdiff(np.arange(order), dims)
        assert np.all([a == b for a, b in zip(shape1[dims_not],
                                              shape2[dims_not])]), \
            'The dimensions of the tensors are not compatible.'

        if dims.size == 1:
            data = np.concatenate([tensor1.data, tensor2.data], dims[0])
        else:
            shape_out = np.array(shape1)
            shape_out[dims] = shape1[dims] + shape2[dims]

            padding = np.transpose([[0]*order, shape_out - shape1])
            data = np.pad(tensor1.data, padding)
            padding = np.transpose([shape_out - shape2, [0]*order])
            data += np.pad(tensor2.data, padding)

        return FullTensor(data)

    def reduce_sum(self, dims=None):
        '''
        Compute the sum of elements across dimensions dims of a tensor.

        Similar to tensorflow.reduce_sum.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            The dimensions to be reduced. The default is None, indicating all
            the dimensions.

        Returns
        -------
        FullTensor
            The reduced tensor.

        '''
        return FullTensor(np.sum(self.data, dims, keepdims=True))

    def reduce_mean(self, dims=None):
        '''
        Compute the mean of elements across dimensions dims of a tensor.

        Similar to tensorflow.mean.

        Parameters
        ----------
        dims : list or numpy.ndarray, optional
            The dimensions to be reduced. The default is None, indicating all
            the dimensions.

        Returns
        -------
        FullTensor
            The reduced tensor.

        '''
        return FullTensor(np.mean(self.data, dims, keepdims=True))

    def kron(self, tensor2):
        '''
        Kronecker product of tensors.

        Similar to numpy.kron but for arbitrary tensors.

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor of the Kronecker product.

        Returns
        -------
        FullTensor
            The tensor resulting from the Kronecker product.

        '''
        order1 = self.order
        order2 = tensor2.order
        order_max = np.max((order1, order2))

        shape1 = np.concatenate((self.shape,
                                np.ones(order_max - order1, dtype=int)))
        shape2 = np.concatenate((tensor2.shape,
                                np.ones(order_max - order2, dtype=int)))

        data1 = np.reshape(self.data, [-1, 1])
        data2 = np.reshape(tensor2.data, [1, -1])

        perm = np.reshape(np.transpose(np.reshape(np.arange(2*order_max),
                                                  [2, order_max])),
                          2*order_max)

        data = np.reshape(np.transpose(np.reshape(np.matmul(data1, data2),
                                                  np.concatenate((shape1,
                                                                  shape2))),
                                       perm), shape1 * shape2)
        return FullTensor(data, shape=self.shape*tensor2.shape)

    def orth(self, dim=None):
        '''
        Orthogonalize the tensor.

        Parameters
        ----------
        dim : int, optional
            The dimension of the orthogonal dim-matricization of self. The
            default is None, returning a copy of the original tensor.

        Returns
        -------
        tensor : FullTensor
            A tensor whose dim-matricization is an orthogonal matrix
            corresponding to the Q factor of a QR factorization of the
            dim-matricization of self.
        r_matrix : numpy.ndarray
            The R factor.

        '''
        tensor = FullTensor(self)  # Copy the tensor

        if dim is None:
            return tensor, np.array([])

        if dim == -1:
            dim = tensor.order-1

        dims = np.concatenate((np.arange(dim),
                               np.arange(dim+1, tensor.order),
                               [dim]))
        tensor = tensor.transpose(dims)

        shape0 = np.array(tensor.shape)
        tensor = tensor.reshape([np.prod(shape0[:-1]), shape0[-1]])

        try:
            from tensorflow.python.ops.gen_linalg_ops import qr
            q_tf, r_tf = qr(tensor.data, full_matrices=False)
            tensor.data, r_matrix = q_tf.numpy(), r_tf.numpy()
        except ImportError:
            tensor.data, r_matrix = np.linalg.qr(tensor.data)

        shape0[-1] = r_matrix.shape[0]
        tensor = tensor.reshape(shape0)
        tensor = tensor.itranspose(dims)
        tensor.is_orth = True
        tensor.orth_dim = dim
        return tensor, r_matrix

    def dot_with_rank_one_metric(self, tensor2, matrix):
        '''
        Compute the weighted inner product of two tensors.

        Compute the weighted canonical inner product of self and tensor2,
        where the inner product related to dimension k is weighted by
        matrix[k]. It is equivalent to
        self.dot(tensor2.tensor_matrix_product(matrix)),
        but can be much faster.

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor of the inner product.
        matrix : list or numpy.ndarray or FullTensor
            The weight matrix.

        Returns
        -------
        numpy.float
            The weighted inner product.

        '''
        return self.dot(tensor2.tensor_matrix_product(matrix))

    def tensordot_matrix_product_except_dim(self, tensor2, matrices, dim):
        '''
        Particular type of contraction.

        Compute a special contraction of two tensors self, tensor2, a list of
        matrices matrices and a particular dimension dim. Note that dim must
        be a scalar, while matrices must be a list array with self.order
        elements.

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor of the contraction.
        matrices : list
            The list of matrices of the contraction.
        dim : int
            The excluded dimension.

        Returns
        -------
        numpy.ndarray
            The result of the contraction.

        '''
        assert isinstance(matrices, list), 'matrices should be a list.'
        assert len(matrices) == self.order, \
            'len(matrices) must be self.order.'

        dims = tensap.fast_setdiff(np.arange(self.order), dim)
        matrices = [matrices[i] for i in dims]
        tmp = tensor2.tensor_matrix_product(matrices, dims)
        tmp = self.tensordot(tmp, dims, dims)
        return tmp

    def tensordot(self, tensor2, dims1, dims2=None):
        '''
        Contract two tensors along specified dimensions.

        Similar to tensorflow.tensordot.

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor of the contraction.
        dims1 : list or int
            The dimensions of contractions for the first tensor.
        dims2 : list or int, optional
            The dimensions of contractions for the second tensor. The default
            is None which indicates, if dims1 = 0, to perform the outer
            product of the two tensors, similarly to tensorflow.tensordot.

        Returns
        -------
        out : FullTensor
            The resulting tensor.

        '''
        dims1 = np.atleast_1d(dims1)
        if dims1.size == 1 and dims1 == 0 and dims2 is None:
            # Outer product (notation similar to tensorflow)
            out = FullTensor(np.tensordot(self.data, tensor2.data, 0))
        else:
            dims2 = np.atleast_1d(dims2)
            assert np.all([self.shape[i] == tensor2.shape[j] for i, j in
                           zip(dims1, dims2)]), \
                'The dimensions of the tensors are not compatible.'
            out = FullTensor(np.tensordot(self.data, tensor2.data,
                                          [dims1, dims2]))
        return out

    def tensordot_eval_diag(self, tensor2, dims1, dims2, diag_dims1,
                            diag_dims2, diag=False):
        '''
        Evaluate of the diagonal of a tensor obtained by contraction of two
        tensors.

        The contraction is performed along the dimensions dims1 for self and
        dims2 for tensor2, and the diagonal is evaluated according to the
        dimensions diag_dims1 for self and diag_dims2 for tensor2.

        The boolean diag indicates if the several diagonals are evaluated, for
        instance:
        - if diag is False, for order-4 tensors x and y,
        z = x.tensordot_eval_diag(y,[1,3],[2,3],2,0) returns an order-3 tensor
        z(i1,k,j2) = sum_{l1,l2} x(i1,l1,k,l2) y(k,j2,l1,l2)
        - if diag is True, for order-5 tensors x and y,
        z = x.tensordot_eval_diag(y,[1,3],[2,3],[0,2],[1,4]) returns an
        order-4 tensor
        z(k,l,i5,j1) = sum_{l1,l2} x(k,l1,l,l2,i5) y(j1,k,l1,l2,l)

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor of the product.
        dims1 : list or numpy.ndarray
            Dimensions of the first tensor for the contraction.
        dims2 : list or numpy.ndarray
            Dimensions of the second tensor for the contraction.
        diag_dims1 : list or numpy.ndarray
            Indices of the first tensor for the evaluation of the diagonal.
        diag_dims2 : list or numpy.ndarray
            Indices of the second tensor for the evaluation of the diagonal.
        diag : bool, optional
            Boolean enabling the evaluation of multiple diagonals. The default
            is False.

        Returns
        -------
        FullTensor
            The evaluated tensor.

        '''
        # Check if an outer product is asked with dims1 and dims2 equal to None
        if dims1 is None and dims2 is None:
            dims1 = []
            dims2 = []

        dims1 = np.atleast_1d(dims1)
        dims2 = np.atleast_1d(dims2)
        diag_dims1 = np.atleast_1d(diag_dims1)
        diag_dims2 = np.atleast_1d(diag_dims2)

        ind1 = np.arange(self.ndim)
        ind2 = ind1.size + np.arange(tensor2.ndim)
        if diag and diag_dims1.size != 0 and diag_dims2.size != 0:
            ind2[diag_dims2] = ind1[diag_dims1]
        elif not diag:
            ind1[diag_dims1] = ind1[diag_dims1[0]]
            ind2[diag_dims2] = ind1[diag_dims1[0]]
        if dims1.size != 0 and dims2.size != 0:
            ind2[dims2] = ind1[dims1]
        ind12 = np.concatenate((ind1, ind2))
        indexes = np.unique(ind12, return_index=True)[1]
        # Retain only the unique values without sorting the vector
        ind_out = np.array([ind12[index] for index in sorted(indexes)])
        # Remove from ind_out the contracted dimensions without sorting it
        ind_out = ind_out[[i not in np.atleast_1d(dims1) for i in ind_out]]
        array = np.einsum(self.data, ind1.tolist(), tensor2.data,
                          ind2.tolist(), ind_out.tolist())
        # alph = list(ascii_lowercase)
        # ind1 = [alph[i] for i in ind1]
        # ind2 = [alph[i] for i in ind2]
        # ind_out = [alph[i] for i in ind_out]
        # array = np.einsum(''.join(ind1) + ', ' + ''.join(ind2) + ' -> ' +
        #                   ''.join(ind_out), self.data, tensor2.data)
        return FullTensor(array)

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
        FullTensor
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

        # Numpy implementation
        # matrices = [np.array(x) for x in matrices]
        # data = self.numpy()
        # if self.order == 1:
        #     data = np.matmul(matrices[0], data)
        # else:
        #     k = 0
        #     for dim in np.nditer(dims):
        #         perm_dims = np.concatenate(([dim], np.arange(dim),
        #                                     np.arange(dim+1, self.order)))
        #         data = np.transpose(data, perm_dims)
        #         shape0 = np.array(data.shape)
        #         data = np.reshape(data, [shape0[0], np.prod(shape0[1:])])
        #         data = np.matmul(matrices[k], data)
        #         shape0[0] = matrices[k].shape[0]
        #         data = np.reshape(data, shape0)
        #         data = np.transpose(data, np.argsort(perm_dims))
        #         k += 1
        # return FullTensor(data)

        tensor = FullTensor(self)
        matrices = [FullTensor(x) for x in matrices]
        for i, dim in enumerate(dims):
            index = np.concatenate((
                tensap.fast_setdiff(np.arange(tensor.order), dim), [dim]))
            tensor = tensor.tensordot(matrices[i], dim, 1).itranspose(index)
        return tensor

    def tensor_vector_product(self, vectors, dims=None):
        '''
        Compute the contraction of the tensor with vectors.

        Compute the contraction of self with each vector contained in the list
        vectors along dimensions specified by dims. The operation is such that
        V[k] is contracted with the dims[k]-th dimension of self.

        Parameters
        ----------
        vectors : numpy.ndarray or list of numpy.ndarray
            The vectors to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        FullTensor
            The tensor after the contractions with the vectors.

        '''
        if dims is None:
            assert isinstance(vectors, list), 'vectors should be a list.'
            assert len(vectors) == self.order, \
                'len(vectors) must be self.order.'
            dims = np.arange(self.order)
        else:
            dims = np.array(dims)
            if not isinstance(vectors, list):
                vectors = [vectors]
            assert len(vectors) == dims.size, \
                'len(vectors) must be equal to dims.size.'

        vectors = [FullTensor(x, 2, [1, -1]) for x in vectors]
        return self.tensor_matrix_product(vectors, dims).squeeze(dims.tolist())

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
        out : FullTensor
            The diagonal of the contractions of the tensor with the matrices.

        '''
        if dims is None:
            assert isinstance(matrices, list), 'matrices should be a list.'
            assert len(matrices) == self.order, \
                'len(matrices) must be self.order.'
            dims = np.arange(self.order)
        else:
            dims = np.atleast_1d(dims)
            if not isinstance(matrices, list):
                matrices = [matrices]
            assert len(matrices) == dims.size, \
                'len(matrices) must be equal to dims.size.'

        matrices = [FullTensor(x) for x in matrices]
        ind = np.flip(np.argsort(dims))
        out = matrices[ind[0]].tensordot(self, 1, dims[ind[0]])
        for i in ind[1:]:
            out = matrices[i].tensordot_eval_diag(out, 1, dims[i]+1, 0, 0)

        # if out.order == 1:
        #     out = out.numpy()
        return out

    def tensor_diagonal_matrix_product(self, matrices, dims=None):
        '''
        Contract a FullTensor with matrices built from their diagonals.

        The second dimension of the matrix matrices[k] is contracted with the
        k-th dimension of self, with the indices k given in dims (if provided).

        Parameters
        ----------
        matrices : numpy.ndarray or list of numpy.ndarray
            The diagonals of the matrices to use in the product.
        dims : list or numpy.ndarray, optional
            Indices of the contractions. The default is None, indicating all
            the dimensions.

        Returns
        -------
        FullTensor
            The tensor after the contractions with the matrices.

        '''
        if dims is None:
            assert isinstance(matrices, list), 'matrices should be a list.'
            assert len(matrices) == self.order, \
                'len(matrices) must be self.order.'
            dims = range(self.order)
        else:
            dims = np.array(dims)
            if not isinstance(matrices, list):
                matrices = [matrices]
            assert len(matrices) == dims.size, \
                'len(matrices) must be equal to dims.size.'

        matrices = [FullTensor(np.diag(np.reshape(x, [-1])))
                    for x in matrices]
        return self.tensor_matrix_product(matrices, dims)

    def matricize(self, dims1, dims2=None):
        '''
        Return the matricization of the tensor.

        Parameters
        ----------
        dims1 : list or numpy.ndarray
            The dimensions of the tensor corresponding to the first dimension
            of the matricization.
        dims2 : list or numpy.ndarray, optional
            The dimensions of the tensor corresponding to the first dimension
            of the matricization. The default is None, for which they are
            deduced from dims1.

        Returns
        -------
        FullTensor
            The matricization of the tensor.

        '''
        dims1 = np.atleast_1d(dims1)
        if dims1.size == 1 and dims1 == -1:
            dims1 = np.array([self.order-1])
        if dims2 is None:
            dims2 = tensap.fast_setdiff(np.arange(self.order), dims1)
        else:
            dims2 = np.atleast_1d(dims2)
        shape1 = [self.shape[i] for i in dims1]
        shape2 = [self.shape[i] for i in dims2]

        tensor = FullTensor(self)
        tensor = tensor.transpose(np.concatenate((dims1, dims2)))
        tensor = tensor.reshape([np.prod(shape1), np.prod(shape2)])
        return FullTensor(tensor)

    def outer_product_eval_diag(self, tensor2, dims1, dims2, diag=False):
        '''
        Compute the diagonal of the outer product of two tensors.

        Equivalent to
        self.tensordot_eval_diag(tensor2, None, None, dims1, dims2, diag)

        Parameters
        ----------
        tensor2 : FullTensor
            The second tensor of the product.
        dims1 : list or numpy.ndarray
            Indices of the first tensor for the evaluation of the diagonal.
        dims2 : list or numpy.ndarray,
            Indices of the second tensor for the evaluation of the diagonal.
        diag : bool, optional
            Boolean enabling the evaluation of multiple diagonals. The default
            is False.

        Returns
        -------
        FullTensor
            The evaluated tensor.

        '''
        return self.tensordot_eval_diag(tensor2, None, None,
                                        dims1, dims2, diag)

    def principal_components(self, parameter=None):
        '''
        Compute the principal components of an order-2 tensor.

        Parameters
        ----------
        parameter : float or int, optional
            A parameter controlling the number of principal components.
            - If it is an integer, the number of principal components is the
            minimum between parameter and self.shape[0].
            - If it is a float smaller than 1, the number of principal
            components is determined such that ||x - VV'x||_F < t ||x||_F,
            with x the tensor, V the matrix of principal components, t the
            parameter, V' the transpose of the matrix V and ||.||_F the
            Frobenius norm.
            The default is self.shape[0].

        Returns
        -------
        principal_components : numpy.ndarray
            The principal components of the tensor.
        singular_values : numpy.ndarray
            The diagonal matrix of the associated singular values.

        '''
        assert self.order == 2, 'The order of the tensor must be 2.'
        if parameter is None or parameter > self.shape[0]:
            parameter = self.shape[0]

        if parameter < 1:
            truncator = tensap.Truncator(tolerance=parameter, max_rank=np.inf)
        else:
            truncator = tensap.Truncator(tolerance=0, max_rank=parameter)
        tensor = truncator.truncate(self)
        principal_components = tensor.space[0]
        singular_values = np.diag(tensor.core.data)
        return principal_components, singular_values

    def alpha_principal_components(self, alpha, parameter=None):
        '''
        Compute the alpha-principal components of a tensor.

        Return the principal components of the alpha-matricization
        M_alpha(self) of the tensor self of order d.

        See also the method principal_components.

        Parameters
        ----------
        alpha : int
            The index of the alpha-matricization.
        parameter : float or int, optional
            A parameter controlling the number of principal components.
            The default is M_alpha(self).shape[0].

        Returns
        -------
        principal_components : numpy.ndarray
            The principal components of the tensor.
        singular_values : numpy.ndarray
            The diagonal matrix of the associated singular values.

        '''
        principal_components, singular_values = \
            self.matricize(alpha).principal_components(parameter)
        return principal_components, singular_values

    def singular_values(self):
        '''
        Compute the higher-order singular values of a tensor (the collection
        of singular values of d different matricizations).

        Returns
        -------
        sin_val : numpy.ndarray or list of numpy.ndarray.
            The higher-order singular values.

        '''
        if self.order == 2:
            sin_val = np.linalg.svd(self.data, compute_uv=False)
        else:
            sin_val = []
            for ind in range(self.order):
                mat = self.matricize(ind)
                sin_val.append(np.linalg.svd(mat.data,
                                             compute_uv=False))
        return sin_val

    @staticmethod
    def create(generator, shape):
        '''
        Create a FullTensor of shape shape using a given generator.

        Parameters
        ----------
        generator : function
            Function generating a numpy.ndarray, given a shape.
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        FullTensor
            The created tensor.

        '''
        return FullTensor(generator(np.atleast_1d(shape)))

    @staticmethod
    def zeros(shape):
        '''
        Create a FullTensor of shape shape with entries equal to 0.

        Parameters
        ----------
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        FullTensor
            The created tensor.

        '''
        return FullTensor.create(np.zeros, shape)

    @staticmethod
    def ones(shape):
        '''
        Create a FullTensor of shape shape with entries equal to 1.

        Parameters
        ----------
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        FullTensor
            The created tensor.

        '''
        return FullTensor.create(np.ones, shape)

    @staticmethod
    def randn(shape):
        '''
        Create a FullTensor of shape shape with i.i.d. entries drawn according
        to the standard gaussian distribution.

        Parameters
        ----------
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        FullTensor
            The created tensor.

        '''
        return FullTensor.create(lambda x: np.random.randn(*x), shape)

    @staticmethod
    def rand(shape):
        '''
        Create a FullTensor of shape shape with i.i.d. entries drawn according
        to the uniform distribution on [0, 1].

        Parameters
        ----------
        shape : numpy.ndarray or list
            The shape of the tensor.

        Returns
        -------
        FullTensor
            The created tensor.

        '''
        return FullTensor.create(lambda x: np.random.rand(*x), shape)

    @staticmethod
    def diag(diag, order):
        '''
        Create a diagonal tensor x of order order, such that
        x[i, ..., i] = diag[i] for i = 0, ..., diag.size - 1.

        Parameters
        ----------
        diag : list or numpy.ndarray
            The diagonal of the tensor.
        order : int
            The order of the tensor.

        Returns
        -------
        FullTensor
            The created tensor.

        '''
        diag = np.atleast_1d(diag)

        ones_v = np.ones(order, dtype=int)
        shape_v = diag.size * ones_v

        data = np.zeros(shape_v)
        for ind, diag_ind in enumerate(diag):
            data[tuple(ind * ones_v)] = diag_ind
        return FullTensor(data)
