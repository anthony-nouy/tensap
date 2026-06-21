# Copyright (c) 2024
# This file is part of tensap (tensor approximation package).

"""
Module tucker_like_tensor.
"""

import numpy as np
import tensap


class TuckerLikeTensor:
    """
    Class TuckerLikeTensor.
    Algebraic tensors in Tucker-like tensor format.

    Attributes
    ----------
    core : tensap.FullTensor, tensap.TreeBasedTensor, tensap.DiagonalTensor,
           or tensap.SparseTensor
        The core of the Tucker tensor.
    space : tensap.TSpaceVectors or tensap.TSpaceOperators
        The tensor product space (factor matrices).
    order : int
        The order of the tensor.
    shape : numpy.ndarray
        The shape of the tensor in the full physical space.
    ranks : numpy.ndarray
        The Tucker ranks (subspace dimensions).
    is_orth : bool
        Boolean indicating if the representation of the tensor is orthogonal.
    """

    def __init__(self, core, space=None):
        """
        Constructor for the class TuckerLikeTensor.

        Parameters
        ----------
        core : tensap.FullTensor, tensap.TreeBasedTensor, tensap.DiagonalTensor,
               tensap.SparseTensor, or numpy.ndarray
            The core tensor. If a numpy.ndarray, cast to FullTensor.
            If a TreeBasedTensor and space is None, converts the
            TreeBasedTensor into Tucker-like format: for each dimension,
            the factor matrix is extracted from the corresponding leaf
            node (if active) or set to the identity (if inactive).
        space : tensap.TSpaceVectors, tensap.TSpaceOperators, or list, optional
            The tensor product space. If a list of arrays, cast to
            TSpaceVectors.
        """
        if isinstance(core, np.ndarray):
            core = tensap.FullTensor(core)

        if space is not None:
            if isinstance(space, (list, tuple)):
                space = tensap.TSpaceVectors(space)
            self.core = core
            self.space = space
        elif isinstance(core, tensap.TreeBasedTensor):
            y = core
            tree = y.tree
            space_list = [None] * y.order
            for mu in range(y.order):
                nod = tree.dim2ind[mu]
                idx = nod - 1
                if y.is_active_node[idx]:
                    space_list[mu] = y.tensors[idx].data
                else:
                    space_list[mu] = np.eye(y.shape[mu])
            self.core = y.tensors[0]
            self.space = tensap.TSpaceVectors(space_list)
            self.is_orth = y.is_orth
        else:
            self.core = core
            self.space = space

        self._update_properties()

    def _update_properties(self):
        """Update order, shape, ranks, and orthogonality flags."""
        self.order = self.space.order
        if isinstance(self.space, tensap.TSpaceOperators):
            self.shape = np.column_stack(
                [self.space.dims_out, self.space.dims_in]
            )
        else:
            self.shape = self.space.dims_out.copy()
        self.ranks = self.space.ranks.copy()
        self.is_orth = self.core.is_orth and self.space.is_orth

    # ---- Conversion ----

    def full(self):
        """
        Convert the object to a dense tensap.FullTensor.

        Returns
        -------
        tensap.FullTensor
            The Tucker tensor reconstructed as a dense tensor.
        """
        if isinstance(self.space, tensap.TSpaceOperators):
            target = np.column_stack(
                [self.space.dims_out, self.space.dims_in]
            )
            return self.vectorize().full().reshape(target.ravel('C'))
        mats = [s.reshape(-1, s.shape[2]) for s in self.space.spaces]
        return self.core.tensor_matrix_product(mats).full()

    def numpy(self):
        """
        Convert the TuckerLikeTensor to a dense numpy.ndarray.
        """
        return self.full().numpy()

    def tree_based_tensor(self):
        """Convert into a TreeBasedTensor.

        Returns
        -------
        tensap.TreeBasedTensor
        """
        if isinstance(self.space, tensap.TSpaceOperators):
            raise NotImplementedError(
                "Method not implemented for TSpaceOperators."
            )
        if isinstance(self.core, tensap.TreeBasedTensor):
            x = self.core
            x.tensors[x.tree.dim2ind] = self.space.spaces
            return x
        elif isinstance(self.core, tensap.FullTensor):
            tree = tensap.DimensionTree.trivial(self.order)
            tensors = [self.core]
            for dim in range(self.order):
                mat = self.space.spaces[dim].reshape(
                    self.shape[dim], self.ranks[dim]
                )
                tensors.append(tensap.FullTensor(mat))
            return tensap.TreeBasedTensor(tensors, tree)
        elif isinstance(self.core, tensap.DiagonalTensor):
            from functools import reduce
            x = reduce(lambda a, b: a + b, self.space.spaces)
            # TODO: proper DiagonalTensor case
            raise NotImplementedError(
                "DiagonalTensor core to TreeBasedTensor not yet implemented."
            )
        else:
            raise TypeError(
                f"Unsupported core type: {type(self.core)}"
            )

    # ---- Unary operators ----

    def __neg__(self):
        """Return the negative of the tensor."""
        return TuckerLikeTensor(-self.core, self.space)

    def __abs__(self):
        return TuckerLikeTensor(tensap.FullTensor(abs(self.core.data)), self.space)

    # ---- Arithmetic operators ----

    def __add__(self, other):
        if isinstance(other, TuckerLikeTensor):
            core_self, core_other = tensap.convert_tensors(self.core, other.core)
            new_core = core_self.cat(core_other)
            new_space = self.space.cat(other.space)
            return TuckerLikeTensor(new_core, new_space)
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, TuckerLikeTensor):
            return self + (-other)
        return NotImplemented

    def __mul__(self, other):
        if np.isscalar(other):
            return TuckerLikeTensor(self.core * other, self.space)
        return NotImplemented

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if np.isscalar(other):
            return TuckerLikeTensor(self.core / other, self.space)
        return NotImplemented

    # ---- Core operations ----

    def dot(self, tensor_2):
        """
        Return the inner product of two TuckerLikeTensors.

        Parameters
        ----------
        tensor_2 : TuckerLikeTensor

        Returns
        -------
        float
        """
        assert isinstance(tensor_2, TuckerLikeTensor), \
            "Argument must be a TuckerLikeTensor."
        core_self, core_other = tensap.convert_tensors(
            self.core, tensor_2.core
        )
        M = self.space.dot(tensor_2.space)
        core_2_projected = core_other.tensor_matrix_product(M)
        return core_self.dot(core_2_projected)

    def dot_with_rank_one_metric(self, tensor_2, matrices):
        """
        Weighted inner product: dot(self, times_matrix(tensor_2, matrices)).

        Parameters
        ----------
        tensor_2 : TuckerLikeTensor
        matrices : list of numpy.ndarray

        Returns
        -------
        float
        """
        tmp = tensor_2.tensor_matrix_product(matrices)
        return self.dot(tmp)

    def norm(self):
        """Compute the canonical Frobenius norm."""
        if self.is_orth:
            return self.core.norm()
        else:
            return np.sqrt(np.abs(self.dot(self)))

    def orth(self):
        """
        Orthogonalize the TuckerLikeTensor.

        Orthogonalises the factor matrices (TSpace) via QR, absorbs
        the transformation into the core, then orthogonalises the
        core itself (matching MATLAB behaviour).

        Returns
        -------
        TuckerLikeTensor
            The orthogonalized tensor (self, modified in-place).
        """
        dims = range(self.order)
        self.space, M = self.space.orth()
        self.core = self.core.tensor_matrix_product(M, dims)
        core_orth = self.core.orth()
        if isinstance(core_orth, tuple):
            self.core = core_orth[0]
        else:
            self.core = core_orth
        self._update_properties()
        return self

    # ---- Storage ----

    def storage(self):
        """Return the storage complexity."""
        return self.space.storage() + self.core.storage()

    def sparse_storage(self):
        """Return the number of non-zero entries."""
        return self.space.sparse_storage() + self.core.sparse_storage()

    def representation_rank(self):
        """Return the representation rank (prod of subspace dims)."""
        return int(np.prod(self.ranks))

    # ---- Contractions ----

    def tensor_matrix_product(self, matrices, dims=None):
        """
        Contract the Tucker tensor with matrices along physical dimensions.

        Left-multiplies the factor matrices by ``matrices[mu]`` along
        dimensions ``dims``.

        Parameters
        ----------
        matrices : list of numpy.ndarray
        dims : list of int, optional

        Returns
        -------
        TuckerLikeTensor
        """
        new_space = self.space.matrix_times_space(matrices, dims)
        return TuckerLikeTensor(self.core, new_space)

    def tensor_vector_product(self, vectors, dims=None):
        """
        Contract the tensor with vectors along given dimensions.

        Parameters
        ----------
        vectors : list of numpy.ndarray
        dims : list of int, optional

        Returns
        -------
        TuckerLikeTensor or numpy.ndarray (if all dims contracted)
        """
        assert isinstance(self.space, tensap.TSpaceVectors), \
            "The TSpace must be of TSpaceVectors type."

        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        if isinstance(vectors, np.ndarray):
            vectors = [vectors]
        vectors = [np.atleast_2d(v.ravel()) for v in vectors]

        # Left-multiply space by row vectors: v(1,N) @ space(N,1,R) -> (1,1,R)
        new_space = self.space.matrix_times_space(vectors, dims)
        # Reshape contracted factor mats from (1,1,R) to (R,1,1) for core
        new_spaces = list(new_space.spaces)
        for i, d in enumerate(dims):
            new_spaces[d] = new_spaces[d].transpose(2, 1, 0)
        xs = tensap.TSpaceVectors(new_spaces, is_orth=False)

        # Contract core with the transposed factor matrices
        xc = self.core.tensor_vector_product(xs.spaces, dims)

        if len(dims) != self.order:
            xs = xs.remove_space(dims)
            return TuckerLikeTensor(xc, xs)
        else:
            return xc

    def tensor_diag_matrix_product(self, M, dims=None):
        """
        Contract the tensor with diagonal matrices.

        Parameters
        ----------
        M : numpy.ndarray or list of numpy.ndarray
        dims : list of int, optional

        Returns
        -------
        TuckerLikeTensor
        """
        if isinstance(M, np.ndarray):
            M = [M]
        if dims is None:
            dims = range(self.order)
        diag_mats = [np.diag(m.ravel()) for m in M]
        return self.tensor_matrix_product(diag_mats, dims)

    # ---- Tensor algebra ----

    def cat(self, other):
        """
        Concatenate two TuckerLikeTensors.

        Uses block-diagonal concatenation for spaces and core concatenation
        along all dimensions.

        Parameters
        ----------
        other : TuckerLikeTensor

        Returns
        -------
        TuckerLikeTensor
        """
        new_core = self.core.cat(other.core)
        new_space = self.space.diag_cat(other.space)
        return TuckerLikeTensor(new_core, new_space)

    def kron(self, other):
        """
        Kronecker product of two TuckerLikeTensors.

        Parameters
        ----------
        other : TuckerLikeTensor

        Returns
        -------
        TuckerLikeTensor
        """
        new_core = self.core.kron(other.core)
        new_spaces = [
            np.kron(sx, sy)
            for sx, sy in zip(self.space.spaces, other.space.spaces)
        ]
        new_space = self.space.__class__(new_spaces, is_orth=False)
        return TuckerLikeTensor(new_core, new_space)

    # ---- Dimension manipulation ----

    def permute(self, dims):
        """
        Permute the dimensions of the tensor.

        Parameters
        ----------
        dims : array_like
            New ordering of dimensions.

        Returns
        -------
        TuckerLikeTensor
        """
        new_core = self.core.transpose(dims)
        new_space = self.space.permute(dims)
        return TuckerLikeTensor(new_core, new_space)

    def squeeze(self, dims=None):
        """
        Remove singleton dimensions.

        Parameters
        ----------
        dims : list of int, optional
            Dimensions to remove. Defaults to all singleton dims.

        Returns
        -------
        TuckerLikeTensor or numpy scalar
        """
        if dims is None:
            if self.shape.ndim == 2:
                dims = np.where(np.all(self.shape == 1, axis=1))[0]
            else:
                dims = np.where(self.shape == 1)[0]
        dims = np.atleast_1d(dims)

        if len(dims) == 0:
            return self

        # Contract singleton dimensions into the core
        xsp = [self.space.spaces[d].transpose(2, 1, 0) for d in dims]
        new_core = self.core.tensor_vector_product(xsp, dims)

        # Remove those dimensions from space
        keep_dims = [d for d in range(self.order) if d not in dims]
        new_space = self.space.keep_space(keep_dims)
        return TuckerLikeTensor(new_core, new_space)

    # ---- Subspace extraction ----

    def sub_tensor(self, *args):
        """
        Extract a subtensor.

        Parameters
        ----------
        *args : indices for each dimension (arrays or ':')
        """
        if isinstance(self.space, tensap.TSpaceVectors):
            for k in range(self.order):
                if not isinstance(args[k], str) or args[k] != ':':
                    idx = np.atleast_1d(args[k])
                    self.space.spaces[k] = self.space.spaces[k][idx, :, :]
        elif isinstance(self.space, tensap.TSpaceOperators):
            for k in range(self.order):
                I1 = args[2 * k]
                I2 = args[2 * k + 1]
                for n in range(self.space.ranks[k]):
                    if not (isinstance(I1, str) and I1 == ':'):
                        self.space.spaces[k][:, :, n] = \
                            self.space.spaces[k][I1, :, n]
                    if not (isinstance(I2, str) and I2 == ':'):
                        self.space.spaces[k][:, :, n] = \
                            self.space.spaces[k][:, I2, n]
        self.space = self.space.__class__(
            list(self.space.spaces), is_orth=False
        )
        self._update_properties()

    # ---- Operators specific ----

    def transpose(self):
        """Transpose of a TuckerLikeTensor of type Operator."""
        assert isinstance(self.space, tensap.TSpaceOperators), \
            "The TSpace must be of TSpaceOperators type."
        return TuckerLikeTensor(self.core, self.space.T)

    @property
    def T(self):
        return self.transpose()

    def ctranspose(self):
        """Conjugate transpose of a TuckerLikeTensor of type Operator."""
        assert isinstance(self.space, tensap.TSpaceOperators), \
            "The TSpace must be of TSpaceOperators type."
        return TuckerLikeTensor(self.core, self.space.H)

    @property
    def H(self):
        return self.ctranspose()

    def to_operator(self):
        """Convert a vector TuckerLikeTensor to an operator TuckerLikeTensor.

        Each basis vector v is replaced by the diagonal operator
        diag(v).
        """
        assert isinstance(self.space, tensap.TSpaceVectors), \
            "The TSpace must be of TSpaceVectors type."
        new_spaces = [s.copy() for s in self.space.spaces]
        for mu in range(self.order):
            n_out, n_in, r = new_spaces[mu].shape
            ops = np.zeros((n_out, n_out, r))
            for k in range(r):
                ops[:, :, k] = np.diag(new_spaces[mu][:, 0, k])
            new_spaces[mu] = ops
        new_space = tensap.TSpaceOperators(new_spaces, is_orth=False)
        return TuckerLikeTensor(self.core, new_space)

    def vectorize(self):
        """
        Vectorize an operator TuckerLikeTensor into a vector TuckerLikeTensor.

        Returns
        -------
        TuckerLikeTensor
        """
        new_space = self.space.vectorize()
        return TuckerLikeTensor(self.core, new_space)

    def unvectorize(self, sz):
        """
        Unvectorize a vector TuckerLikeTensor into an operator one.

        Parameters
        ----------
        sz : numpy.ndarray of shape (2, K)

        Returns
        -------
        TuckerLikeTensor
        """
        new_space = self.space.unvectorize(sz)
        return TuckerLikeTensor(self.core, new_space)

    # ---- Evaluation ----

    def eval_diag(self, dims=None):
        """
        Extract the diagonal of the tensor.

        Parameters
        ----------
        dims : list of int, optional

        Returns
        -------
        numpy.ndarray or TuckerLikeTensor
        """
        if dims is None:
            dims = list(range(self.order))
        else:
            dims = np.atleast_1d(dims).tolist()

        if len(dims) == 1:
            return self

        if isinstance(self.core, tensap.DiagonalTensor) and \
                isinstance(self.space, tensap.TSpaceVectors):
            s = self.space.spaces[dims[0]]
            for k in dims[1:]:
                s = s * self.space.spaces[k]
            if len(dims) == self.order:
                return s[:, 0, :] @ self.core.data
            new_spaces = [s] + [self.space.spaces[k] for k in
                                range(self.order) if k not in dims]
            ns = self.space.__class__(new_spaces, is_orth=False)
            n_order = 1 + (self.order - len(dims))
            new_core = tensap.DiagonalTensor(
                self.core.data, order=n_order
            )
            return TuckerLikeTensor(new_core, ns)

        # For general case: contract core with all spaces to get full tensor,
        # then extract diagonal.  For TSpaceOperators, this yields the
        # diagonal of the vectorized (flattened) operator, consistent with
        # MATLAB's generic else branch (line 651-653).
        s = self.core.tensor_matrix_product(
            [sp[:, 0, :] for sp in self.space.spaces]
        )
        return s.eval_diag(dims)

    def singular_values(self):
        """
        Return the singular values of the tensor.

        Returns
        -------
        numpy.ndarray or list
        """
        if self.order == 2:
            self.orth()
            return np.linalg.svd(self.core.data, compute_uv=False)
        else:
            if isinstance(self.core, tensap.FullTensor):
                self.orth()
                return self.core.singular_values()
            else:
                raise NotImplementedError(
                    f"singular_values not implemented for "
                    f"core type {type(self.core)}."
                )

    # ---- Normalization ----

    def normalize_basis(self):
        """
        Normalize the elements of the bases of subspaces.
        """
        N = self.space.dot(self.space)
        N = [np.sqrt(np.diag(n)) for n in N]
        diag_mats = [np.diag(n) for n in N]
        self.core = self.core.tensor_matrix_product(diag_mats)
        inv_mats = [np.diag(1.0 / n) for n in N]
        self.space = self.space.space_times_matrix(inv_mats)
        return self

    # ---- Static constructors ----

    @staticmethod
    def create(generator, ranks, shape):
        """
        Create a TuckerLikeTensor using a given generator.

        Parameters
        ----------
        generator : callable
            Function generating a numpy array from a shape tuple.
        ranks : array_like
            Tucker ranks (ranks of the core for each dimension).
        shape : array_like
            Physical shape of the tensor.

        Returns
        -------
        TuckerLikeTensor
        """
        ranks = np.atleast_1d(np.asarray(ranks, dtype=int))
        shape = np.atleast_1d(np.asarray(shape, dtype=int))
        space = tensap.TSpaceVectors.create(generator, shape, ranks)
        core = tensap.FullTensor(generator(tuple(ranks)))
        return TuckerLikeTensor(core, space)

    @staticmethod
    def zeros(ranks, shape):
        return TuckerLikeTensor.create(np.zeros, ranks, shape)

    @staticmethod
    def ones(ranks, shape):
        return TuckerLikeTensor.create(np.ones, ranks, shape)

    @staticmethod
    def rand(ranks, shape):
        return TuckerLikeTensor.create(
            lambda x: np.random.rand(*x), ranks, shape
        )

    @staticmethod
    def randn(ranks, shape):
        return TuckerLikeTensor.create(
            lambda x: np.random.randn(*x), ranks, shape
        )

    @staticmethod
    def eye(shape):
        """
        Construct the identity operator in TuckerLikeTensor format.

        Parameters
        ----------
        shape : array_like
            Size along each dimension.

        Returns
        -------
        TuckerLikeTensor
        """
        shape = np.atleast_1d(np.asarray(shape, dtype=int))
        d = len(shape)
        core = tensap.DiagonalTensor(np.array([1.0]), order=d)
        space = tensap.TSpaceOperators.eye(shape)
        return TuckerLikeTensor(core, space)

    # ---- Display ----

    def __repr__(self):
        if self.shape.ndim == 2:
            shape_str = "x".join(
                f"{out}x{inn}" for out, inn in self.shape
            )
        else:
            shape_str = "x".join(map(str, self.shape))
        return (
            "<{} TuckerLikeTensor:\n"
            + "\torder = {},\n"
            + "\tshape = {},\n"
            + "\tranks = {},\n"
            + "\tis_orth = {}>"
        ).format(
            shape_str,
            self.order,
            self.shape,
            self.ranks,
            self.is_orth,
        )
