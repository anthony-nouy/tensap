import numpy as np

from .tensor_space import TSpace


class TSpaceVectors(TSpace):
    """Tensor product of vector spaces.

    Each space must have ``dims_in == 1`` (shape ``(N, 1, Rank)``).
    """

    def __init__(self, spaces, is_orth=False):
        super().__init__(spaces, is_orth)
        if not np.all(self.dims_in == 1):
            raise ValueError(
                "TSpaceVectors requires dims_in == 1 for all dimensions."
            )

    # ---- Vector-specific methods ----

    def diag_cat(self, other):
        """Block-diagonal concatenation of two TSpaceVectors.

        For each dimension ``mu``, builds the block-diagonal matrix
        ``[Vx, 0; 0, Vy]``, increasing both the physical dimension and
        the rank.

        Parameters
        ----------
        other : TSpaceVectors

        Returns
        -------
        TSpaceVectors
        """
        if self.order != other.order:
            raise ValueError("The spaces must have the same order")

        new_spaces = []
        for mu in range(self.order):
            sx = self.spaces[mu]    # (Nx, 1, Rx)
            sy = other.spaces[mu]   # (Ny, 1, Ry)
            nx, _, rx = sx.shape
            ny, _, ry = sy.shape
            block = np.zeros((nx + ny, 1, rx + ry))
            block[:nx, :, :rx] = sx
            block[nx:, :, rx:] = sy
            new_spaces.append(block)

        return TSpaceVectors(new_spaces, is_orth=False)

    def dot_with_metrics(self, y, A, order=None):
        """Inner products of basis vectors with a metric operator.

        Computes ``M[mu](i1, i2, i3) = x[:,i2]' * A[:,:,i1] * y[:,i3]``
        for each dimension ``mu``.

        Parameters
        ----------
        y : TSpaceVectors
        A : TSpaceOperators
            Metric operator (must be square: ``dims_out == dims_in``).
        order : list of int, optional
            Dimensions to process. Defaults to all.

        Returns
        -------
        list of numpy.ndarray
            Each array has shape ``(A_rank, self_rank, y_rank)``.
        """
        if order is None:
            order = range(self.order)

        M = []
        for mu in order:
            if self.dims_out[mu] != A.dims_out[mu]:
                raise ValueError(
                    f"Output dim mismatch at dim {mu} "
                    f"between self and A."
                )
            if A.dims_out[mu] != A.dims_in[mu]:
                raise ValueError(
                    f"A must be square at dim {mu} to be a metric."
                )
            if self.dims_out[mu] != y.dims_out[mu]:
                raise ValueError(
                    f"Output dim mismatch at dim {mu} "
                    f"between self and y."
                )

            X = self.spaces[mu][:, 0, :]   # (N, Rx)
            Op = A.spaces[mu]               # (N, N, Ra)
            Y = y.spaces[mu][:, 0, :]       # (N, Ry)

            Mmu = np.einsum("ix, ija, jz -> a x z", X, Op, Y)
            M.append(Mmu)

        return M

    def eval_at_indices(self, I):
        """Evaluate basis vectors at given indices.

        Selects rows of each subspace basis according to ``I``.

        Parameters
        ----------
        I : ndarray of shape (N_points, order)
            Indices for each dimension.

        Returns
        -------
        TSpaceVectors
        """
        I = np.asarray(I)
        new_spaces = []
        for mu in range(self.order):
            new_spaces.append(self.spaces[mu][I[:, mu], :, :])
        return TSpaceVectors(new_spaces, is_orth=False)

    def unvectorize(self, sz, dims=None, P=None):
        """Convert a TSpaceVectors into a TSpaceOperators.

        Each basis vector (column) is reshaped into an operator of
        size ``sz[:, mu]``.

        Parameters
        ----------
        sz : ndarray of shape (2, K)
            Target sizes for the operators. ``sz[0, mu]`` is the output
            dimension, ``sz[1, mu]`` is the input dimension.
        dims : list of int, optional
            Dimensions to convert. Defaults to all.
        P : list, optional
            Sparsity pattern (not yet implemented).

        Returns
        -------
        TSpaceOperators
        """
        from .tensor_space_operators import TSpaceOperators

        sz = np.asarray(sz)
        if dims is None:
            dims = range(self.order)

        new_spaces = list(self.spaces)
        for mu in dims:
            vectors = self.spaces[mu][:, 0, :]  # (N, R)
            r = vectors.shape[1]
            ops = np.zeros((sz[0, mu], sz[1, mu], r))
            for k in range(r):
                ops[:, :, k] = vectors[:, k].reshape(sz[0, mu], sz[1, mu])
            new_spaces[mu] = ops

        return TSpaceOperators(new_spaces, is_orth=False)

    def to_operators(self):
        """Convert a TSpaceVectors into a TSpaceOperators.

        Each basis vector becomes a column operator of shape ``(N, 1)``.
        The data is identical; the type changes.

        Returns
        -------
        TSpaceOperators
        """
        from .tensor_space_operators import TSpaceOperators

        return TSpaceOperators(self.spaces, is_orth=self.is_orth)

    # ---- Static constructors ----

    @staticmethod
    def create(generator, sz, dim=None):
        """Create a TSpaceVectors from a generator function.

        Parameters
        ----------
        generator : callable((n, m)) -> ndarray
            Function that generates an (n, m) matrix from a tuple (n, m).
        sz : array_like
            Output dimension of each subspace.
        dim : array_like, optional
            Rank (number of basis vectors) of each subspace.
            Defaults to ones.
        """
        sz = np.asarray(sz, dtype=int).ravel()
        if dim is None:
            dim = np.ones_like(sz)
        dim = np.asarray(dim, dtype=int).ravel()

        spaces = [
            generator((s, d)).reshape(s, 1, d)
            for s, d in zip(sz, dim)
        ]
        return TSpaceVectors(spaces, is_orth=False)

    @staticmethod
    def zeros(sz, dim=None):
        return TSpaceVectors.create(np.zeros, sz, dim)

    @staticmethod
    def ones(sz, dim=None):
        return TSpaceVectors.create(np.ones, sz, dim)

    @staticmethod
    def rand(sz, dim=None):
        return TSpaceVectors.create(
            lambda x: np.random.rand(*x), sz, dim
        )

    @staticmethod
    def randn(sz, dim=None):
        return TSpaceVectors.create(
            lambda x: np.random.randn(*x), sz, dim
        )

    @staticmethod
    def eye(sz, dim=None):
        """Create canonical basis vectors (identity matrix columns).

        If dim is None, dim = sz (full canonical basis).
        """
        sz = np.asarray(sz, dtype=int)
        if dim is None:
            dim = sz
        return TSpaceVectors.create(lambda x: np.eye(*x), sz, dim)
