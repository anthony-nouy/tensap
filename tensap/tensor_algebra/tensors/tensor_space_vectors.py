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

    # ---- Vector-specific methods (stubs) ----

    def diag_cat(self, other):
        """Block-diagonal concatenation of two TSpaceVectors."""
        raise NotImplementedError("TSpaceVectors.diag_cat")

    def dot_with_metrics(self, y, A, order=None):
        """Inner products of basis vectors with a metric operator."""
        raise NotImplementedError("TSpaceVectors.dot_with_metrics")

    def eval_at_indices(self, I):
        """Evaluate basis vectors at given indices."""
        raise NotImplementedError("TSpaceVectors.eval_at_indices")

    def unvectorize(self, sz, dims=None, P=None):
        """Convert a TSpaceVectors into a TSpaceOperators."""
        raise NotImplementedError("TSpaceVectors.unvectorize")

    def to_operators(self):
        """Convert to TSpaceOperators."""
        raise NotImplementedError("TSpaceVectors.to_operators")

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
