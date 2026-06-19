import numpy as np

from .tensor_space import TSpace
from .tensor_space_vectors import TSpaceVectors


class TSpaceOperators(TSpace):
    """Tensor product of operator spaces.

    Each space must have ``dims_in > 1`` for all dimensions (strict
    operator spaces).
    """

    def __init__(self, spaces, is_orth=False):
        super().__init__(spaces, is_orth)
        if not np.all(self.dims_in > 1):
            raise ValueError(
                "TSpaceOperators requires dims_in > 1 for all dimensions."
            )

    # ---- Operator-specific methods ----

    def mtimes(self, other, dims=None):
        """Multiply two tensor spaces.

        ``operator * operator`` gives an operator space.
        ``operator * vector`` gives a vector space.

        Parameters
        ----------
        other : TSpace
        dims : list of int, optional
            Dimensions to multiply. Defaults to all.
        """
        if dims is None:
            dims = range(self.order)
        else:
            dims = np.atleast_1d(dims)

        for mu in dims:
            if self.dims_in[mu] != other.dims_out[mu]:
                raise ValueError(
                    f"Dimension mismatch at dim {mu}: "
                    f"self.dims_in[{mu}]={self.dims_in[mu]} != "
                    f"other.dims_out[{mu}]={other.dims_out[mu]}."
                )

        new_spaces = []
        for mu in dims:
            res = np.einsum(
                "okx, kiy -> oixy",
                self.spaces[mu],
                other.spaces[mu],
            )
            new_spaces.append(res.reshape(res.shape[0], res.shape[1], -1))

        result_dims_in = np.array([s.shape[1] for s in new_spaces])
        if np.all(result_dims_in == 1):
            return TSpaceVectors(new_spaces, is_orth=False)
        return TSpaceOperators(new_spaces, is_orth=False)

    def transpose(self, conjugate=False):
        """Swap the physical axes (N_out, N_in) for all operators.

        Parameters
        ----------
        conjugate : bool, optional
            If True, also apply complex conjugation.
        """
        new_spaces = []
        for mu in range(self.order):
            t = self.spaces[mu].transpose(1, 0, 2)
            if conjugate:
                t = np.conj(t)
            new_spaces.append(t)
        return TSpaceOperators(new_spaces, is_orth=self.is_orth)

    @property
    def T(self):
        """Standard transpose."""
        return self.transpose(conjugate=False)

    @property
    def H(self):
        """Hermitian (conjugate) transpose."""
        return self.transpose(conjugate=True)

    def vectorize(self, dims=None):
        """Convert a TSpaceOperators into a TSpaceVectors.

        Each basis operator is flattened into a column vector.
        """
        raise NotImplementedError("TSpaceOperators.vectorize")

    # ---- Static constructors ----

    @staticmethod
    def create(generator, sz1, sz2=None, dims=None):
        """Create a TSpaceOperators from a generator function.

        Parameters
        ----------
        generator : callable((n, m)) -> ndarray
            Function that generates an (n, m) matrix from a tuple (n, m).
        sz1 : array_like
            Output dimension of each subspace.
        sz2 : array_like, optional
            Input dimension of each subspace. Defaults to sz1.
        dims : array_like, optional
            Rank (number of basis operators) of each subspace.
            Defaults to ones.
        """
        sz1 = np.asarray(sz1, dtype=int).ravel()
        d = len(sz1)
        if sz2 is None:
            sz2 = sz1.copy()
        sz2 = np.asarray(sz2, dtype=int).ravel()
        if dims is None:
            dims = np.ones(d, dtype=int)
        dims = np.asarray(dims, dtype=int).ravel()

        spaces = []
        for mu in range(d):
            ops = np.stack(
                [generator((sz1[mu], sz2[mu])) for _ in range(dims[mu])],
                axis=2,
            )
            spaces.append(ops)
        return TSpaceOperators(spaces, is_orth=False)

    @staticmethod
    def zeros(sz1, sz2=None, dims=None):
        return TSpaceOperators.create(np.zeros, sz1, sz2, dims)

    @staticmethod
    def ones(sz1, sz2=None, dims=None):
        return TSpaceOperators.create(np.ones, sz1, sz2, dims)

    @staticmethod
    def rand(sz1, sz2=None, dims=None):
        return TSpaceOperators.create(
            lambda x: np.random.rand(*x), sz1, sz2, dims
        )

    @staticmethod
    def randn(sz1, sz2=None, dims=None):
        return TSpaceOperators.create(
            lambda x: np.random.randn(*x), sz1, sz2, dims
        )

    @staticmethod
    def eye(sz1, sz2=None, dims=None):
        """Create identity operators.

        If dims is None, dims = 1 (one identity per subspace).
        """
        sz1 = np.asarray(sz1, dtype=int)
        if sz2 is None:
            sz2 = sz1.copy()
        if dims is None:
            dims = np.ones(len(sz1), dtype=int)
        return TSpaceOperators.create(
            lambda x: np.eye(*x), sz1, sz2, dims
        )
