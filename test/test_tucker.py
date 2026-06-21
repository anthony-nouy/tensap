"""Tests for TuckerLikeTensor."""

import numpy as np
import pytest
import tensap


class TestTuckerLikeTensor:
    @staticmethod
    def _make_tucker(order=2, seed=0):
        np.random.seed(seed)
        ranks = np.random.randint(2, 5, size=order)
        shape = np.random.randint(5, 10, size=order)
        return tensap.TuckerLikeTensor.randn(ranks, shape)

    def test_construction_from_list(self):
        core = tensap.FullTensor.randn([3, 4])
        space = [np.random.rand(5, 3), np.random.rand(6, 4)]
        t = tensap.TuckerLikeTensor(core, space)
        assert t.order == 2
        assert np.all(t.shape == [5, 6])
        assert np.all(t.ranks == [3, 4])
        assert isinstance(t.space, tensap.TSpaceVectors)

    def test_construction_from_tspace(self):
        core = tensap.FullTensor.randn([3, 4])
        space = tensap.TSpaceVectors.rand([5, 6], [3, 4])
        t = tensap.TuckerLikeTensor(core, space)
        assert t.order == 2
        assert np.all(t.shape == [5, 6])

    def test_construction_ndarray_core(self):
        core = np.random.rand(3, 4)
        space = [np.random.rand(5, 3), np.random.rand(6, 4)]
        t = tensap.TuckerLikeTensor(core, space)
        assert isinstance(t.core, tensap.FullTensor)

    def test_full(self):
        t = self._make_tucker(2)
        U = [s.reshape(-1, s.shape[2]) for s in t.space.spaces]
        expected = U[0] @ t.core.data @ U[1].T
        assert np.allclose(t.full().data, expected)

    def test_numpy(self):
        t = self._make_tucker(3)
        assert np.allclose(t.numpy(), t.full().numpy())

    def test_dot(self):
        t = self._make_tucker(2)
        d = t.dot(t)
        assert np.isclose(d, np.linalg.norm(t.numpy()) ** 2)

    def test_norm(self):
        t = self._make_tucker(2)
        assert np.isclose(t.norm(), np.linalg.norm(t.numpy()))

    def test_orth(self):
        t = self._make_tucker(2)
        expected_norm = t.norm()
        t.orth()
        assert t.space.is_orth
        assert np.isclose(t.norm(), expected_norm)

    def test_neg(self):
        t = self._make_tucker(2)
        assert np.allclose((-t).numpy(), -t.numpy())

    def test_mul_scalar(self):
        t = self._make_tucker(2)
        assert np.allclose((t * 2.5).numpy(), 2.5 * t.numpy())

    def test_rmul_scalar(self):
        t = self._make_tucker(2)
        assert np.allclose((3.0 * t).numpy(), 3.0 * t.numpy())

    def test_truediv_scalar(self):
        t = self._make_tucker(2)
        assert np.allclose((t / 2.0).numpy(), t.numpy() / 2.0)

    def test_add(self):
        np.random.seed(0)
        t1 = tensap.TuckerLikeTensor.randn([2, 3], [10, 20])
        t2 = tensap.TuckerLikeTensor.randn([2, 3], [10, 20])
        result = t1 + t2
        assert np.allclose(result.numpy(), t1.numpy() + t2.numpy())

    def test_sub(self):
        np.random.seed(0)
        t1 = tensap.TuckerLikeTensor.randn([2, 3], [10, 20])
        t2 = tensap.TuckerLikeTensor.randn([2, 3], [10, 20])
        result = t1 - t2
        assert np.allclose(result.numpy(), t1.numpy() - t2.numpy())

    def test_tensor_matrix_product(self):
        t = self._make_tucker(2)
        shape = t.shape
        A = [np.random.rand(7, shape[0]), np.random.rand(8, shape[1])]
        result = t.tensor_matrix_product(A)
        expected = A[0] @ t.numpy() @ A[1].T
        assert np.allclose(result.numpy(), expected)

    def test_tensor_vector_product(self):
        t = self._make_tucker(2)
        v = [np.random.rand(t.shape[0]), np.random.rand(t.shape[1])]
        result = t.tensor_vector_product(v)
        expected = np.tensordot(np.tensordot(t.numpy(), v[0], (0, 0)),
                                v[1], (0, 0))
        assert np.allclose(result, expected)

    def test_tensor_diag_matrix_product(self):
        t = self._make_tucker(2)
        M = [np.random.rand(t.shape[0]), np.random.rand(t.shape[1])]
        result = t.tensor_diag_matrix_product(M)
        expected = np.diag(M[0]) @ t.numpy() @ np.diag(M[1])
        assert np.allclose(result.numpy(), expected)

    def test_permute(self):
        t = self._make_tucker(3)
        perm = [2, 0, 1]
        result = t.permute(perm)
        assert np.all(result.shape == t.shape[perm])
        assert np.allclose(result.numpy(), np.transpose(t.numpy(), perm))

    def test_squeeze(self):
        np.random.seed(0)
        t = tensap.TuckerLikeTensor.randn([2, 3, 4], [5, 1, 7])
        result = t.squeeze()
        assert np.all(result.shape == [5, 7])
        assert np.allclose(result.numpy(), t.numpy().squeeze())

    def test_squeeze_noop(self):
        t = self._make_tucker(2)
        result = t.squeeze()
        assert np.all(result.shape == t.shape)

    def test_cat(self):
        np.random.seed(0)
        t1 = tensap.TuckerLikeTensor.randn([2, 3], [5, 6])
        t2 = tensap.TuckerLikeTensor.randn([2, 3], [7, 8])
        result = t1.cat(t2)
        assert np.all(result.shape == [12, 14])
        expected = np.zeros([12, 14])
        expected[:5, :6] = t1.numpy()
        expected[5:, 6:] = t2.numpy()
        assert np.allclose(result.numpy(), expected)

    def test_kron(self):
        np.random.seed(0)
        t1 = tensap.TuckerLikeTensor.randn([2, 3], [5, 6])
        t2 = tensap.TuckerLikeTensor.randn([2, 3], [7, 8])
        result = t1.kron(t2)
        assert np.all(result.shape == [35, 48])
        assert np.allclose(result.numpy(), np.kron(t1.numpy(), t2.numpy()))

    def test_storage(self):
        t = self._make_tucker(2)
        assert t.storage() > 0

    def test_sparse_storage(self):
        t = self._make_tucker(2)
        assert t.sparse_storage() >= 0

    def test_representation_rank(self):
        t = self._make_tucker(3)
        assert t.representation_rank() == np.prod(t.ranks)

    def test_zeros(self):
        t = tensap.TuckerLikeTensor.zeros([2, 3], [5, 6])
        assert np.allclose(t.numpy(), np.zeros([5, 6]))

    def test_ones(self):
        t = tensap.TuckerLikeTensor.ones([2, 3], [5, 6])
        assert np.allclose(t.numpy(), 6 * np.ones([5, 6]))

    def test_eye(self):
        t = tensap.TuckerLikeTensor.eye([5, 6])
        assert isinstance(t.space, tensap.TSpaceOperators)
        assert t.shape.ndim == 2
        assert np.all(t.shape == [[5, 5], [6, 6]])

    def test_operator_shape(self):
        rng = np.random.RandomState(0)
        core = tensap.FullTensor(rng.rand(2, 3))
        sz1 = np.array([5, 7])
        sz2 = np.array([3, 4])
        spaces = [rng.rand(sz1[0], sz2[0], 2), rng.rand(sz1[1], sz2[1], 3)]
        op_space = tensap.TSpaceOperators(spaces)
        t = tensap.TuckerLikeTensor(core, op_space)
        assert t.shape.ndim == 2
        assert np.all(t.shape[:, 0] == sz1)
        assert np.all(t.shape[:, 1] == sz2)

    def test_operator_full_reshape(self):
        rng = np.random.RandomState(0)
        core = tensap.FullTensor(rng.rand(2, 3))
        sz1 = np.array([5, 7])
        sz2 = np.array([3, 4])
        spaces = [rng.rand(sz1[0], sz2[0], 2), rng.rand(sz1[1], sz2[1], 3)]
        op_space = tensap.TSpaceOperators(spaces)
        t = tensap.TuckerLikeTensor(core, op_space)
        tf = t.full()
        expected_shape = np.array([sz1[0], sz2[0], sz1[1], sz2[1]])
        assert np.all(tf.shape == expected_shape)

    def test_vector_shape(self):
        rng = np.random.RandomState(0)
        core = tensap.FullTensor(rng.rand(2, 3))
        shape = [5, 6]
        space = tensap.TSpaceVectors([rng.rand(shape[0], 1, 2),
                                       rng.rand(shape[1], 1, 3)])
        t = tensap.TuckerLikeTensor(core, space)
        assert t.shape.ndim == 1
        assert np.all(t.shape == shape)

    def test_tree_based_tensor(self):
        t = self._make_tucker(2)
        tbt = t.tree_based_tensor()
        assert isinstance(tbt, tensap.TreeBasedTensor)

    def test_truncate_tight(self):
        t = self._make_tucker(3)
        expected_norm = t.norm()
        space, M = t.space.truncate(tol=1e-16)
        t2 = tensap.TuckerLikeTensor(t.core, space)
        t2.core = t2.core.tensor_matrix_product(M)
        assert np.isclose(t2.norm(), expected_norm)
        assert space.is_orth

    def test_truncate_reduces_rank(self):
        t = self._make_tucker(3)
        space, M = t.space.truncate(tol=0.9)
        assert np.all(space.ranks <= t.ranks)
        assert np.any(space.ranks < t.ranks)

    def test_abs(self):
        t = self._make_tucker(2)
        assert np.allclose(abs(t).core.data, abs(t.core.data))

    def test_radd(self):
        t = self._make_tucker(2)
        result = 0 + t
        assert np.allclose(result.numpy(), t.numpy())

    def test_dot_with_rank_one_metric(self):
        rng = np.random.RandomState(0)
        shape = [5, 6]
        t = tensap.TuckerLikeTensor.randn([2, 3], shape)
        t2 = tensap.TuckerLikeTensor.randn([2, 3], shape)
        # Square matrices preserve output dimensions
        matrices = [rng.randn(shape[0], shape[0]), rng.randn(shape[1], shape[1])]
        d = t.dot_with_rank_one_metric(t2, matrices)
        expected = np.sum(t.numpy() * (matrices[0] @ t2.numpy() @ matrices[1].T))
        assert np.isclose(d, expected)

    def test_sub_tensor(self):
        t = self._make_tucker(2)
        original = t.numpy().copy()
        t.sub_tensor(np.array([0, 1]), np.array([0, 1, 2]))
        assert np.allclose(t.numpy(), original[:2, :3])
        t2 = self._make_tucker(2)
        t2.sub_tensor(":", np.array([0, 2]))
        assert np.allclose(t2.numpy(), original[:, [0, 2]])

    def test_transpose_operator(self):
        t = tensap.TuckerLikeTensor.eye([5, 6])
        tt = t.transpose()
        assert isinstance(tt.space, tensap.TSpaceOperators)
        assert np.all(tt.space.dims_out == t.space.dims_in)
        assert np.all(tt.space.dims_in == t.space.dims_out)

    def test_T_property(self):
        t = tensap.TuckerLikeTensor.eye([5, 6])
        assert np.all(t.T.space.dims_out == t.space.dims_in)

    def test_ctranspose(self):
        t = tensap.TuckerLikeTensor.eye([5, 6])
        th = t.ctranspose()
        assert isinstance(th.space, tensap.TSpaceOperators)
        assert np.all(th.space.dims_out == t.space.dims_in)
        assert np.all(th.space.dims_in == t.space.dims_out)

    def test_H_property(self):
        t = tensap.TuckerLikeTensor.eye([5, 6])
        assert np.all(t.H.space.dims_out == t.space.dims_in)

    def test_transpose_error_vector(self):
        t = self._make_tucker(2)
        try:
            t.transpose()
            assert False
        except AssertionError:
            pass

    def test_ctranspose_error_vector(self):
        t = self._make_tucker(2)
        try:
            t.ctranspose()
            assert False
        except AssertionError:
            pass

    def test_to_operator(self):
        rng = np.random.RandomState(0)
        t = tensap.TuckerLikeTensor.randn([2, 3], [5, 6])
        op = t.to_operator()
        assert isinstance(op.space, tensap.TSpaceOperators)
        assert np.all(op.space.dims_out == t.shape)
        assert np.all(op.space.dims_in == t.shape)

    def test_to_operator_error_operator(self):
        t = tensap.TuckerLikeTensor.eye([5, 6])
        try:
            t.to_operator()
            assert False
        except AssertionError:
            pass

    def test_vectorize_unvectorize_roundtrip(self):
        rng = np.random.RandomState(0)
        # Create a TuckerLikeTensor with TSpaceOperators space
        sz1 = np.array([5, 7])
        sz2 = np.array([3, 4])
        ops = tensap.TSpaceOperators.rand(sz1, sz2, nb_vectors=[2, 3])
        core = tensap.FullTensor.randn([2, 3])
        op = tensap.TuckerLikeTensor(core, ops)
        vec = op.vectorize()
        assert isinstance(vec.space, tensap.TSpaceVectors)
        sz = np.vstack([sz1, sz2])
        op2 = vec.unvectorize(sz)
        assert np.allclose(op.full().numpy(), op2.full().numpy())

    def test_eval_diag(self):
        rng = np.random.RandomState(0)
        t = tensap.TuckerLikeTensor.randn([3, 3], [5, 5])
        d = t.eval_diag()
        assert np.allclose(d, np.diag(t.numpy()))

    def test_singular_values_order2(self):
        rng = np.random.RandomState(0)
        t = tensap.TuckerLikeTensor.randn([5, 5], [10, 10])
        sv = t.singular_values()
        assert sv.shape == (5,)
        # singular_values returns SVD of the core after orth()
        expected = np.linalg.svd(t.core.data, compute_uv=False)
        assert np.allclose(sv, expected)

    def test_singular_values_order3(self):
        rng = np.random.RandomState(0)
        t = tensap.TuckerLikeTensor.randn([2, 3, 4], [5, 6, 7])
        sv = t.singular_values()
        assert len(sv) == 3
        assert sv[0].shape == (2,)

    def test_normalize_basis(self):
        rng = np.random.RandomState(0)
        t = tensap.TuckerLikeTensor.randn([2, 3], [5, 6])
        expected_norm = t.norm()
        t.normalize_basis()
        for mu in range(t.order):
            n = np.linalg.norm(t.space.spaces[mu][:, 0, :], axis=0)
            assert np.allclose(n, np.ones(t.ranks[mu]), atol=1e-10)
        assert np.isclose(t.norm(), expected_norm)

    def test_static_create(self):
        t = tensap.TuckerLikeTensor.create(
            lambda x: np.ones(x), [2, 3], [5, 6])
        assert np.allclose(t.numpy(), 6 * np.ones([5, 6]))

    def test_static_rand(self):
        t = tensap.TuckerLikeTensor.rand([2, 3], [5, 6])
        assert t.shape.tolist() == [5, 6]
        assert t.ranks.tolist() == [2, 3]

    def test_static_randn(self):
        t = tensap.TuckerLikeTensor.randn([2, 3], [5, 6])
        assert t.shape.tolist() == [5, 6]

    def test_construction_from_tree_based_tensor(self):
        t = tensap.TuckerLikeTensor.randn([2, 3], [5, 6])
        tbt = t.tree_based_tensor()
        t2 = tensap.TuckerLikeTensor(tbt)
        assert np.allclose(t.numpy(), t2.numpy())
        assert np.all(t.shape == t2.shape)
        assert np.all(t.ranks == t2.ranks)

    def test_repr(self):
        t = self._make_tucker(2)
        r = repr(t)
        assert "TuckerLikeTensor" in r
        assert "order" in r

    # ---- convert_tensors tests ----

    def test_convert_tensors_same_type(self):
        ft = tensap.FullTensor(np.ones((2, 3)))
        x, y = tensap.convert_tensors(ft, ft)
        assert x is ft and y is ft

    def test_convert_tensors_full_diag(self):
        ft = tensap.FullTensor(np.ones((2, 3, 4)))
        dt = tensap.DiagonalTensor(np.ones(4), order=3)
        x, y = tensap.convert_tensors(ft, dt)
        assert isinstance(x, tensap.FullTensor)
        assert isinstance(y, tensap.FullTensor)

    def test_convert_tensors_diag_full(self):
        ft = tensap.FullTensor(np.ones((2, 3, 4)))
        dt = tensap.DiagonalTensor(np.ones(4), order=3)
        x, y = tensap.convert_tensors(dt, ft)
        assert isinstance(x, tensap.FullTensor)
        assert isinstance(y, tensap.FullTensor)

    def test_convert_tensors_tlt_add_mixed_cores(self):
        rng = np.random.RandomState(0)
        core_ft = tensap.FullTensor(rng.rand(3, 3))
        core_dt = tensap.DiagonalTensor(rng.rand(3), order=2)
        space = tensap.TSpaceVectors([rng.rand(5, 1, 3), rng.rand(6, 1, 3)])
        t1 = tensap.TuckerLikeTensor(core_ft, space)
        t2 = tensap.TuckerLikeTensor(core_dt, space)
        result = t1 + t2
        assert isinstance(result, tensap.TuckerLikeTensor)
        assert np.allclose(result.numpy(), t1.numpy() + t2.numpy())

    def test_convert_tensors_tlt_dot_mixed_cores(self):
        rng = np.random.RandomState(1)
        core_ft = tensap.FullTensor(rng.rand(3, 3))
        core_dt = tensap.DiagonalTensor(rng.rand(3), order=2)
        space = tensap.TSpaceVectors([rng.rand(5, 1, 3), rng.rand(6, 1, 3)])
        t1 = tensap.TuckerLikeTensor(core_ft, space)
        t2 = tensap.TuckerLikeTensor(core_dt, space)
        d = t1.dot(t2)
        expected = np.sum(t1.numpy() * t2.numpy())
        assert np.isclose(d, expected)

    def test_convert_tensors_fallback(self):
        with pytest.raises(NotImplementedError):
            tensap.convert_tensors(
                tensap.FullTensor(np.ones(3)),
                "not_a_tensor",
            )
