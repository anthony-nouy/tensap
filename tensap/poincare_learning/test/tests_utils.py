

import numpy as np
import scipy
import tensap


def _build_test_case():

    # Build a multivariate polynomial basis with total degree at most 2
    d = 5  # input dimension
    n = 3  # output dimension
    N = 50  # number of samples
    m = 2  # number of active features to learn
    x = np.random.RandomState(0).uniform(-1, 1, size=(N, d))
    h = tensap.PolynomialFunctionalBasis(tensap.CanonicalPolynomials(), range(5))
    H = tensap.FunctionalBases.duplicate(h, d)
    I0 = tensap.MultiIndices.with_bounded_norm(d, 1, 2).remove_indices(0)
    basis = tensap.SparseTensorProductFunctionalBasis(H, I0)
    K = basis.eval(x).shape[1]

    G0 = np.array(np.random.RandomState(0).normal(scale=1 / np.sqrt(K), size=(K, m)))
    G1 = np.array(np.random.RandomState(1).normal(scale=1 / np.sqrt(K), size=(K, m)))
    G2 = np.array(np.random.RandomState(2).normal(scale=1 / np.sqrt(K), size=(K, m)))

    def g(x):
        return basis.eval(x) @ G0

    def jac_g(x):
        jac_basis = basis.eval_jacobian(x)
        out = np.array([G0.T @ jb for jb in jac_basis])
        return out

    def f(z):
        out = np.block([[np.sin(z.prod(axis=1))], [np.cos(z).prod(axis=1)]]).T
        return out

    def jac_f(z):
        out = np.zeros((z.shape[0], n, z.shape[1]))
        out[:, 0, :] = np.array([
            np.cos(zi.prod()) * zi.prod() * np.ones(zi.shape[0]) / zi
            for zi in z])
        out[:, 1, :] = np.array([
            - np.cos(zi).prod() * np.ones(zi.shape[0]) * np.sin(zi) / np.cos(zi)
            for zi in z])
        return out

    def u(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return f(g(x))

    def jac_u(x):
        if x.ndim == 1:
            x = x.reshape(-1)
        jac_fz = jac_f(g(x))
        jac_gx = jac_g(x)
        out = np.array([jf @ jg for jf, jg in zip(jac_fz, jac_gx)])
        return out

    R = np.random.RandomState(0).normal(size=(G0.shape[0], G0.shape[0]))
    R = R @ R.T

    loss = tensap.PoincareLossVectorSpace(jac_u(x), basis.eval_jacobian(x), basis, R)
    loss1 = tensap.PoincareLossVectorSpace(jac_u(x)[:, [0], :], basis.eval_jacobian(x), basis, R)
    loss2 = tensap.PoincareLossVectorSpace(jac_u(x)[:, [1], :], basis.eval_jacobian(x), basis, R)
    resolution = np.sqrt(np.finfo(G0.dtype).resolution)

    return G0, G1, G2, loss, loss1, loss2, resolution


def test_eval():
    G0, G1, _, loss, loss1, loss2, resolution = _build_test_case()

    l0 = loss.eval(G0)
    l01 = loss1.eval(G0)
    l02 = loss2.eval(G0)
    l1 = loss.eval(G1)
    l11 = loss1.eval(G1)
    l12 = loss2.eval(G1)

    np.testing.assert_allclose([l0, l01, l02], 0, atol=resolution)
    np.testing.assert_allclose(l01 + l02 - l0, 0, atol=resolution)
    np.testing.assert_allclose(l11 + l12 - l1, 0, atol=resolution)
    np.testing.assert_array_less([-l1, -l11, -l12], -resolution)


def test_eval_gradient():
    G0, G1, _, loss, loss1, loss2, resolution = _build_test_case()

    l0 = loss.eval_gradient(G0)
    l01 = loss1.eval_gradient(G0)
    l02 = loss2.eval_gradient(G0)
    l1 = loss.eval_gradient(G1)
    l1_bis = 2 * (loss.eval_SG_X(G1, G1) - loss.eval_HG_X(G1, G1))
    l11 = loss1.eval_gradient(G1)
    l12 = loss2.eval_gradient(G1)

    np.testing.assert_allclose(
        [l0, l01, l02], 0, atol=resolution, err_msg="Gradient should be zero at min")
    np.testing.assert_allclose(l11 + l12, l1)
    np.testing.assert_allclose(l1, l1_bis)
    np.testing.assert_array_less(
        -np.linalg.norm(l1), -resolution, err_msg="Gradient should be non zero for G1")


def test_eval_SGinv():
    _, G1, G2, loss, _, _, resolution = _build_test_case()
    SG1_G2 = loss.eval_SG_X(G1, G2)
    err = np.linalg.norm(G2 - loss.eval_SGinv_X(G1, SG1_G2, None, {'rtol': 1e-10}))
    np.testing.assert_allclose(err, 0, atol=resolution)


def test_hessian():
    G0, G1, G2, loss, loss1, loss2, resolution = _build_test_case()

    hl1_2 = loss.eval_HessG_X(G1, G2)  # Hess(G1) @ G2
    hl11_2 = loss1.eval_HessG_X(G1, G2)
    hl12_2 = loss2.eval_HessG_X(G1, G2)

    H1 = loss.eval_HessG_full(G1)
    H0 = loss.eval_HessG_full(G0)
    w0 = np.linalg.eigvals(H0)

    err_sym = np.linalg.norm(H0 - H0.T) / np.linalg.norm(H0)
    err_sym += np.linalg.norm(H1 - H1.T) / np.linalg.norm(H1)
    err_spd = w0 / np.linalg.norm(H0)

    np.testing.assert_allclose(hl11_2 + hl12_2, hl1_2)
    np.testing.assert_allclose(err_sym, 0, atol=resolution, err_msg="Hessian should be symmetric")
    np.testing.assert_array_less(-err_spd, resolution, err_msg="Hessian should be spd at minimum")


def test_eval_surrogate_matrices():
    G0, _, _, loss, loss1, _, _ = _build_test_case()
    R = loss.R
    G0_orth = G0 @ np.linalg.inv(np.linalg.cholesky(G0.T @ R @ G0).T)

    A, B, C = loss.eval_surrogate_matrices(None)
    w, v = scipy.linalg.eigh(B - A, R)
    G_fit = v[:, :G0.shape[1]]
    sigma = np.linalg.svd(G_fit.T @ R @ G0_orth)[1].min()

    A, B, C = loss1.eval_surrogate_matrices(G0[:, [0]])
    w, v = scipy.linalg.eigh(B - A + C, R)
    G_fit = v[:, :G0.shape[1]]
    sigma1 = np.linalg.svd(G_fit.T @ R @ G0_orth[:, [1]])[1].min()

    np.testing.assert_allclose(sigma, 1, err_msg="Expected exact recovery of all features")
    np.testing.assert_allclose(sigma1, 1, err_msg="Expected exact recovery of second feature")


if __name__ == "__main__":
    test_eval()
    test_eval_gradient()
    test_eval_SGinv()
    test_hessian()
    test_eval_surrogate_matrices()
