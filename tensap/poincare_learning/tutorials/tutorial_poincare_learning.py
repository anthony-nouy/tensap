

# %% Imports
import numpy as np
import tensap
import matplotlib.pyplot as plt
import scipy
from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import build_benchmark_torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


# %% Build orthonormal polynomial basis of the feature space

def build_basis(X, max_deg=2, p_norm=1):
    bases = tensap.FunctionalBases([
        tensap.PolynomialFunctionalBasis(poly, range(max_deg + 1))
        for poly in X.orthonormal_polynomials()
    ])
    I0 = tensap.MultiIndices.with_bounded_norm(X.ndim(), p_norm, max_deg)
    I0 = I0.remove_indices(0)
    basis = tensap.SparseTensorProductFunctionalBasis(bases, I0)
    return basis


# %% Build samples

def build_samples(N, X, fun, jac_fun, basis, R=None):
    x_set = X.lhs_random(N)
    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)
    basis_set = basis.eval(x_set)
    jac_basis_set = basis.eval_jacobian(x_set)

    if fun_set.ndim == 1:
        fun_set = fun_set[:,None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:,None,:]

    loss_set = tensap.PoincareLossVectorSpace(jac_fun_set, jac_basis_set, basis, R)

    return x_set, fun_set, jac_fun_set, basis_set, jac_basis_set, loss_set


# %% Fit kernel ridge regressor

def build_krr_rergessor(z_set, u_set):

    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        param_grid={
            "alpha": np.logspace(-7, -3, 10),
            "gamma": np.logspace(-5, 1, 10)},
        scoring="neg_mean_squared_error"
    )

    kr.fit(z_set, u_set)
    print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")

    return kr


# %% Definition of the benchmark

u, jac_u, X = build_benchmark_torch("borehole")
#u, jac_u, X = build_benchmark_torch("sin_of_squared_norm", d=8)
#u, jac_u, X = build_benchmark_torch("exp_mean_sin_exp_cos", d=8)


# %% build a polynomial basis

max_deg = 2
basis = build_basis(X, max_deg=max_deg, p_norm=0.8)
K = basis.cardinal()

# %% Sampling

N_train = 300
x_train, u_train, jac_u_train, basis_train, jac_basis_train, loss_train = build_samples(
    N_train, X, u, jac_u, basis)


N_test = 500
x_test, u_test, jac_u_test, basis_test, jac_basis_test, loss_test = build_samples(
    N_test, X, u, jac_u, basis)


# %% Initialize minimization algorithm
m = 1
G0 = np.random.normal(size=(K, m))
G0 = np.linalg.svd(G0, full_matrices=False)[0]

print(f"Random {m} features")
print("Poincare loss on train set: ", loss_train.eval(G0))
print("Poincare loss on test set: ", loss_test.eval(G0))

# %% Minimize the Poicare loss using CG on Grassmann Manifold
optimizer_kwargs = {'max_iterations': 50, 'verbosity':2}
G = loss_train.minimize_pymanopt(
    G0, use_precond=True, optimizer_kwargs=optimizer_kwargs)


# %% Plot for eyeball regression wrt first feature

z_train = basis.eval(x_train) @ G
z_test = basis.eval(x_test) @ G

plt.scatter(z_train[:,:1], u_train, label='train')
plt.scatter(z_test[:,:1], u_test, label='test')
plt.xlabel('g_1(X)')
plt.ylabel('u(X)')
plt.legend()
plt.title(f"Degree {max_deg} poly features on {x_train.shape[0]} train samples")
plt.show()


# %% Fit Kernel Ridge regression with sklearn

kr_regressor = build_krr_rergessor(z_train, u_train)
def f(z) : return kr_regressor.predict(z)


# %% Evaluate performances

y_train = f(z_train)
err_train = np.mean((y_train - u_train)**2)
rel_err_train = err_train / (u_train**2).mean()

y_test = f(z_test)
err_test = np.mean((y_test - u_test)**2)
rel_err_test = err_test / (u_test**2).mean()

print(f"\nKernel regression based on {G.shape[1]} features")
print(f"MSE on train set    : {err_train:.3e}")
print(f"MSE on test set     : {err_test:.3e}")
print(f"RMSE on train set   : {rel_err_train:.3e}")
print(f"RMSE on test set    : {rel_err_test:.3e}")


# %% Plot final regression

plt.scatter(y_train, u_train, label='train')
plt.scatter(y_test, u_test, label='test')
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"Degree {max_deg} poly features and kernel ridge regression on {x_train.shape[0]} train samples")
plt.show()

