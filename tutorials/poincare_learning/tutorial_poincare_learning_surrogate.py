

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from tensap.poincare_learning.benchmarks.poincare_benchmarks import build_benchmark
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging
logging.basicConfig(level=logging.INFO)

# %% Function to generate samples


def generate_samples(N, X, fun, jac_fun, basis, R=None):
    x_set = X.lhs_random(N)
    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)
    basis_set = basis.eval(x_set)
    jac_basis_set = basis.eval_jacobian(x_set)

    if fun_set.ndim == 1:
        fun_set = fun_set[:, None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:, None, :]

    loss_set = PoincareLossVectorSpace(jac_fun_set, jac_basis_set, basis, R)

    return x_set, fun_set, jac_fun_set, basis_set, jac_basis_set, loss_set


# %% Functions to fit regressors

def fit_krr_regressor(z_set, u_set):

    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        param_grid={
            "alpha": np.logspace(-11, -5, 10),
            "gamma": np.logspace(-6, 2, 10)},
        scoring="neg_mean_squared_error"
    )

    kr.fit(z_set, u_set)
    print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")

    return kr


def fit_poly_regressor(z_set, u_set):
    model = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression(fit_intercept=False))
    ])

    param_grid = {
        'poly__degree': [1, 2, 3, 4, 5, 6]
    }

    cv = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=10
    )

    cv.fit(z_set, u_set)
    print(f"Best Poly with params: {cv.best_params_} and MSE score: {-cv.best_score_:.3f}")

    return cv


# %% Definition of the benchmark

# if pytorch is installed
try:
    from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import build_benchmark_torch
    u, jac_u, X = build_benchmark_torch("borehole")

except ImportError:
    u, jac_u, X = build_benchmark("exp_mean_sin_exp_cos", d=8)
    #  u, jac_u, X = build_benchmark("sin_squared_norm", d=8)


# %% build a polynomial basis

p_norm = 1  # p-norm of the multi-indices
max_deg = 3  # bound on the p_norm
basis = _build_ortho_poly_basis(X, p=p_norm, m=max_deg)
K = basis.cardinal()
R = basis.gram_matrix_h1_0()

# %% Sampling

N_train = 300
x_train, u_train, jac_u_train, basis_train, jac_basis_train, loss_train = generate_samples(
    N_train, X, u, jac_u, basis, R)


N_test = 500
x_test, u_test, jac_u_test, basis_test, jac_basis_test, loss_test = generate_samples(
    N_test, X, u, jac_u, basis, R)


# %% Minimize the surrogate
m = 1
G_surr, _, _ = loss_train.minimize_surrogate(m=m)
G_surr = np.linalg.svd(G_surr, full_matrices=False)[0]

# %% Evaluate performances

print(f"\nPoincare loss and Surrogate on {G_surr.shape[1]} features")
print(f"Surrogate on train set:      {loss_train.eval_surrogate(G_surr):.3e}")
print(f"Poincare loss on train set:  {loss_train.eval(G_surr):.3e}")
print(f"Surrogate on test set:       {loss_test.eval_surrogate(G_surr):.3e}")
print(f"Poincare loss on test set:   {loss_test.eval(G_surr):.3e}")

# %% Plot for eyeball regression

z_surr_train = basis.eval(x_train) @ G_surr
z_surr_test = basis.eval(x_test) @ G_surr

fig, ax = plt.subplots(1, z_surr_train.shape[1])
if z_surr_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel('u(X)')

for i in range(z_surr_train.shape[1]):
    ax[i].scatter(z_surr_train[:, i], u_train, label='train')
    ax[i].scatter(z_surr_test[:, i], u_test, label='test')
    ax[i].set_xlabel(f'g_{i}(X)')

fig.suptitle(f"""
    Surrogate only
    Poly features m={z_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples
    """,
    y=0.)
plt.show()


# %% Fit regressor with sklearn

# regressor_surr = fit_krr_regressor(z_surr_train, u_train)
regressor_surr = fit_poly_regressor(z_surr_train, u_train)


def f_surr(z):
    return regressor_surr.predict(z)


# %% Evaluate performances

y_train = f_surr(z_surr_train)
err_train = np.sqrt(np.mean((y_train - u_train)**2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_surr(z_surr_test)
err_test = np.sqrt(np.mean((y_test - u_test)**2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(f"\nSurrogate only | Regression based on {G_surr.shape[1]} features")
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %% Plot final regression

plt.scatter(y_train, u_train, label='train')
plt.scatter(y_test, u_test, label='test')
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate only
    Poly features m={z_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples""")
plt.show()


# %% (Optional) Run Minimization of the Poincare loss on the grassmann manifold

# CG on grassmann manifold using pymanopt, when installed
try:
    optimizer_kwargs = {  # params for pymanopt CG
        'beta_rule': 'PolakRibiere',
        'max_iterations': 50,
        'verbosity': 2,
    }
    G_opt, _, _ = loss_train.minimize_pymanopt(
        G_surr, use_precond=True, optimizer_kwargs=optimizer_kwargs
    )

# quasi newton
except ImportError:
    G_opt, _ = loss_train.minimize_qn(G0=G_surr, m=m, maxiter=50, tol=1e-10)


# %% Plot for eyeball regression

z_opt_train = basis.eval(x_train) @ G_opt
z_opt_test = basis.eval(x_test) @ G_opt

fig, ax = plt.subplots(1, z_opt_train.shape[1])
if z_surr_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel('u(X)')

for i in range(z_opt_train.shape[1]):
    ax[i].scatter(z_opt_train[:, i], u_train, label='train')
    ax[i].scatter(z_opt_test[:, i], u_test, label='test')
    ax[i].set_xlabel(f'g_{i}(X)')

fig.suptitle(f"""
    Surrogate as init
    Poly features m={z_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples
    """,
    y=0.)
plt.show()


# %% Fit regression with sklearn

# regressor_opt = fit_krr_regressor(z_opt_train, u_train)
regressor_opt = fit_poly_regressor(z_opt_train, u_train)


def f_opt(z):
    return regressor_opt.predict(z)


# %% Evaluate performances

y_train = f_opt(z_opt_train)
err_train = np.sqrt(np.mean((y_train - u_train)**2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_opt(z_opt_test)
err_test = np.sqrt(np.mean((y_test - u_test)**2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(f"\nSurrogate as init | Regression based on {G_opt.shape[1]} features")
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %% Plot final regression

plt.scatter(y_train, u_train, label='train')
plt.scatter(y_test, u_test, label='test')
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate as init
    Poly features m={z_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {x_train.shape[0]} train samples""")
plt.show()
