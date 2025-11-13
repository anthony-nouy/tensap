

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from tensap.poincare_learning.benchmarks.poincare_benchmarks import build_benchmark
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace, \
    PoincareLossVectorSpaceTruncated
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging
logging.basicConfig(level=logging.INFO)

# %% Function to generate samples


def generate_samples(N, ind1, X, fun, jac_fun, basis, R=None):
    x_set = X.lhs_random(N)
    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)[:, ind1]
    basis_set = basis.eval(x_set[:, ind1])
    jac_basis_set = basis.eval_jacobian(x_set[:, ind1])

    if fun_set.ndim == 1:
        fun_set = fun_set[:, None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:, None, :]

    loss_set = PoincareLossVectorSpace(jac_fun_set, jac_basis_set, basis, R)

    return x_set, fun_set, jac_fun_set, basis_set, jac_basis_set, loss_set


def generate_samples_tensorized(N1, N2, ind1, X, fun, jac_fun, basis, R=None):

    ind2 = np.delete(np.arange(X.ndim()), ind1)

    X1 = X.marginal(ind1)
    X2 = X.marginal(ind2)

    x1_set = X1.lhs_random(N1)
    x2_set = X2.lhs_random(N2)

    x_set = np.zeros((N1 * N2, X.ndim()))
    jac_basis_set = basis.eval_jacobian(x1_set)
    jac_fun_set_tensorized = np.zeros((N1, N2, X.ndim() - 1))

    for i in range(N1):
        # x2_set = X2.lhs_random(N2)
        x_set_i = np.zeros((N2, X.ndim()))
        x_set_i[:, ind1] = x1_set[i]
        x_set_i[:, ind2] = x2_set
        x_set[i * N2:(i + 1) * N2] = x_set_i
        jac_fun_set_tensorized[i, :, :] = jac_fun(x_set_i)[:, ind1]

    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)
    basis_set = basis.eval(x1_set)
    jac_basis_set = basis.eval_jacobian(x1_set)

    if fun_set.ndim == 1:
        fun_set = fun_set[:, None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:, None, :]

    loss_set_tensorized = PoincareLossVectorSpaceTruncated(
        jac_fun_set_tensorized, jac_basis_set, basis, R)

    return x_set, fun_set, jac_fun_set_tensorized, basis_set, jac_basis_set, loss_set_tensorized


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
    print(f"Best KRR with params: {kr.best_params_} and MSE score: {-kr.best_score_:.3e}")

    return kr


def fit_poly_regressor(z_set, u_set):
    model = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression(fit_intercept=False))
    ])

    param_grid = {
        'poly__degree': np.arange(10)
    }

    cv = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=10
    )

    cv.fit(z_set, u_set)
    print(f"Best Poly with params: {cv.best_params_} and  score: {-cv.best_score_:.3e}")

    return cv


# %% functions to build matrice list

def build_mat_lst_1(dim, n_mat):
    mat_lst = [np.eye(dim)]
    for i in range(1, n_mat):
        mat = np.eye(dim)
        for j in range(1, i + 1):
            mat += np.diag(np.ones(dim - j), k=j)
            mat += np.diag(np.ones(dim - j), k=-j)
        mat = mat / i
        mat_lst.append(mat)
    return mat_lst


def build_mat_lst_2(dim, n_mat):
    mat_lst = [np.eye(dim)]
    for i in range(1, n_mat):
        mat = np.diag(np.ones(dim - i), k=i)
        mat = mat + mat.T
        mat = mat / 2
        mat_lst.append(mat)
    return mat_lst


def build_mat_lst_3(dim, n_mat):
    mat_lst = []
    for i in range(n_mat):
        mat = np.random.RandomState(i).normal(size=(dim, dim))
        mat = mat.T @ mat
        mat = mat / np.linalg.norm(mat, ord=2)
        mat_lst.append(mat)
    return mat_lst


def build_mat_lst(dim, n_mat, which=1):
    if which == 1:
        out = build_mat_lst_1(dim, n_mat)
    elif which == 2:
        out = build_mat_lst_2(dim, n_mat)
    elif which == 3:
        out = build_mat_lst_3(dim, n_mat)
    else:
        raise NotImplementedError
    return out

# %% Definition of the benchmark


d = 8 + 1
n_mat = 3
ind1 = np.arange(d - 1)
ind2 = np.delete(np.arange(d), ind1)
mat_lst = build_mat_lst(d - 1, n_mat, which=2)

# if pytorch is installed
try:
    from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import build_benchmark_torch
    u, jac_u, X = build_benchmark_torch("quartic_sin_collective", d=d, mat_lst=mat_lst)


except ImportError:
    u, jac_u, X = build_benchmark("quartic_sin_collective", d=d, mat_lst=mat_lst)

# %% build a polynomial basis

p_norm = 1  # p-norm of the multi-indices
max_deg = 2  # bound on the p_norm
basis = _build_ortho_poly_basis(X.marginal(ind1), p=p_norm, m=max_deg)
K = basis.cardinal()
R = basis.gram_matrix_h1_0()

#####################################
# %% Tensorized sample with surrogate
#####################################

# %% Sampling

N1_train = 25
N2_train = 4
x_train, u_train, _, _, _, loss_train_tensorized = generate_samples_tensorized(
    N1_train, N2_train, ind1, X, u, jac_u, basis, R)

N_test = 1000
x_test, u_test, _, _, _, loss_test = generate_samples(
    N_test, ind1, X, u, jac_u, basis, R)

# %% Minimize the surrogate

m = len(mat_lst) - 0
loss_train_tensorized.truncate(m)
G_surr, _, _ = loss_train_tensorized.minimize_surrogate(m=m)
G_surr = np.linalg.svd(G_surr, full_matrices=False)[0]

# %% Evaluate performances

print(f"\nPoincare loss and Surrogate on {G_surr.shape[1]} features")
print(f"Surrogate on tensorized train set:      {loss_train_tensorized.eval_surrogate(G_surr):.3e}")
print(f"Poincare loss on tensorized train set : {loss_train_tensorized.eval(G_surr):.3e}")
print(f"Poincare loss on test set:              {loss_test.eval(G_surr):.3e}")

# %% Plot for eyeball regression

z_surr_train = basis.eval(x_train[:, ind1]) @ G_surr
z_surr_test = basis.eval(x_test[:, ind1]) @ G_surr

fig, ax = plt.subplots(1, z_surr_train.shape[1])
if z_surr_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel('u(X)')

for i in range(z_surr_train.shape[1]):
    for j in range(N1_train):
        ax[i].scatter(z_surr_train[j::N2_train, i], u_train[j::N2_train], label='train', s=5.)
    ax[i].set_xlabel(f'g_{i}(X)')

fig.suptitle(f"""
    Surrogate only tensorized sample
    Poly features m={z_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples
    """,
    y=0.)
plt.show()


# %% Fit regressor with sklearn

# add the last parameter as a feature
zy_surr_train = np.hstack([z_surr_train, x_train[:, ind2]])
zy_surr_test = np.hstack([z_surr_test, x_test[:, ind2]])

# shuffle = np.random.permutation(len(zy_surr_train))
shuffle = np.arange(len(zy_surr_train))

# regressor_surr = fit_krr_regressor(zy_surr_train[shuffle], u_train[shuffle])
regressor_surr = fit_poly_regressor(zy_surr_train[shuffle], u_train[shuffle])


def f_surr(z):
    return regressor_surr.predict(z)


# %% Evaluate performances

y_train = f_surr(zy_surr_train).reshape(u_train.shape)
err_train = np.sqrt(np.mean((y_train - u_train)**2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_surr(zy_surr_test).reshape(u_test.shape)
err_test = np.sqrt(np.mean((y_test - u_test)**2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(f"\nSurrogate only tensorized sample | Regression based on {G_surr.shape[1]} features")
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %% Plot final regression

plt.scatter(y_test, u_test, label='test', s=5.)
plt.scatter(y_train, u_train, label='train', s=5.)
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate only tensorized sample
    Poly features m={zy_surr_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples""")
plt.show()

##############################################################
# %% (Optional) Run Minimization of the Poincare loss on the grassmann manifold
##############################################################

# CG on grassmann manifold using pymanopt, when installed
try:
    optimizer_kwargs = {  # params for pymanopt CG
        'beta_rule': 'PolakRibiere',
        'max_iterations': 50,
        'verbosity': 2,
    }
    G_opt, _, _ = loss_train_tensorized.minimize_pymanopt(
        G_surr, use_precond=True, optimizer_kwargs=optimizer_kwargs
    )

# quasi newton
except ImportError:
    G_opt, _ = loss_train_tensorized.minimize_qn(G0=G_surr, m=m, maxiter=50, tol=1e-10)


# %% Evaluate performances

print(f"\nPoincare loss and Surrogate as init on {G_opt.shape[1]} features")
print(f"Surrogate on tensorized train set:      {loss_train_tensorized.eval_surrogate(G_opt):.3e}")
print(f"Poincare loss on tensorized train set : {loss_train_tensorized.eval(G_opt):.3e}")
print(f"Poincare loss on test set:              {loss_test.eval(G_opt):.3e}")

# %% Plot for eyeball regression

z_opt_train = basis.eval(x_train[:, ind1]) @ G_opt
z_opt_test = basis.eval(x_test[:, ind1]) @ G_opt

fig, ax = plt.subplots(1, z_opt_train.shape[1])
if z_opt_train.shape[1] == 1:
    ax = [ax]
ax[0].set_ylabel('u(X)')

for i in range(z_opt_train.shape[1]):
    for j in range(N1_train):
        ax[i].scatter(z_opt_train[j::N2_train, i], u_train[j::N2_train], label='train', s=5.)
    ax[i].set_xlabel(f'g_{i}(X)')

fig.suptitle(f"""
    Surrogate as init tensorized sample
    Poly features m={z_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples
    """,
    y=0.)
plt.show()


# %% Fit regressor with sklearn

# add the last parameter as a feature
zy_opt_train = np.hstack([z_opt_train, x_train[:, ind2]])
zy_opt_test = np.hstack([z_opt_test, x_test[:, ind2]])

# shuffle = np.random.permutation(len(zy_opt_train))
shuffle = np.arange(len(zy_opt_train))

# regressor_opt = fit_krr_regressor(zy_opt_train[shuffle], u_train[shuffle])
regressor_opt = fit_poly_regressor(zy_opt_train[shuffle], u_train[shuffle])


def f_opt(z):
    return regressor_opt.predict(z)


# %% Evaluate performances

y_train = f_opt(zy_opt_train).reshape(u_train.shape)
err_train = np.sqrt(np.mean((y_train - u_train)**2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_opt(zy_opt_test).reshape(u_test.shape)
err_test = np.sqrt(np.mean((y_test - u_test)**2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(f"\nSurrogate as init tensorized sample | Regression based on {G_opt.shape[1]} features")
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %% Plot final regression

plt.scatter(y_test, u_test, label='test', s=5.)
plt.scatter(y_train, u_train, label='train', s=5.)
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    Surrogate as init tensorized sample
    Poly features m={zy_opt_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples""")
plt.show()

##############################################################
# %% Compare with non tensorized sample and classical approach
##############################################################

# %% resample
x_train, u_train, _, _, _, loss_train = generate_samples(
    N1_train * N2_train, ind1, X, u, jac_u, basis, R)

# %% minimize the poincare based loss function

# CG on grassmann manifold using pymanopt, when installed
try:
    optimizer_kwargs = {  # params for pymanopt CG
        'beta_rule': 'PolakRibiere',
        'max_iterations': 100,
        'verbosity': 2,
    }
    G_pmo, _, _ = loss_train.minimize_pymanopt(
        m=m, init_method='active_subspace', optimizer_kwargs={'max_iterations': 100})

# quasi newton
except ImportError:
    G_pmo, _ = loss_train.minimize_qn(
        G0=None, m=m, init_method='active_subspace', maxiter=50, tol=1e-10)

# %% Evaluate performances

print(f"\nPoincare loss on {G_pmo.shape[1]} features")
print(f"Poincare loss on train set:  {loss_train.eval(G_pmo):.3e}")
print(f"Poincare loss on test set:   {loss_test.eval(G_pmo):.3e}")


# %% Fit regressor with sklearn

# add the last parameter as a feature
zy_pmo_train = np.hstack([basis.eval(x_train[:, ind1]) @ G_pmo, x_train[:, ind2]])
zy_pmo_test = np.hstack([basis.eval(x_test[:, ind1]) @ G_pmo, x_test[:, ind2]])

regressor_pmo = fit_poly_regressor(zy_pmo_train, u_train)


def f_pmo(z):
    return regressor_pmo.predict(z)

# %% Evaluate performances


y_train = f_pmo(zy_pmo_train).reshape(u_train.shape)
err_train = np.sqrt(np.mean((y_train - u_train)**2))
rel_err_train = err_train / np.sqrt((u_train**2).mean())

y_test = f_pmo(zy_pmo_test).reshape(u_test.shape)
err_test = np.sqrt(np.mean((y_test - u_test)**2))
rel_err_test = err_test / np.sqrt((u_test**2).mean())

print(f"\npmoogate only | Regression based on {G_pmo.shape[1]} features")
print(f"L2 on train set    : {err_train:.3e}")
print(f"L2 on test set     : {err_test:.3e}")
print(f"RL2 on train set   : {rel_err_train:.3e}")
print(f"RL2 on test set    : {rel_err_test:.3e}")


# %% Plot final regression

plt.scatter(y_test, u_test, label='test', s=5.)
plt.scatter(y_train, u_train, label='train', s=5.)
plt.ylabel("u(X)")
plt.xlabel("f(g(X))")
plt.legend()
plt.title(f"""
    pymanopt optim
    Poly features m={zy_pmo_train.shape[1]}
    Multi-indices with {p_norm}-norm bounded by {max_deg}
    {N1_train}x{N2_train}={N1_train * N2_train} train samples""")
plt.show()
# %%
