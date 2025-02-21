

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import build_benchmark_torch
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV


# %% Build samples

def generate_samples(N, X, fun, jac_fun, basis, R=None):
    x_set = X.lhs_random(N)
    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)
    basis_set = basis.eval(x_set)
    jac_basis_set = basis.eval_jacobian(x_set)

    if fun_set.ndim == 1:
        fun_set = fun_set[:,None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:,None,:]

    loss_set = PoincareLossVectorSpace(jac_fun_set, jac_basis_set, basis, R)

    return x_set, fun_set, jac_fun_set, basis_set, jac_basis_set, loss_set


# %% Fit kernel ridge regressor

def fit_krr_regressor(z_set, u_set):

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
basis = _build_ortho_poly_basis(X, p=1, m=max_deg)
K = basis.cardinal()

# %% Sampling

N_train = 300
x_train, u_train, jac_u_train, basis_train, jac_basis_train, loss_train = generate_samples(
    N_train, X, u, jac_u, basis)


N_test = 500
x_test, u_test, jac_u_test, basis_test, jac_basis_test, loss_test = generate_samples(
    N_test, X, u, jac_u, basis)


# %% Define minimization parameters
optimizer_kwargs = {
    'beta_rule': 'PolakRibiere',
    'orth_value': 10,
    'max_iterations': 5, 
    'verbosity':2
    }

pmo_kwargs = {
    'use_precond':True, 
    'precond_kwargs':{}, 
    'optimizer_kwargs':optimizer_kwargs, 
    'ls_kwargs':{}
}

# %% Minimize the Poicare loss greedy surrogate
m_max = 3
G, losses, losses_optimized, surrogates = loss_train.minimize_surrogate_greedy(
    m_max, optimize_poincare=True, tol=1e-7, verbose=2, pmo_kwargs=pmo_kwargs)


# %% Plot for eyeball regression

z_train = basis.eval(x_train) @ G
z_test = basis.eval(x_test) @ G

fig, ax = plt.subplots(1, z_train.shape[1])
ax[0].set_ylabel('u(X)')

for i in range(z_train.shape[1]):
    ax[i].scatter(z_train[:,i], u_train, label='train')
    ax[i].scatter(z_test[:,i], u_test, label='test')
    ax[i].set_xlabel(f'g_{i}(X)')

fig.suptitle(f"Degree {max_deg} poly features on {x_train.shape[0]} train samples", y=0.)
plt.show()


# %% Fit Kernel Ridge regression with sklearn

kr_regressor = fit_krr_regressor(z_train, u_train)
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








# %% greedy surrogate
m = 4
optimizer_kwargs = {'max_iterations': 5, 'verbosity':2,'min_step_size':1e-5}

G, losses, losses_optimized, surrogates = tensap.poincare_surrogate_greedy(
    jac_u_train, jac_basis_train, m=m, R=R, optimize_poincare=True, tol=1e-7,
    use_precond=True, 
    optimizer_kwargs=optimizer_kwargs)


#%% Plot the results

plt.semilogy(losses, label='Poincare loss')
plt.semilogy(losses_optimized, label='Poincare loss optimized')
plt.semilogy(surrogates, label='Surrogates')
plt.xlabel("Greedy iterations")
plt.legend()
plt.grid()
plt.show()


# %% Evaluate the features
G = np.linalg.svd(G, full_matrices=False)[0]
z_train = basis.eval(x_train) @ G
z_test = basis.eval(x_test) @ G

# %% Plot for eyeball regression if one feature

fig, ax = plt.subplots()
ax.scatter(z_train[:,:1], u_train, label='train')
ax.scatter(z_test[:,:1], u_test, label='test')
ax.set_xlabel('g_1(X)')
ax.set_ylabel('u(X)')
ax.legend()
plt.show()

# %% Fit Kernel Ridge regression with sklearn

from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

kr = GridSearchCV(
    KernelRidge(kernel="rbf", gamma=0.1),
    param_grid={
        "alpha": np.logspace(-7, -3, 20),
        "gamma": np.logspace(-5, 1, 20)},
    scoring="neg_mean_squared_error"
)

kr.fit(z_train, u_train)
print(f"Best KRR with params: {kr.best_params_} and R2 score: {kr.best_score_:.3f}")

# %% Evaluate performances

y_train = kr.predict(z_train)
err_train = np.mean((y_train - u_train)**2)

y_test = kr.predict(z_test)
err_test = np.mean((y_test - u_test)**2)

print(f"\nKernel regression based on {m} feature learnt by surrogate")
print("MSE on train set: ", err_train)
print("MSE on test set: ", err_test)

# %% Plot final regression

if z_test.shape[1] == 1:

    fig, ax = plt.subplots(1,2)
    ax[0].scatter(z_train, u_train, label='u(X)')
    ax[0].scatter(z_train, y_train, label='f(g(X))')
    ax[0].set_title(f"Train set mse={err_train:.3e}")
    ax[0].set_xlabel('g(X)')
    ax[0].legend()

    ax[1].scatter(z_test, u_test, label='u(X)')
    ax[1].scatter(z_test, y_test, label='f(g(X))')
    ax[1].set_title(f"Test set mse={err_test:.3e}")
    ax[1].set_xlabel('g(X)')
    ax[1].legend()


    fig.suptitle(f"Degree {max_deg} poly features from surrogate and kernel ridge regression on {x_train.shape[0]} train samples", y=0)


# %% Plot final regression

plt.scatter(y_train, u_train, label='u(X)')
plt.scatter(y_test, u_test, label='f(g(X))')
plt.legend()
plt.show()

# %%
