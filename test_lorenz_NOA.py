import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import matplotlib.gridspec as gridspec
import pandas as pd
import seaborn as sns
import scipy.optimize as optimize



# Lorenz System Definition

def get_Lorenz_data(timesteps, dt):
    # # Generate Lorenz data:

    # Lorenz System parameters:
    sigma = 10
    rho = 28
    beta = 8/3

    # Time span for the solution
    t_span = [0, timesteps]
    # t_eval = np.linspace(t_span[0], t_span[1], num_of_dt)  # a sequence of time points for which to solve for
    # dt = t_eval[1] - t_eval[0]

    # solve ODE
    sol = solve_ivp(lorenz_system, t_span, s0, args=(sigma, rho, beta), method='RK45', t_eval=np.arange(0, timesteps, dt),rtol=1e-12)
    X = sol.y

    # # Plot Lorenz data:
    # x, y, z = sol.y
    # ax.scatter(x, y, z, s=0.5)  # s is point size
    # fig = plt.figure(figsize=(10, 6))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.set_title("Lorenz Attractor")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.tight_layout()
    # plt.show()

    return X, dt

def lorenz_system(t, xyz, sigma, rho, beta):
    x, y, z = xyz
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def get_W(n_reservoir, sparsity, rng, spectral_radius):
    '''generate full rank sparse matrix, normalized so the largest eigenvalue is the spectral_radius'''
    # while True:
    W = rng.random((n_reservoir, n_reservoir)) - 0.5    # Generate dense random matrix with values in [-0.5, 0.5]
    mask = rng.random(W.shape) < sparsity    # Apply sparsity mask
    W *= mask
    # Check full rank
    # if np.linalg.matrix_rank(W) == n_reservoir:
    largest_EV_abs = np.max(np.abs(np.linalg.eigvals(W)))   # largest eigenvalue of W
    W = W * spectral_radius/largest_EV_abs                  # normalizing W so the largest eigenvalue is the spectral_radius
    return W
        
def get_W_diagonal(n_reservoir, rng, spectral_radius):
    '''generate full rank diagonal matrix, normalized so the largest eigenvalue is the spectral_radius'''
    diag_values = rng.uniform(-1, 1, size=n_reservoir)
    W_diag = np.diag(diag_values)
    largest_EV_abs = np.max(diag_values)                # largest eigenvalue
    W_diag = W_diag * spectral_radius/largest_EV_abs    # normalizing
    return W_diag

def StepForward(u_k, state_k, time):
    if u_k.ndim == 1:
        u_k = u_k.reshape(feature_num,1) # convert u_k to a column vector
    state_k = alpha* np.tanh(W.dot(state_k) + W_in.dot(u_k) + sigma_b * np.ones([n_reservoir,1])) + (1 - alpha*leak)*state_k        # update reservoir state taking one step forward
    time += 1
    return state_k, time

def ridge_reg_loss_func(Y_gt, Y_pred, forecast_horizon_ind=None):
    forecast_horizon_ind = len(Y_gt) if forecast_horizon_ind == None else forecast_horizon_ind
    Y_gt = Y_gt[:,:forecast_horizon_ind]
    Y_pred = Y_pred[:,:forecast_horizon_ind]
    ridge_reg_loss = np.sqrt(np.mean((Y_pred - Y_gt) ** 2))  
    ridge_reg_loss_time = np.sqrt(np.mean((Y_pred - Y_gt) ** 2, axis=0))  
    return ridge_reg_loss, ridge_reg_loss_time

def smape_loss_func_t(Y_gt, Y_pred, t, i_start, i_end):
    """
    Computes the sMAPE between time indices i_start and i_end.
    Y_gt, Y_pred: shape (dim, time)
    t: 1D array of time values
    """
    dt = t[1] - t[0]  # assuming uniform time grid
    y_p = Y_pred[:, i_start:i_end+1]
    y_t = Y_gt[:, i_start:i_end+1]
    numer = np.abs(y_p - y_t)
    denom = np.abs(y_p) + np.abs(y_t)
    coef = 2 / (t[i_end] - t[i_start])
    smape = coef * np.sum(np.mean(numer / denom, axis=0)) * dt  
    return smape

def forecast_horizon_sMAPE(Y_gt, Y_pred, t, epsilon):
    i_start = 0
    n = Y_gt.shape[1]
    horizon_index = n - 1  # default - entire time series
    smape_loss = 0
    for i_end in range(i_start + 1, n):
        smape = smape_loss_func_t(Y_gt, Y_pred, t, i_start, i_end)
        if smape > epsilon:
            horizon_index = i_end
            smape_loss = smape
            break
    return t[horizon_index], horizon_index, smape_loss


def plot_3D(Y, Y_pred, train_test):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*Y, s=0.5, color='orange', label='Lorenz Train')  # s is point size
    ax.scatter(*Y_pred, s=0.5, color='blue', label='RC Prediction Train')
    ax.set_title(f"Lorenz Attractor prediction - {train_test}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    plt.show()


def plot_xyz(Y, Y_pred, t, train_test):
    fig, ax = plt.subplots(3,1,figsize=(8,6))
    fig.suptitle(f'Forecasting L63, N_res = {n_reservoir} ({train_test})', fontsize=16)

    ax[0].plot(t, Y[0, :], 'k-', linewidth=3, label='Truth')
    ax[0].plot(t, Y_pred[0,:],'r--', linewidth=2, label='Forecast')
    ax[0].set_xlabel('$t$')
    ax[0].set_ylabel('$x$')
    #ax[0].set_ylim([-20,20])

    ax[1].plot(t, Y[1, :], 'k-', linewidth=3)
    ax[1].plot(t, Y_pred[1,:],'r--',linewidth=2)
    ax[1].set_xlabel('$ t$')
    ax[1].set_ylabel('$y$')

    ax[2].plot(t, Y[2, :], 'k-', linewidth=3)
    ax[2].plot(t, Y_pred[2,:],'r--',linewidth=2)

    ax[2].set_xlabel('$t$')
    ax[2].set_ylabel('$z$')
    plt.tight_layout()
    ax[0].legend()
    plt.show()

def plot_EV(W, cont_discrete = ""):
    # Compute eigenvalues
    eigvals_W = np.linalg.eigvals(W)
    spectral_radius_actual = np.max(np.abs(eigvals_W))

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(eigvals_W.real, eigvals_W.imag, s=20, label="Eigenvalues")

    # Add unit circle (if spectral_radius=1) or actual spectral radius
    circle = plt.Circle((0, 0), spectral_radius_actual, color='blue', fill=False, linestyle='--', label=f"Actual spectral Radius = {spectral_radius_actual:.2f}")
    ax.add_artist(circle)

    # Axis formatting
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(f"Eigenvalues of {cont_discrete} dynamical system (Reservoir)")
    ax.grid(True)
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_rc_results(Y, Y_pred, t, train_test, params=None):
    fig = plt.figure(figsize=(12, 8))
    ridge_loss_str = f"{params['ridge_loss']:.2f}" if params else ""
    fig.suptitle(f'Forecasting L63, N_res = {n_reservoir} ({train_test}), ridge_loss = {ridge_loss_str}', fontsize=16)
    params_str = ", ".join(f"{name} = {p:.2f}" for name, p in params.items()) if params else ""
    fig.text(0.5, 0.93, params_str,
             ha='center', fontsize=12, style='italic', color='gray')

    # Create 3 rows x 2 columns grid
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 3])  # right column is wider

    # Time series x (ax0)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t, Y[0, :], 'k-', linewidth=2, label='Truth')
    ax0.plot(t, Y_pred[0, :], 'r--', linewidth=1.5, label='Forecast')
    ax0.set_ylabel('$x$')
    ax0.set_xlabel('$t$')
    ax0.legend()

    # Time series y (ax1)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(t, Y[1, :], 'k-', linewidth=2)
    ax1.plot(t, Y_pred[1, :], 'r--', linewidth=1.5)
    ax1.set_ylabel('$y$')
    ax1.set_xlabel('$t$')

    # Time series z (ax2)
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(t, Y[2, :], 'k-', linewidth=2)
    ax2.plot(t, Y_pred[2, :], 'r--', linewidth=1.5)
    ax2.set_ylabel('$z$')
    ax2.set_xlabel('$t$')

    # 3D plot spans all rows in right column (ax3)
    ax3 = fig.add_subplot(gs[:, 1], projection='3d')
    ax3.scatter(*Y, s=0.5, color='orange', label='Lorenz')
    ax3.scatter(*Y_pred, s=0.5, color='blue', label='RC Prediction')
    ax3.set_title("Lorenz Attractor")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_zlabel("Z")
    ax3.legend()

    if params and "forecast_horizon" in params:
        fh = params["forecast_horizon"]
        for ax in [ax0, ax1, ax2]:
            ax.axvline(x=fh, color='gray', linestyle='--', linewidth=1)
            ax.text(fh, ax.get_ylim()[1]*0.9, f'{fh:.2f}', rotation=90,
                    va='top', ha='right', fontsize=8, color='gray')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()

#######################################

def create_RC(feature_num, n_reservoir, is_W_diagonal):
    # Generate Reservoir Weights
    rng = np.random.default_rng(0)     # random seed
    W_in = (rng.random((n_reservoir, feature_num)) - 0.5) * 2 * input_scaling   # random input weight matrix between [-input_scaling,input_scaling]. (nodes in input with nodes in reservoir)
    if is_W_diagonal:
        W = get_W_diagonal(n_reservoir, rng, spectral_radius)
    else:
        W = get_W(n_reservoir, sparsity, rng, spectral_radius)                               # reservoir ajacency matrix - random recurrent weight matrix between [-0.5,0.5]. keep only a "sparsity" percent number of connections. normalizing W so the largest eigenvalue is the spectral_radius
    return W, W_in


def Train_RC():
    # Training
    sp = int(num_of_dt * 0.008)
    
    #initiate state:
    state_k = np.zeros([n_reservoir,1])
    time = 0

    #propagate states for transient indices:
    for i in range(sp):
        u_k = U_train[:,i]
        state_k, time = StepForward(u_k, state_k, time)

    # Run Reservoir over training data (get state of reservoir at all times t without sp)
    X_train = np.zeros([n_reservoir, len(U_train[0,sp+1:])])
    train_count = 0
    for i in range(sp,U_train.shape[1] - 1):
        u_k = U_train[:,i]
        state_k, time = StepForward(u_k, state_k, time)
        X_train[:, train_count] = state_k[:,0]
        train_count += 1
        if np.linalg.norm(state_k) > 1e10 or np.isnan(np.linalg.norm(state_k)):
            print(f'Reservoir trajectory blew up after {str(i)} steps in training')
            break
    x_mat = X_train
    u_mat = U_train[:, sp+1:]

    # train output weights using regularized least squares
    lhs = (x_mat.dot(x_mat.T) + ridge_lambda * np.eye(n_reservoir))
    rhs = x_mat.dot(u_mat.T)
    W_out = np.linalg.lstsq(lhs, rhs, rcond=None)[0].T

    # For Ploting the Train trajectory:
    y_train_inds = list(range(train_ind_start+sp,train_ind_end-1))
    y_GT = X[:,y_train_inds]
    Y_train_teach_pred = W_out.dot(X_train)
    t = np.linspace(0, (len(y_train_inds))*dt, len(y_train_inds))
    
    return time, state_k, W_out, train_len, train_ind_end, y_GT, Y_train_teach_pred, t


def predict_RC(time, state_k, W_out, train_len, train_ind_end):
    # number of time steps to forecast
    forecast_len = int(train_len * 0.2)

    # Run Reservoir over test data: and predict
    test_count = 0
    X_test = np.zeros([n_reservoir, forecast_len])
    for i in range(forecast_len):
        temp_u = W_out.dot(state_k)
        state_k, time = StepForward(temp_u, state_k, time)
        X_test[:, test_count] = state_k[:,0]
        test_count += 1

        if np.linalg.norm(state_k) > 1e10 or np.isnan(np.linalg.norm(state_k)):
            print(f'Forecasted trajectory blew up after {str(i)} steps')
            break

    # For Ploting the Prediction trajectory:
    y_test_inds = list(range(train_ind_end,train_ind_end+forecast_len))
    y_GT = X[:, y_test_inds]
    y_test_PRED = W_out.dot(X_test)
    t = np.linspace(0, forecast_len*dt, forecast_len) * 0.9 ### characteristic Lyapunov timescale, 0.9 = lambda1 the largest Lyapunov exponent
    
    return y_GT, y_test_PRED, t
    

def get_fixed_point(cont_discrete):
    def f_r(r):
        '''continuous:
                f(r) = (r(t+dt) - r(t)) / dt
           discrete:
                f(r) = r(t+dt)
        '''
        r = r.reshape(-1, 1)  # ensure column vector
        if cont_discrete == "continuous":
            return ((alpha * np.tanh(W @ r + W_in @ (W_out @ r) + sigma_b * np.ones((n_reservoir, 1))) - alpha * leak * r)/dt).flatten()  # return as flat vector
        elif cont_discrete == "discrete":
            return (alpha * np.tanh(W @ r + W_in @ (W_out @ r) + sigma_b * np.ones((n_reservoir, 1))) + (1 - alpha * leak) * r).flatten()  # return as flat vector
    # Initial guess
    x0 = np.zeros(n_reservoir)
    # Find the root
    sol = optimize.root(f_r, x0)
    r_fixed = sol.x
    return r_fixed

def compute_jacobian(r, W, W_in, W_out, alpha, leak, sigma_b, cont_discrete):
    '''continuous:
            A_tilde = [[d(f_i(r))/d(r_j(t))]] = [[d((r_i(t+dt) + r_i(t))/dt)/d(r_j(t))]]
       discrete:
            A_tilde = [[d(f_i(r))/d(r_j(t))]] = [[d(r_i(t+dt))/d(r_j(t))]]
    '''
    r = r.reshape(-1, 1)  # ensure column vector
    u = W @ r + W_in @ (W_out @ r) + sigma_b * np.ones((n_reservoir, 1))
    tanh_u = np.tanh(u)
    sech2_u = 1 - tanh_u**2  # derivative of tanh
    D = np.diagflat(sech2_u)  # diagonal matrix
    A = W + W_in @ W_out
    if cont_discrete == "continuous":
        J = (alpha * A @ D + ( - alpha * leak) * np.eye(n_reservoir)) / dt
    elif cont_discrete == "discrete":
        J = alpha * D @ A + (1 - alpha * leak) * np.eye(n_reservoir)
    return J






###########################################################################################
###########################################################################################

# INPUT:

input_scaling = 0.08      # Scales how strongly the input affects the reservoir

# set initial condition:
s0 = [-10.0, 1.0, 10.0]    # [-0.1, 1, 1.05] ,[1, 1, 1]  # Initial values for x,y,z

# Timesteps
timesteps = 100
num_of_dt = 10000


# Reservoir Parameters
n_reservoir = 400        # Number of neurons in reservoir
sparsity = 0.02           # Fraction of reservoir weights that are non-zero (controls sparsity)
input_scaling = 0.084      # Scales how strongly the input affects the reservoir
spectral_radius = 0.8    # Controls the echo state property (reservoir stability - stable if smaller than 1)
is_W_diagonal = False

leak = 0.6
sigma_b = 1.6             # Input bias

# ridge regression parameter
ridge_lambda = 8e-8     # Tikhonov regularization parameter - small vlaue ensures stability in case X not invertible [det(X)=0 ; (X^T X)^(-1)->inf]


# # state propagation in time parameters
# alpha_range = np.linspace(0, 2, 10)
# sigma_b_range = np.linspace(0.5, 2.5, 10)             # Input bias


##############################################################

#CODE:

# integrate the Lorenz system:
X, dt = get_Lorenz_data(timesteps=100, dt=0.01)

# train split:
feature_num = X.shape[0]
train_len = int(X.shape[1] * 0.8)
train_ind_start = 0
train_ind_end = train_len
train_inds = list(range(train_ind_start,train_ind_end))
U_train = X[:,train_inds]


results = []
results_smape = []
results_horizon = []
results_score_temp = []
max_forecast_horizon = 0
for alpha in [0.67]:   # [0.67]
    print(f"alpha: {alpha}")
    for sigma_b in [0.94]:   # [0.94]
        print(f"sigma_b: {sigma_b}")
        
        # Initialize and train the ESN:
        W, W_in = create_RC(feature_num, n_reservoir, is_W_diagonal)
        time, state_k, W_out, train_len, train_ind_end, y_train_GT, Y_train_teach_pred, t_train = Train_RC()

        # Forecast:
        y_test_GT, y_test_PRED, t_test = predict_RC(time, state_k, W_out, train_len, train_ind_end)

# Plot
plot_rc_results(y_test_GT, y_test_PRED, t_test, "Test", None)


######################################################

#         # forecast horizon with sMAPE:
#         forecast_horizon, forecast_horizon_ind, smape_loss = forecast_horizon_sMAPE(y_test_GT, y_test_PRED, t_test, epsilon)
#         # ridge regression:
#         test_loss, test_loss_time = ridge_reg_loss_func(y_test_GT, y_test_PRED, forecast_horizon_ind)
#         print(f"test loss: {test_loss}")

#         #for param tuning
#         results.append({"alpha": alpha, "sigma_b": sigma_b, "ridge_loss": test_loss})
#         results_smape.append({"alpha": alpha, "sigma_b": sigma_b, "ridge_loss": test_loss, "smape_loss": smape_loss})
#         results_horizon.append({"alpha": alpha, "sigma_b": sigma_b, "ridge_loss": test_loss, "forecast_horizon": forecast_horizon})
#         #combined smape and horizon
#         results_score_temp.append((alpha, sigma_b, test_loss, smape_loss, forecast_horizon))  #loss function is last param for sorting
#         max_forecast_horizon = forecast_horizon if forecast_horizon > max_forecast_horizon else max_forecast_horizon

# #get results_score:
# loss_scores = [float("{:.2f}".format(lambda_score * smape_loss - (1 - lambda_score) * forecast_horizon / max_forecast_horizon)) for _, _, _, smape_loss, forecast_horizon in results_score_temp]
# print(f"loss scores: {loss_scores}")
# results_score = []
# for i in range(len(loss_scores)):
#     alpha, sigma_b, test_loss, smape_loss, forecast_horizon = results_score_temp[i]
#     results_score.append({"alpha": alpha, "sigma_b": sigma_b, "ridge_loss": test_loss, "smape_loss": smape_loss, "forecast_horizon": forecast_horizon, "loss_score": loss_scores[i]})


# #Plot eigenvalues of W (reservoir matrix):
# # plot_EV(W)

# #Plot eigenvalues of linear approx of states equation (reservoir matrix):
# cont_discrete = "continuous"        # "continuous", "discrete"

# dt = t_test[1] - t_test[0]
# r_fixed = get_fixed_point(cont_discrete)
# print(f"r_fixed: {r_fixed}")
# J_at_fixed = compute_jacobian(r_fixed, W, W_in, W_out, alpha, leak, sigma_b, cont_discrete)
# print(f"Jacobian at fixed point: {J_at_fixed}")
# plot_EV(J_at_fixed, cont_discrete)




# ########################################################
# # FIND OPTIMAL PARAMS:

# params_best = min(results_score, key=lambda x: x[test_loss_func])  # sort by the loss function








