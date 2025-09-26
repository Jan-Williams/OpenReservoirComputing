# imports
import equinox as eqx
import jax
import jax.numpy as jnp

import orc
from orc.utils.regressions import ridge_regression  



# define Embedding function - two-layer MLP with an ELU activation function: u --ELU(W1u+b1)>> u1 --ELU(W2u1+b2)>> r
class ELUEmbedding(orc.embeddings.EmbedBase):
    """
    orc.embeddings.EmbedBase expects us to specify: 
    input dimension in_dim, 
    reservoir dimension res_dim, 
    embed - define a method that maps from in_dim to res_dim
    """

    W1: jnp.ndarray # weight matrix for first layer of ELU MLP
    W2: jnp.ndarray # weight matrix for second layer of ELU MLP
    b1: jnp.ndarray # bias for first layer of ELU MLP
    b2: jnp.ndarray # bias for second layer of ELU MLP

    def __init__(self, in_dim, res_dim, seed=0):
        super().__init__(in_dim=in_dim, res_dim=res_dim)
        rkey = jax.random.key(seed)
        W1key, W2key, b1key, b2key = jax.random.split(rkey, 4)
        # random initialization of parameters of ELU MLP
        self.W1 = jax.random.normal(W1key, shape=(res_dim // 2, in_dim)) / jnp.sqrt((res_dim // 2) * in_dim)
        self.W2 = jax.random.normal(W2key, shape=(res_dim, res_dim // 2))/ jnp.sqrt(res_dim * (res_dim // 2))
        self.b1 = jax.random.normal(b1key, shape=(res_dim // 2)) / jnp.sqrt(res_dim // 2)
        self.b2 = jax.random.normal(b1key, shape=(res_dim)) / jnp.sqrt(res_dim)

    def embed(self, in_state):
        in_state = self.W1 @ in_state + self.b1
        in_state = jax.nn.elu(in_state)
        in_state = self.W2 @ in_state + self.b2
        in_state = jax.nn.elu(in_state)
        return in_state
    

# define driver function - update equations of a gated recurrent unit (GRU), already implemented in equinox
class GRUDriver(orc.drivers.DriverBase):

    gru: eqx.Module

    def __init__(self, res_dim, seed=0):
        super().__init__(res_dim=res_dim)
        key = jax.random.key(seed)
        self.gru = eqx.nn.GRUCell(res_dim, res_dim, key=key)
    
    def advance(self, res_state, in_state):
        return self.gru(in_state, res_state)
    

# define readout function - W_O r (initialize W_O with zeros)
class Readout(orc.readouts.ReadoutBase):
    res_dim: int
    out_dim: int
    W_O: jnp.ndarray
    
    def __init__(self, out_dim, res_dim):
        super().__init__(out_dim, res_dim)
        self.W_O = jnp.zeros((out_dim, res_dim))
    
    def readout(self, res_state):
        return self.W_O @ res_state
    

# define RC - embedding + driver + readout
class Forecaster(orc.rc.RCForecasterBase):
    driver: orc.drivers.DriverBase
    readout: orc.readouts.ReadoutBase
    embedding: orc.embeddings.EmbedBase


##########################################################################

# CODE:


# integrate Rossler system
tN = 200
dt = 0.01
u0 = jnp.array([-10, 2, 1], dtype=jnp.float64)
U,t = orc.data.rossler(tN=tN, dt=dt, u0=u0)
split_idx = int(U.shape[0] * 0.8)
U_train = U[:split_idx]
U_test = U[split_idx:]
t_test = t[split_idx:]
# orc.utils.visualization.plot_time_series(
#     U,
#     t,
#     state_var_names=["$u_1$", "$u_2$", "$u_3$"],
#     title="Rossler Data",
#     x_label= "$t$",
# )


# create RC
Nr = 500
Nu = 3
driver = GRUDriver(Nr)
embedding = ELUEmbedding(Nu, Nr)
readout = Readout(Nu, Nr)
model = Forecaster(driver, readout, embedding)


# teacher force the reservoir
forced_seq = model.force(U_train[:-1], res_state=jnp.zeros((Nr)))

# shift the indices of the target sequence of training data
target_seq = U_train[1:]

# set transient to discard
spinup = 200

# learn W_O
readout_mat = ridge_regression(forced_seq[spinup:], target_seq[spinup:], beta=1e-7)


#under the hood many ORC objects are instances of equinox.Module, which are immutable. Thus, we need to create a new Forecaster object with readout.W_O set to readout_mat:
# define where in the forecaster model we need to update
def where(model: Forecaster):
    return model.readout.W_O
model = eqx.tree_at(where, model, readout_mat)


# perform forecast
U_pred = model.forecast_from_IC(fcast_len=U_test.shape[0], spinup_data=U_train[-spinup:])

# plot forecast
orc.utils.visualization.plot_time_series(
    [U_test, U_pred],
    (t_test - t_test[0]) * 0.07,
    state_var_names=["$u_1$", "$u_2$", "$u_3$"],
    time_series_labels=["True", "Predicted"],
    line_formats=["-", "r--"],
    x_label= r"$\lambda _1 t$",
)


######################################################################

orc.utils.visualization.plot_in_3D_state_space(
    [U_test, U_pred],
    state_var_names=["$u_1$", "$u_2$", "$u_3$"],
    time_series_labels=["True", "Predicted"],
    line_formats=["-", "r--"],
)