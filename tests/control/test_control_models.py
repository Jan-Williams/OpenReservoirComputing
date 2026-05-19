import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import orc
import orc.control
import orc.data


@pytest.fixture
def dummy_control_problem_params():
    """Set up dummy data for testing ESN controllers."""
    Nx = 8
    control_dim = 2
    time_steps = 500
    # Create simple sinusoidal system data
    frequencies = jnp.linspace(0.5, 2.0, Nx)
    time = jnp.arange(time_steps).reshape(-1, 1)
    dummy_data = jnp.sin(2 * jnp.pi * frequencies * time / time_steps)
    # Create simple control inputs
    control_data = 0.1 * jnp.sin(
        2
        * jnp.pi
        * jnp.linspace(0, 1, time_steps).reshape(-1, 1)
        * jnp.array([1.0, 1.5])
    )
    U_test = dummy_data[-100:]
    C_test = control_data[-100:]
    U_train = dummy_data[:-100]
    C_train = control_data[:-100]
    return Nx, control_dim, U_train, C_train, U_test, C_test


####################### ESN CONTROLLER DIMENSION TESTS #####################


def test_esn_controller_initialization():
    """Test that ESNController initializes with correct dimensions."""
    data_dim = 3
    control_dim = 2
    res_dim = 100

    controller = orc.control.ESNController(
        data_dim=data_dim, control_dim=control_dim, res_dim=res_dim, seed=0
    )

    assert controller.data_dim == data_dim
    assert controller.control_dim == control_dim
    assert controller.res_dim == res_dim
    assert controller.in_dim == data_dim
    assert controller.out_dim == data_dim
    # Embedding should accept data_dim + control_dim
    assert controller.embedding.in_dim == data_dim + control_dim


def test_esn_controller_force_shapes():
    """Test that force method produces correct output shapes."""
    data_dim = 3
    control_dim = 2
    res_dim = 100
    seq_len = 50

    controller = orc.control.ESNController(
        data_dim=data_dim, control_dim=control_dim, res_dim=res_dim, seed=0
    )

    # Create dummy sequences
    in_seq = jax.random.normal(jax.random.PRNGKey(0), (seq_len, data_dim))
    control_seq = jax.random.normal(jax.random.PRNGKey(1), (seq_len, control_dim))
    res_state = jnp.zeros(res_dim, dtype=jnp.float64)

    # Force the reservoir
    res_seq = controller.force(in_seq, control_seq, res_state)

    # Check output shape
    assert res_seq.shape == (seq_len, res_dim)


def test_esn_controller_apply_control_shapes():
    """Test that apply_control method produces correct output shapes."""
    data_dim = 3
    control_dim = 2
    res_dim = 100
    fcast_len = 25

    controller = orc.control.ESNController(
        data_dim=data_dim, control_dim=control_dim, res_dim=res_dim, seed=0
    )

    # Create dummy control sequence
    control_seq = jax.random.normal(jax.random.PRNGKey(0), (fcast_len, control_dim))
    res_state = jax.random.normal(jax.random.PRNGKey(1), (res_dim,))

    # Apply control
    output_seq = controller.apply_control(control_seq, res_state)

    # Check output shape
    assert output_seq.shape == (fcast_len, data_dim)


####################### ESN CONTROLLER TRAINING TESTS #####################


def test_esn_controller_train_basic(dummy_control_problem_params):
    """Test basic training of ESNController on dummy data."""
    Nx, control_dim, U_train, C_train, U_test, C_test = dummy_control_problem_params

    res_dim = 200

    # Initialize controller
    controller = orc.control.ESNController(
        data_dim=Nx,
        control_dim=control_dim,
        res_dim=res_dim,
        seed=0,
    )

    # Train controller
    controller_trained, R = orc.control.train_ESNController(
        controller,
        train_seq=U_train,
        control_seq=C_train,
        spinup=50,
        beta=1e-6,
    )

    # Check that training completed
    assert controller_trained is not None
    assert R is not None
    assert R.shape[0] == U_train.shape[0]
    assert R.shape[1] == res_dim


def test_esn_controller_control_application(dummy_control_problem_params):
    """Test applying control sequence in closed-loop."""
    Nx, control_dim, U_train, C_train, U_test, C_test = dummy_control_problem_params

    res_dim = 200
    fcast_len = 50

    # Initialize and train controller
    controller = orc.control.ESNController(
        data_dim=Nx,
        control_dim=control_dim,
        res_dim=res_dim,
        seed=0,
    )

    controller_trained, R = orc.control.train_ESNController(
        controller,
        train_seq=U_train,
        control_seq=C_train,
        spinup=50,
        beta=1e-6,
    )

    U_controlled = controller_trained.apply_control(
        control_seq=C_test[:fcast_len], res_state=R[-1]
    )

    # Check output
    assert U_controlled.shape == (fcast_len, Nx)
    assert jnp.all(jnp.isfinite(U_controlled))


####################### LORENZ CONTROL TEST #####################


def test_lorenz_control_basic():
    """
    Test ESNController on Lorenz63 system with control.
    This is a basic sanity check - not a full stabilization test.
    """
    res_dim = 500
    data_dim = 3
    control_dim = 3
    tN = 50
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])
    test_perc = 0.2

    # Get Lorenz data
    U, _ = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    split_idx = int((1 - test_perc) * U.shape[0])
    U_train = U[:split_idx, :]
    U_test = U[split_idx:, :]

    # Create simple control sequence (small perturbations)
    C_train = 0.1 * jax.random.normal(
        jax.random.PRNGKey(42), (U_train.shape[0], control_dim)
    )
    C_test = 0.1 * jax.random.normal(
        jax.random.PRNGKey(43), (U_test.shape[0], control_dim)
    )

    controller = orc.control.ESNController(
        data_dim=data_dim,
        control_dim=control_dim,
        res_dim=res_dim,
        seed=0,
        quadratic=True,
    )

    controller_trained, R = orc.control.train_ESNController(
        controller,
        train_seq=U_train,
        control_seq=C_train,
        spinup=100,
        beta=1e-7,
    )

    # Apply control for a short horizon
    fcast_len = 50
    U_controlled = controller_trained.apply_control(
        control_seq=C_test[:fcast_len], res_state=R[-1]
    )

    assert U_controlled.shape == (fcast_len, data_dim)
    assert jnp.all(jnp.isfinite(U_controlled))
    # Check that the output is in a reasonable range for Lorenz system
    assert jnp.all(jnp.abs(U_controlled) < 100)


def test_esn_controller_reconstruction():
    """
    Test that controller can learn to reconstruct training data.
    """
    res_dim = 300
    data_dim = 3
    control_dim = 3
    tN = 30
    dt = 0.01
    u0 = np.array([0.05, 1, 1.05])

    # Get Lorenz data
    U, _ = orc.data.lorenz63(tN=tN, dt=dt, u0=u0)
    U_train = U[:2000, :]

    # Zero control
    C_train = jnp.zeros((U_train.shape[0], control_dim))

    # Train controller
    controller = orc.control.ESNController(
        data_dim=data_dim,
        control_dim=control_dim,
        res_dim=res_dim,
        seed=42,
    )

    controller_trained, R_ctrl = orc.control.train_ESNController(
        controller, train_seq=U_train, control_seq=C_train, spinup=200, beta=1e-7
    )

    # Compare training reconstruction
    U_ctrl_recon = jax.vmap(controller_trained.readout.readout)(R_ctrl[200:])

    # Check that controller produces reasonable outputs
    assert jnp.all(jnp.isfinite(U_ctrl_recon))
    reconstruction_error = (
        jnp.linalg.norm(U_ctrl_recon - U_train[200:]) / U_train[200:].shape[0]
    )
    # With zero control, controller should still reconstruct well
    assert reconstruction_error < 1.0  # Loose bound for sanity check


def test_compute_control(dummy_control_problem_params):
    """Test that control can be computed."""
    Nx, control_dim, U_train, C_train, U_test, C_test = dummy_control_problem_params

    res_dim = 200
    fcast_len = 20

    # Initialize and train controller
    controller = orc.control.ESNController(
        data_dim=Nx,
        control_dim=control_dim,
        res_dim=res_dim,
        seed=0,
    )

    controller_trained, R = orc.control.train_ESNController(
        controller,
        train_seq=U_train,
        control_seq=C_train,
        spinup=50,
        beta=1e-6,
    )

    control = controller_trained.compute_control(
        jnp.zeros((fcast_len, control_dim)), R[-1], jnp.zeros((fcast_len, Nx))
    )

    assert control.shape == (fcast_len, control_dim)


####################### UNIFIED train_RCController TESTS #####################


def test_train_rccontroller_esn(dummy_control_problem_params):
    """Test that train_RCController gives same results to train_ESNController."""
    Nx, control_dim, U_train, C_train, U_test, C_test = dummy_control_problem_params

    res_dim = 200

    controller = orc.control.ESNController(
        data_dim=Nx,
        control_dim=control_dim,
        res_dim=res_dim,
        seed=0,
    )

    # Train with both functions
    ctrl_old, R_old = orc.control.train_ESNController(
        controller,
        train_seq=U_train,
        control_seq=C_train,
        spinup=50,
        beta=1e-6,
    )

    ctrl_new, R_new = orc.control.train_RCController(
        controller,
        train_seq=U_train,
        control_seq=C_train,
        spinup=50,
        beta=1e-6,
    )

    # Verify identical results
    assert jnp.allclose(ctrl_old.readout.wout, ctrl_new.readout.wout)
    assert jnp.allclose(R_old, R_new)


def test_train_rccontroller_quadratic(dummy_control_problem_params):
    """Test train_RCController with quadratic readout."""
    Nx, control_dim, U_train, C_train, U_test, C_test = dummy_control_problem_params

    res_dim = 200

    controller = orc.control.ESNController(
        data_dim=Nx,
        control_dim=control_dim,
        res_dim=res_dim,
        seed=0,
        quadratic=True,
    )

    ctrl_trained, R = orc.control.train_RCController(
        controller,
        train_seq=U_train,
        control_seq=C_train,
        spinup=50,
        beta=1e-6,
    )

    assert ctrl_trained is not None
    assert R.shape[0] == U_train.shape[0]
    assert R.shape[1] == res_dim

    # Apply control
    fcast_len = 50
    U_controlled = ctrl_trained.apply_control(
        control_seq=C_test[:fcast_len], res_state=R[-1]
    )
    assert U_controlled.shape == (fcast_len, Nx)
    assert jnp.all(jnp.isfinite(U_controlled))


##################### JAX TRANSFORM & AD TESTS #####################


@pytest.fixture
def esn_controller():
    return orc.control.ESNController(
        data_dim=3, control_dim=2, res_dim=200, seed=0
    )


def test_controller_transform_stability(esn_controller):
    """Verify the controller closed-loop is compatible with vmap and jit."""
    model = esn_controller
    key = jax.random.key(999)
    fcast_len = 10
    batch_size = 3

    control_seq = jax.random.normal(key, shape=(fcast_len, model.control_dim))
    state_shape = model.driver.default_state().shape
    batch_state = jax.random.normal(key, shape=(batch_size, *state_shape))

    def fwd(res_state):
        return model.apply_control(control_seq, res_state)

    vmap_fwd = eqx.filter_vmap(fwd)
    assert jnp.allclose(
        vmap_fwd(batch_state),
        jnp.stack([fwd(s) for s in batch_state]),
    )

    jit_fwd = eqx.filter_jit(fwd)
    assert jnp.all(jnp.isfinite(jit_fwd(batch_state[0])))


def test_controller_differentiability(esn_controller):
    """Verify gradients flow backward through apply_control via Equinox."""
    model = esn_controller
    key = jax.random.key(999)
    fcast_len = 10

    control_seq = jax.random.normal(key, shape=(fcast_len, model.control_dim))
    res_state = jax.random.normal(key, shape=model.driver.default_state().shape)

    @eqx.filter_value_and_grad
    def loss_fn(m, c, s):
        return jnp.sum(m.apply_control(c, s))

    loss, grads = loss_fn(model, control_seq, res_state)
    assert jnp.isfinite(loss)

    def check_finite(g):
        if eqx.is_array(g) and jnp.issubdtype(g.dtype, jnp.inexact):
            assert jnp.all(jnp.isfinite(g))

    jax.tree_util.tree_map(check_finite, grads)
