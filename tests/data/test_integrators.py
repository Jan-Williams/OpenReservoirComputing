import diffrax
import jax
import jax.numpy as jnp

import orc
import orc.data


def test_lorenz63_output_shapes():
    """Test the output shapes of the lorenz63 integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)

    us, ts = orc.data.lorenz63(tN, dt)

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_lorenz63_custom_args():
    """Test lorenz63 with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    u0 = jnp.array([1.0, 2.0, 3.0])
    rho = 20.0
    sigma = 5.0
    beta = 2.0

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.lorenz63(
        tN,
        dt,
        u0=u0,
        rho=rho,
        sigma=sigma,
        beta=beta,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_lorenz63_symmetry():
    """Test the symmetry of the lorenz63 integrator. If (u1,u2,u3) is a solution,
    then (-u1,-u2,u3) should also be a solution."""
    tN = 10.0
    dt = 0.01

    u0_1 = jnp.array([1.0, 2.0, 3.0])
    u0_2 = jnp.array([-1.0, -2.0, 3.0])

    us_1, _ = orc.data.lorenz63(tN, dt, u0=u0_1)
    us_2, _ = orc.data.lorenz63(tN, dt, u0=u0_2)

    assert jnp.allclose(us_1[1:, -1], us_2[1:, -1]), "Symmetry test failed"


def test_rossler_output_shapes():
    """Test the output shapes of the rossler integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)

    us, ts = orc.data.rossler(tN, dt)

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_rossler_custom_args():
    """Test rossler with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    u0 = jnp.array([0.5, 0.5, 0.5])
    a = 0.2
    b = 0.2
    c = 5.7

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.rossler(
        tN,
        dt,
        u0=u0,
        a=a,
        b=b,
        c=c,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_sakaraya_output_shapes():
    """Test the output shapes of the sakaraya integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)

    us, ts = orc.data.sakaraya(tN, dt)

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_sakaraya_custom_args():
    """Test sakaraya with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    u0 = jnp.array([1.0, 1.0, 1.0])
    a = 0.5
    b = 0.5
    m = 0.5

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.sakaraya(
        tN,
        dt,
        u0=u0,
        a=a,
        b=b,
        m=m,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_colpitts_output_shapes():
    """Test the output shapes of the colpitts integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)

    us, ts = orc.data.colpitts(tN, dt)

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_colpitts_custom_args():
    """Test colpitts with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    u0 = jnp.array([0.5, -0.5, 0.5])
    alpha = 6.0
    gamma = 0.1
    q = 0.7
    eta = 7.0

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.colpitts(
        tN,
        dt,
        u0=u0,
        alpha=alpha,
        gamma=gamma,
        q=q,
        eta=eta,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, 3), f"Expected us shape ({Nt}, 3), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_hyper_lorenz63_output_shapes():
    """Test the output shapes of the hyper_lorenz63 integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)

    us, ts = orc.data.hyper_lorenz63(tN, dt)

    assert us.shape == (Nt, 4), f"Expected us shape ({Nt}, 4), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_hyper_lorenz63_custom_args():
    """Test hyper_lorenz63 with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    u0 = jnp.array([-5.0, 3.0, 1.0, 5.0])
    a = 12.0
    b = 30.0
    c = 2.0
    d = -0.5

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.hyper_lorenz63(
        tN,
        dt,
        u0=u0,
        a=a,
        b=b,
        c=c,
        d=d,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, 4), f"Expected us shape ({Nt}, 4), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_hyper_xu_output_shapes():
    """Test the output shapes of the hyper_xu integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)

    us, ts = orc.data.hyper_xu(tN, dt)

    assert us.shape == (Nt, 4), f"Expected us shape ({Nt}, 4), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_hyper_xu_custom_args():
    """Test hyper_xu with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    u0 = jnp.array([-1.0, -0.5, -1.0, -5.0])
    a = 12.0
    b = 45.0
    c = 3.0
    d = 2.5
    e = 18.0

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.hyper_xu(
        tN,
        dt,
        u0=u0,
        a=a,
        b=b,
        c=c,
        d=d,
        e=e,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, 4), f"Expected us shape ({Nt}, 4), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_double_pendulum_output_shapes():
    """Test the output shapes of the double_pendulum integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)

    us, ts = orc.data.double_pendulum(tN, dt)

    assert us.shape == (Nt, 4), f"Expected us shape ({Nt}, 4), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_double_pendulum_custom_args():
    """Test double_pendulum with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    u0 = jnp.array([jnp.pi / 2, 0.0, jnp.pi, 0.0])
    m1 = 2.0
    m2 = 0.5
    L1 = 1.5
    L2 = 0.8
    g = 10.0
    damping = 0.1

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.double_pendulum(
        tN,
        dt,
        u0=u0,
        m1=m1,
        m2=m2,
        L1=L1,
        L2=L2,
        g=g,
        damping=damping,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, 4), f"Expected us shape ({Nt}, 4), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_lorenz96_output_shapes():
    """Test the output shapes of the lorenz96 integrator."""
    tN = 10.0
    dt = 0.01
    Nt = int(tN / dt)
    N = 5  # Use a smaller N for testing

    us, ts = orc.data.lorenz96(tN, dt, N=N)

    assert us.shape == (Nt, N), f"Expected us shape ({Nt}, {N}), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"


def test_lorenz96_custom_args():
    """Test lorenz96 with custom initial conditions and parameters."""
    tN = 5.0
    dt = 0.02
    Nt = int(tN / dt)
    N = 8
    u0 = jnp.cos(jnp.arange(N))  # Different initial condition
    F = 10.0

    # Use a different solver and stepsize controller
    solver = diffrax.Euler()
    stepsize_controller = diffrax.ConstantStepSize()

    # different saveat to ensure we get the correct number of points
    saveat = diffrax.SaveAt(ts=jnp.linspace(0, tN, Nt))

    us, ts = orc.data.lorenz96(
        tN,
        dt,
        u0=u0,
        N=N,
        F=F,
        solver=solver,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
    )

    assert us.shape == (Nt, N), f"Expected us shape ({Nt}, {N}), but got {us.shape}"
    assert ts.shape == (Nt,), f"Expected ts shape ({Nt},), but got {ts.shape}"
    assert jnp.allclose(us[0], u0), "Initial condition not set correctly"


def test_KS_1D_output_shapes():
    """Test the output shapes of the KS_1D integrator with defaults."""
    tN = 5.0  # Shorter time for faster testing
    Nx = 64
    dt = 0.25
    Nt = int(tN / dt)

    U, t = orc.data.KS_1D(tN=tN, Nx=Nx)

    assert U.shape == (Nt, Nx), f"Expected U shape ({Nt}, {Nx}), but got {U.shape}"
    assert t.shape == (Nt,), f"Expected t shape ({Nt},), but got {t.shape}"


def test_KS_1D_custom_args():
    """Test KS_1D with custom arguments."""
    tN = 2.5
    dt = 0.1
    Nx = 32
    Nt = int(tN / dt)
    domain = (0, 10)
    x = jnp.linspace(domain[0], domain[1], Nx + 1, endpoint=True)
    u0 = jnp.cos(2 * jnp.pi * x / domain[1])

    U, t = orc.data.KS_1D(tN=tN, dt=dt, Nx=Nx + 1, domain=domain, u0=u0)

    assert U.shape == (Nt, Nx + 1), (
        f"Expected U shape ({Nt}, {Nx + 1}), \
        but got {U.shape}"
    )
    assert t.shape == (Nt,), f"Expected t shape ({Nt},), but got {t.shape}"
    assert jnp.allclose(U[0], u0), "Initial condition not set correctly"


def test_KS_1D_conservation():
    """Test KS conservation property. Integral over x of u should be 0."""
    tN = 1000
    domain = (0, 22)
    Nx = 64
    x0 = jnp.linspace(domain[0], domain[1], Nx, endpoint=True)
    u0 = jnp.sin((32 / domain[1]) * jnp.pi * x0)
    u, _ = orc.data.KS_1D(tN=tN, u0=u0, domain=domain, Nx=Nx)
    conserved_quantity = jnp.sum(jax.scipy.integrate.trapezoid(u, axis=1))
    assert jnp.isclose(conserved_quantity, 0.0, atol=1e-10), (
        f"Conserved quantity is not close to zero: {conserved_quantity}"
    )
