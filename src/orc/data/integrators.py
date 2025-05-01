"""Integrators for solving ODEs and PDEs."""
import functools

import diffrax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

#TODO: typing
######################## Basic Chaotic ODEs ########################
@jax.jit
def _lorenz63_f(t,u,args):
    """Define Lorenz 63 ODE."""
    rho, sigma, beta = args
    u1, u2, u3 = u
    du1dt = sigma*(u2 - u1)
    du2dt = u1*(rho - u3) - u2
    du3dt = u1*u2 - beta*u3
    dudt = du1dt, du2dt, du3dt
    return jnp.array(dudt)

def lorenz63(tN,
             dt,
             u0 = (-10.0,1.0,10.0),
             rho = 28.0,
             sigma = 10.0,
             beta = 8.0/3.0,
             **diffeqsolve_kwargs):
    """Solve the Lorenz 63 system of ODEs.

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is (-10, 1, 10).
    rho : float, optional
        The rho parameter for the Lorenz system. Default is 28.0.
    sigma : float, optional
        The sigma parameter for the Lorenz system. Default is 10.0.
    beta : float, optional
        The beta parameter for the Lorenz system. Default is 8.0/3.0.
    diffeqsolve_kwargs : dict, optional
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, 3), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                  diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_lorenz63_f)
    args = (rho, sigma, beta)
    sol = diffrax.diffeqsolve(term,
                              t0=0,
                              t1=tN,
                              y0=u0,
                              args=args,
                              **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts


@jax.jit
def _rossler_f(t,u,args):
    """Define Rössler ODE."""
    a,b,c = args
    u1, u2, u3 = u
    du1dt = -u2 - u3
    du2dt = u1 + a*u2
    du3dt = b+u3*(u1-c)
    dudt = du1dt, du2dt, du3dt
    return jnp.array(dudt)

def rossler(tN,
            dt,
            u0 = (1.0,1.0,1.0),
            a = 0.1,
            b = 0.1,
            c = 14.0,
            **diffeqsolve_kwargs):
    """Solve the Rossler system of ODEs.

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is (1.0, 1.0, 1.0).
    a : float, optional
        The a parameter for the Rössler system. Default is 0.1.
    b : float, optional
        The b parameter for the Rössler system. Default is 0.1.
    c : float, optional
        The c parameter for the Rössler system. Default is 14.0.
    diffeqsolve_kwargs : dict, optional
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, 3), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                  diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_rossler_f)
    args = (a, b, c)
    sol = diffrax.diffeqsolve(term,
                              t0=0,
                              t1=tN,
                              y0=u0,
                              args=args,
                              **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts


@jax.jit
def _sakaraya_f(t,u,args):
    """Define Sakaraya ODE."""
    a, b, m = args
    u1, u2, u3 = u
    du1dt = a * u1 + u2 + u2*u3
    du2dt = -u1*u3 + u2*u3
    du3dt = -u3 - m*u1*u2 + b

    return jnp.array([du1dt, du2dt, du3dt])

def sakaraya(tN,
             dt,
             u0 = (-2.8976045, 3.8877978, 3.07465),
             a = 1.0,
             b = 1.0,
             m = 1.0,
             **diffeqsolve_kwargs):
    """Solve the Sakaraya system of ODEs.

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is
        (-2.8976045, 3.8877978, 3.07465).
    a : float, optional
        The a parameter for the Sakaraya system. Default is 1.0.
    b : float, optional
        The b parameter for the Sakaraya system. Default is 1.0.
    m : float, optional
        The m parameter for the Sakaraya system. Default is 1.0.
    diffeqsolve_kwargs : dict, optional
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, 3), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_sakaraya_f)
    args = (a, b, m)
    sol = diffrax.diffeqsolve(term,
                                t0=0,
                                t1=tN,
                                y0=u0,
                                args=args,
                                **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts


@jax.jit
def _colpitts_f(t,u,args):
    """Define Colpitts oscillator ODE."""
    u1, u2, u3 = u
    alpha, gamma, q, eta = args
    du1dt = alpha * u2
    du2dt = -gamma*(u1 + u3) - q*u2
    du3dt = eta*(u2 + 1 - jnp.exp(-u1))
    return jnp.array([du1dt, du2dt, du3dt])

def colpitts(tN,
             dt,
             u0 = (1.0, -1.0, 1.0),
             alpha = 5.0,
             gamma = 0.0797,
             q = 0.6898,
             eta = 6.2723,
             **diffeqsolve_kwargs):
    """Solve the Colpitts oscillator system of ODEs.

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is (1.0, -1.0, 1.0).
    alpha : float, optional
        The alpha parameter for the Colpitts oscillator system. Default is
        5.0 (Platt, 2020).
    gamma : float, optional
        The gamma parameter for the Colpitts oscillator system. Default is
        0.0797 (Platt, 2020).
    q : float, optional
        The q parameter for the Colpitts oscillator system. Default is
        0.6898 (Platt, 2020).
    eta : float, optional
        The eta parameter for the Colpitts oscillator system. Default is
        6.2723 (Platt, 2020).
    diffeqsolve_kwargs : dict, optional
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, 3), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                  diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_colpitts_f)
    args = (alpha, gamma, q, eta)
    sol = diffrax.diffeqsolve(term,
                              t0=0,
                              t1=tN,
                              y0=u0,
                              args=args,
                              **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts



######################## Hyper-chaotic ODEs ########################
@jax.jit
def _hyper_lorenz63_f(t,u,args):
    """Define Hyper-Lorenz 63 ODE."""
    a, b, c, d = args
    u1, u2, u3, u4 = u
    du1dt = a*(u2 - u1) + u4
    du2dt = u1*(b - u3) - u2
    du3dt = u1*u2 - c*u3
    du4dt = d*u4 - u2*u3
    return jnp.array([du1dt, du2dt, du3dt, du4dt])

def hyper_lorenz63(tN,
                   dt,
                   u0 = (-10.0, 6.0, 0.0, 10.0),
                   a = 10.0,
                   b = 28.0,
                   c = 8.0/3.0,
                   d = -1.0,
                   **diffeqsolve_kwargs):
    """Solve the Hyper-Lorenz 63 system of ODEs.

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is (-10.0, 6.0, 0.0, 10.0).
    a : float, optional
        The a parameter for the Hyper-Lorenz system. Default is 10.0.
    b : float, optional
        The b parameter for the Hyper-Lorenz system. Default is 28.0.
    c : float, optional
        The c parameter for the Hyper-Lorenz system. Default is 8.0/3.0.
    d : float, optional
        The d parameter for the Hyper-Lorenz system. Default is -1.0.
    diffeqsolve_kwargs : dict, optional
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, 4), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                  diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_hyper_lorenz63_f)
    args = (a, b, c, d)
    sol = diffrax.diffeqsolve(term,
                              t0=0,
                              t1=tN,
                              y0=u0,
                              args=args,
                              **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts


@jax.jit
def _hyper_xu_f(t, u, args):
    """Define Hyper-Xu ODE."""
    a, b, c, d, e = args
    u1, u2, u3, u4 = u
    du1dt = a * (u2 - u1) + u4
    du2dt = b * u1 + e * u1*u3
    du3dt = -c*u3 - u1*u2
    du4dt = u1*u3 - d*u2
    return jnp.array([du1dt, du2dt, du3dt, du4dt])

def hyper_xu(tN,
             dt,
             u0 = (-2.0, -1.0, -2.0, -10.0),
             a = 10.0,
             b = 40.0,
             c = 2.5,
             d = 2.0,
             e = 16.0,
             **diffeqsolve_kwargs):
    """Solve the Hyper-Xu system of ODEs.

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is (-2.0, -1.0, -2.0, -10.0).
    a : float, optional
        The a parameter for the Hyper-Xu system. Default is 10.0.
    b : float, optional
        The b parameter for the Hyper-Xu system. Default is 40.0.
    c : float, optional
        The c parameter for the Hyper-Xu system. Default is 2.5.
    d : float, optional
        The d parameter for the Hyper-Xu system. Default is 2.0.
    e : float, optional
        The e parameter for the Hyper-Xu system. Default is 16.0.
    diffeqsolve_kwargs : dict, optional
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, 4), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                  diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_hyper_xu_f)
    args = (a, b, c, d, e)
    sol = diffrax.diffeqsolve(term,
                              t0=0,
                              t1=tN,
                              y0=u0,
                              args=args,
                              **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts



####################### Hamiltonian Systems #######################
@jax.jit
def _double_pendulum_f(t,u,args):
    """Define the equations of motion for double pendulum."""
    m1, m2, L1, L2, g, damping = args
    theta1, omega1, theta2, omega2 = u

    #define some vars to shorten the expressions
    delta_theta = theta1 - theta2
    M_tot = m1 + m2
    alpha = m1 + m2*jnp.sin(delta_theta)**2

    # compute derivs
    dtheta1_dt = omega1
    dtheta2_dt = omega2
    domega1_dt_num =  (-jnp.sin(delta_theta) *
                       (m2*L1*omega1**2*jnp.cos(delta_theta) + m2*L2*omega2**2) -
                       g*(M_tot * jnp.sin(theta1) -
                       m2*jnp.sin(theta2)*jnp.cos(delta_theta)))
    domega1_dt_denom = (L1 * alpha)
    domega1_dt_damp = damping * (omega1 - omega2) + damping * omega1
    domega1_dt = domega1_dt_num / domega1_dt_denom - domega1_dt_damp
    domega2_dt_num = (jnp.sin(delta_theta) *
                      (M_tot*L1*omega1**2 + m2*L2*omega2**2*jnp.cos(delta_theta)) +
                      g*(M_tot*jnp.sin(theta1)*jnp.cos(delta_theta) -
                      M_tot*jnp.sin(theta2)))
    domega2_dt_denom = (L2 * alpha)
    domega2_dt_damp = damping*(omega2-omega1)
    domega2_dt = domega2_dt_num / domega2_dt_denom - domega2_dt_damp

    return jnp.array([dtheta1_dt, domega1_dt, dtheta2_dt, domega2_dt])

def double_pendulum(tN,
                    dt,
                    u0 = (jnp.pi/4, -1.0, jnp.pi/2, 1.0),
                    m1 = 1.0,
                    m2 = 1.0,
                    L1 = 1.0,
                    L2 = 1.0,
                    g = 9.81,
                    damping = 0.0,
                    **diffeqsolve_kwargs):
    """Solve the equations of motion for a damped double pendulum.

    The state u is represented as a 4-tuple (theta1, omega1, theta2, omega2) where:
        - theta1 is the angle of the first pendulum from vertical (in radians).
        - omega1 is the angular velocity of the first pendulum (in radians/s).
        - theta2 is the angle of the second pendulum from vertical (in radians).
        - omega2 is the angular velocity of the second pendulum (in radians/s).

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is (jnp.pi/4, -1.0, jnp.pi/2, 1.0).
    m1 : float, optional
        The mass of the first pendulum bob. Default is 1.0.
    m2 : float, optional
        The mass of the second pendulum bob. Default is 1.0.
    L1 : float, optional
        The length of the first pendulum rod. Default is 1.0.
    L2 : float, optional
        The length of the second pendulum rod. Default is 1.0.
    g : float, optional
        The acceleration due to gravity. Default is 9.81.
    damping : float, optional
        The damping coefficient for the pendulum system. Default is 0.0 (no damping).
    diffeqsolve_kwargs : dict, optional #TODO test which sovler best conserves energy
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, 4), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                  diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_double_pendulum_f)
    args = (m1, m2, L1, L2, g, damping)
    sol = diffrax.diffeqsolve(term,
                              t0=0,
                              t1=tN,
                              y0=u0,
                              args=args,
                              **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts



###################### High Dimensional ODEs ######################
@jax.jit
def _lorenz96_interior(i,u,F):
    """Define the interior points of the Lorenz 96 ODE."""
    return (u[i+1] - u[i-2])*u[i-1]-u[i] + F

@functools.partial(jax.jit, static_argnames=["args"])
def _lorenz96_f(t,u,args):
    """Define the Lorenz 96 ODE."""
    N, F = args

    # boundary at N
    dudt_N = (u[0] - u[N-3])*u[N-2]-u[N-1] + F

    # calculate all other points (interior plus boundary at 0)
    dudt_func = jax.vmap(_lorenz96_interior, in_axes=(0,None,None))
    interior_idxs = jnp.arange(N-1)
    dudt_interior = dudt_func(interior_idxs, u, F)

    return jnp.append(dudt_interior, dudt_N)

def lorenz96(tN,
             dt,
             u0 = None,
             N = 10,
             F = 8.0,
             **diffeqsolve_kwargs):
    """Solve the Lorenz 96 system of ODEs.

    Parameters
    ----------
    tN : float
        The final time to solve the ODEs to.
    dt : float
        The time step size for the interpolated solution. Will be overridden if
        `diffeqsolve_kwargs` contains a saveat argument with a different time grid.
    u0 : jnp.ndarray, optional
        The initial conditions for the ODEs. Default is None, which initializes y0
        to `jnp.sin(jnp.arange(N))`.
    N : int, optional
        The number of variables in the Lorenz 96 system. Default is 10.
    F : float, optional
        The forcing parameter for the Lorenz 96 system. Default is 8.0.
    diffeqsolve_kwargs : dict, optional
        Additional keyword arguments to pass to the `diffrax.diffeqsolve` function.
        Default solver is `diffrax.Tsit5()`, saveat is set to a grid of times from 0 to
        tN with step size dt, and stepsize_controller is set to
        `diffrax.PIDController(rtol=1e-3, atol=1e-6)`. dt0 is set to dt. max_steps is
        set to None.

    Returns
    -------
    us : jnp.ndarray
        The solution array with shape (Nt, N), where Nt is the number of time steps.
    ts : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # set kwarg defaults
    diffeqsolve_kwargs.setdefault('solver', diffrax.Tsit5())
    diffeqsolve_kwargs.setdefault('saveat', diffrax.SaveAt(ts=jnp.arange(0,tN,dt)))
    diffeqsolve_kwargs.setdefault('stepsize_controller',
                                  diffrax.PIDController(rtol=1e-3, atol=1e-6))
    diffeqsolve_kwargs.setdefault('dt0', dt)
    diffeqsolve_kwargs.setdefault('max_steps', None)

    # solve
    if u0 is None:
        u0 = jnp.sin(jnp.arange(N))
    u0 = jnp.array(u0)
    term = diffrax.ODETerm(_lorenz96_f)
    args = (N, F)
    sol = diffrax.diffeqsolve(term,
                              t0=0,
                              t1=tN,
                              y0=u0,
                              args=args,
                              **diffeqsolve_kwargs)
    us = sol.ys
    return us, sol.ts



########################### Chaotic PDEs ###########################
@functools.partial(jax.jit, static_argnames=["tN", "dt", "Nx"])
def KS_1D(tN, u0=None, dt=0.25, domain=(0, 22), Nx=64):
    """Solve the Kuramoto-Sivashinsky equation in 1D with periodic boundary conditions.

    The KS PDE solved is:
        u_t + u*u_x + u_xx + u_xxxx = 0

    The solver uses a fixed time-step ETDRK4 (Kassam & Trefethen 2005) method
    for handling the stiffness of the PDE. Dealiasing (2/3 rule) is applied.

    Parameters
    ----------
    tN : float
        The final time to solve the PDE to.
    u0 : jnp.ndarray, optional
        The initial condition for the PDE (shape (Nx,)). Default is None, which
        initializes u0 to `sin((32/domain[1])*pi*x)`.
    dt : float, optional
        The time step size for the solution. Default is 0.25.
    domain : tuple[float, float], optional
        The spatial domain (x_min, x_max). Default is (0, 22).
    Nx : int, optional
        The number of spatial grid points. Default is 64.

    Returns
    -------
    U : jnp.ndarray
        The solution array with shape (Nt, Nx+1), where Nt is the number of time steps.
        Includes the periodic boundary point.
    t : jnp.ndarray
        The time vector corresponding to the solution steps.
    """
    # Setup spatial grid
    if u0 is None:
        x0 = jnp.linspace(domain[0], domain[1], Nx, endpoint=True)
        u0 = jnp.sin((32/domain[1])*jnp.pi * x0) #TODO find a better default
    u0 = u0[:-1]
    Nx = Nx - 1  # remove duplicate periodic point
    x = jnp.linspace(domain[0], domain[1], Nx, endpoint=False)
    dx = x[1] - x[0]

    Nt = int(tN / dt)
    U = jnp.zeros((Nx, Nt))
    U = U.at[:, 0].set(u0)

    # Wavenumbers
    k = jnp.fft.fftfreq(Nx, d=dx) * 2 * jnp.pi
    k2 = k**2
    k4 = k**4
    L_op = k2 - k4

    # Dealiasing (2/3 rule)
    def dealias(f_hat):
        cutoff = Nx // 3
        f_hat = f_hat.at[cutoff:-cutoff].set(0)
        return f_hat

    # nonlinear operators on u and u_hat
    def N_op_u(u):
        return dealias(1j * k * jnp.fft.fft(-0.5 * u ** 2))
    def N_op_uhat(u_hat):
        return dealias(1j * k * jnp.fft.fft(-0.5 * jnp.real(jnp.fft.ifft(u_hat)) ** 2))

    # ETDRK4 coefficients (Kassam & Trefethen 2005)
    E1 = jnp.exp(L_op * dt)
    E2 = jnp.exp(L_op * dt / 2)
    M = 16
    r = jnp.exp(1j * jnp.pi * (jnp.arange(1, M + 1) - 0.5) / M)
    LR = dt * jnp.column_stack([L_op]*M) + jnp.vstack([r]*Nx)
    Q = dt * jnp.mean((jnp.exp(LR/2) - 1) / LR, axis=1)
    f1 = dt * jnp.mean((-4 - LR + jnp.exp(LR)*(4 - 3*LR + LR**2)) / LR**3, axis=1)
    f2 = dt * jnp.mean((2 + LR + jnp.exp(LR)*(-2 + LR)) / LR**3, axis=1)
    f3 = dt * jnp.mean((-4 - 3*LR - LR**2 + jnp.exp(LR)*(4 - LR)) / LR**3, axis=1)

    def _KS_ETDRK4_step(carry, _):
        u, E1, E2, Q, f1, f2, f3 = carry

        u_hat = jnp.fft.fft(u)

        a = E2 * u_hat + Q * N_op_u(u)
        b = E2 * u_hat + Q * N_op_uhat(a)
        c = E2 * a + Q * (2*N_op_uhat(b) - N_op_u(u))

        u_hat = E1 * u_hat + f1 * N_op_u(u) \
            + f2 * (N_op_uhat(a) + N_op_uhat(b)) + f3 * N_op_uhat(c)

        # Enforce conservation by zeroing the mean mode
        u_hat = u_hat.at[0].set(0.0)

        u_next = jnp.real(jnp.fft.ifft(u_hat, n=Nx))
        carry_next = (u_next, E1, E2, Q, f1, f2, f3)
        return carry_next, u_next

    _, u_vals = jax.lax.scan(_KS_ETDRK4_step, (u0, E1, E2, Q, f1, f2, f3), length=Nt-1)

    # add back in the initial point and boundary points
    U = jnp.concatenate([u0[None,:], u_vals], axis=0)
    U = jnp.concatenate((U,U[:,0:1]), axis=1)

    # create time vector for output
    t = jnp.arange(0, tN, dt)

    return U,t

