import jax.numpy as jnp
import matplotlib.pyplot as plt
 

#  TODO: Write the ODE function Question 1
def system_ode(y, b, m, l, g):
    """
    Define the ODE function F(y) for the pendulum system.

    Parameters:
        y (jnp.array): State vector [theta, omega].
        b, m, l, g (float): ODE parameters (damping, mass, length, gravity).

    Returns:
        dy_dt (jnp.array): Time derivative of the state vector.
    """
    
    assert m > 0, "Mass (m) must be positive"
    assert l > 0, "Length (l) must be positive"
    assert g > 0, "Gravity (g) must be positive"
    
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(b / (m * l)) * omega - (g / l) * jnp.sin(theta)
    
    return jnp.array([dtheta_dt, domega_dt])


#  TODO: Write the function for the euler method Question 2.
def solve_pendulum_euler(pendulum_ode, y0, t_span, dt, b, m, l, g):
    """
    Solve the ODE using the Euler method.

    Parameters:
        pendulum_ode (function): ODE function for the pendulum system.
        y0 (jnp.array): Initial state [theta, omega].
        t_span (jnp.array): Time vector.
        dt (float): Time step.
        b, m, l, g (float): ODE parameters (damping, mass, length, gravity).

    Returns:
        y (jnp.array): Solution array for [theta, omega] over time.
    """
    t0, t_n = t_span
    t = jnp.arange(t0, t_n, dt)
    
    y = jnp.zeros((len(t), len(y0)))
    y = y.at[0].set(y0)
    
    # dt = t[1] - t[0]
    for i in range(1, len(t)):
        y = y.at[i].set(y[i - 1] + dt * pendulum_ode(y[i - 1], b, m, l, g))
    
    return t, y


#  TODO: Write the function for the Runge–Kutta methods. method Question 2.
def runge_kutta_4(pendulum_ode, y0, t_span, dt, b, m, l, g):
    """
    Solve the ODE using the 4th order Runge-Kutta method.

    Parameters:
        pendulum_ode (function): ODE function for the pendulum system.
        y0 (jnp.array): Initial state [theta, omega].
        t_span (jnp.array): Time vector.
        dt (float): Time step.
        b, m, l, g (float): ODE parameters (damping, mass, length, gravity).

    Returns:
        t (jnp.array): Time vector.
        y (jnp.array): Solution array for [theta, omega] over time.
        
    """
    
    t0, t_end = t_span
    t = jnp.arange(t0, t_end, dt)
    y = jnp.zeros((len(t), len(y0)))
    y = y.at[0].set(y0)

    for i in range(1, len(t)):
        k1 = pendulum_ode(y[i - 1], b, m, l, g)
        k2 = pendulum_ode(y[i - 1] + 0.5 * dt * k1, b, m, l, g)
        k3 = pendulum_ode(y[i - 1] + 0.5 * dt * k2, b, m, l, g)
        k4 = pendulum_ode(y[i - 1] + dt * k3, b, m, l, g)
        y = y.at[i].set(y[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        
    return t, y


#  TODO: Write the function for plotting the solution  Question 3.
def plot_solution(t, y_euler, y_runge_kutta, labels):
    """
    Plot the solution obtained by Euler and Runge-Kutta methods.

    Parameters:
        t (jnp.array): Time vector.
        y_euler, y_runge_kutta (jnp.array): Solution from the solvers.
        labels (list): Labels for the plot (theta, omega).
    """
    
    fig, axs = plt.subplots(2, figsize=(10, 6))
    
    axs[0].plot(t, y_euler[:, 0 ], label="Euler Method")
    axs[0].plot(t, y_runge_kutta[:, 0], label="Runge-Kutta Method", linestyle='--')
    axs[0].set_title(f"{labels[0]} over Time")
    axs[0].set_ylabel(labels[0])
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(t, y_euler[:, 1], label="Euler Method")
    axs[1].plot(t, y_runge_kutta[:, 1], label="Runge-Kutta Method", linestyle='--')
    axs[1].set_title(f"{labels[1]} over Time")
    axs[1].set_ylabel(labels[1])
    axs[1].legend()
    axs[1].grid(True)

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()


def gen_data(t, y):
    """Generate test and train data from the solution of the numerical method."""
    t_sliced, y_sliced = (
        t[jnp.arange(t.size, step=200)],
        y[jnp.arange(t.size, step=200)],
    )
    split_index = int(0.8 * len(t_sliced))
    t_train, y_train = t_sliced[:split_index], y_sliced[:split_index, 0]
    t_test, y_test = t_sliced[split_index:], y_sliced[split_index:, 0]
    return t_train, y_train, t_test, y_test


if __name__ == "__main__":
    # TODO: Add plotting functionality
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    
    # Euler Method
    t_euler, y_euler = solve_pendulum_euler(system_ode, y0, t_span, dt, b, m, l, g)
    
    # Runge–Kutta Method
    t_rk, y_rk = runge_kutta_4(system_ode, y0, t_span, dt, b, m, l, g)
    
    # Plot the solutions
    plot_solution(t_euler, y_euler, y_rk, labels=["Theta (rad)", "Omega (rad/s)"])
    
    # Generate data
    t_train, y_train, t_test, y_test = gen_data(t_rk, y_rk)
    
    
    
    
