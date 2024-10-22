import time
import os
import jax.numpy as jnp
from jax import lax, block_until_ready, jit
import matplotlib.pyplot as plt
from functools import partial


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
    domega_dt = -(b / (m)) * omega - (g / l) * jnp.sin(theta)
    
    return jnp.array([dtheta_dt, domega_dt])


@jit
def system_ode_with_jit(y, b, m, l, g):
    
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(b / (m)) * omega - (g / l) * jnp.sin(theta)
    
    return jnp.array([dtheta_dt, domega_dt])

# Check input parameters before using JIT-compiled function
def check_parameters(m, l, g):
    
    assert m > 0, "Mass (m) must be positive"
    assert l > 0, "Length (l) must be positive"
    assert g > 0, "Gravity (g) must be positive"


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
    
    if dt <= 0:
        raise ValueError("Time step (dt) must be positive.")
    if t_span[1] <= t_span[0]:
        raise ValueError("Invalid time range in t_span.")
    
    y = jnp.zeros((len(t), len(y0)))
    y = y.at[0].set(y0)
    
    for i in range(1, len(t)):
        y = y.at[i].set(y[i - 1] + dt * pendulum_ode(y[i - 1], b, m, l, g))
    
    return t, y


def solve_pendulum_euler_with_scan(pendulum_ode, y0, t_span, dt, b, m, l, g):
    "Euler method using JAX scan"
    
    t0, t_n = t_span
    t = jnp.arange(t0, t_n, dt)

    def step_fn(y, _):
        y_next = y + dt * pendulum_ode(y, b, m, l, g)
        return y_next, y_next

    _, y = lax.scan(step_fn, y0, t)
    
    return t, y


@jit
def solve_pendulum_euler_with_jit( y0, t, dt, b, m, l, g):
    "JIT-compiled Euler method"
    
    y = jnp.zeros((len(t), len(y0)))
    y = y.at[0].set(y0)
    
    for i in range(1, len(t)):
        y = y.at[i].set(y[i - 1] + dt * system_ode_with_jit(y[i - 1], b, m, l, g))
    
    return t, y


@jit
def solve_pendulum_euler_with_scan_and_jit(y0, t, dt, b, m, l, g):
    "JIT-compiled Euler method using JAX scan"
    

    def step_fn(y, _):
        y_next = y + dt * system_ode_with_jit(y, b, m, l, g)
        return y_next, y_next

    _, y = lax.scan(step_fn, y0, t)
    
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

    for i in range(1, len(t )):
        k1 = pendulum_ode(y[i - 1], b, m, l, g)
        k2 = pendulum_ode(y[i - 1] + 0.5 * dt * k1, b, m, l, g)
        k3 = pendulum_ode(y[i - 1] + 0.5 * dt * k2, b, m, l, g)
        k4 = pendulum_ode(y[i - 1] + dt * k3, b, m, l, g)
        y = y.at[i].set(y[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        
    return t, y


def runge_kutta_4_with_scan(pendulum_ode, y0, t_span, dt, b, m, l, g):
    "Runge-Kutta method using JAX scan"
    
    t0, t_n = t_span
    t = jnp.arange(t0, t_n, dt)

    def step_fn(y, _):
        k1 = pendulum_ode(y, b, m, l, g)
        k2 = pendulum_ode(y + 0.5 * dt * k1, b, m, l, g)
        k3 = pendulum_ode(y + 0.5 * dt * k2, b, m, l, g)
        k4 = pendulum_ode(y + dt * k3, b, m, l, g)
        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, y_next

    _, y = lax.scan(step_fn, y0, t)
    
    return t, y


@jit
def runge_kutta_4_with_scan_and_jit( y0, t, dt, b, m, l, g):
    "Runge-Kutta method using JAX scan and jit"

    def step_fn(y, _):
        k1 = system_ode_with_jit(y, b, m, l, g)
        k2 = system_ode_with_jit(y + 0.5 * dt * k1, b, m, l, g)
        k3 = system_ode_with_jit(y + 0.5 * dt * k2, b, m, l, g)
        k4 = system_ode_with_jit(y + dt * k3, b, m, l, g)
        y_next = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return y_next, y_next

    _, y = lax.scan(step_fn, y0, t)
    
    return t, y


@jit
def runge_kutta_4_with_jit(y0, t, dt, b, m, l, g):
    "JIT-compiled Runge-Kutta method"
    
    y = jnp.zeros((len(t), len(y0)))
    y = y.at[0].set(y0)

    for i in range(1, len(t)):
        k1 = system_ode_with_jit(y[i - 1], b, m, l, g)
        k2 = system_ode_with_jit(y[i - 1] + 0.5 * dt * k1, b, m, l, g)
        k3 = system_ode_with_jit(y[i - 1] + 0.5 * dt * k2, b, m, l, g)
        k4 = system_ode_with_jit(y[i - 1] + dt * k3, b, m, l, g)
        y = y.at[i].set(y[i - 1] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))
        
    return t, y


#  TODO: Write the function for plotting the solution  Question 3.
def plot_solution(t, y_euler, y_runge_kutta, labels, save_dir="./plots" ):
    """
    Plot the solution obtained by Euler and Runge-Kutta methods.

    Parameters:
        t (jnp.array): Time vector.
        y_euler, y_runge_kutta (jnp.array): Solution from the solvers.
        labels (list): Labels for the plot (theta, omega).
    """
    os.makedirs(save_dir, exist_ok=True)
    
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
    plt.savefig(f"{save_dir}/solvers__curves.png")
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


@jit
def gen_data_with_jit(t, y):
    """Generate test and train data from the solution of the numerical method."""
    
    t_sliced, y_sliced = (
        t[jnp.arange(t.size, step=200)],
        y[jnp.arange(t.size, step=200)],
    )
    split_index = int(0.8 * len(t_sliced))
    t_train, y_train = t_sliced[:split_index], y_sliced[:split_index, 0]
    t_test, y_test = t_sliced[split_index:], y_sliced[split_index:, 0]
    return t_train, y_train, t_test, y_test


def comparaison_over_time(func, *argcs):
    start_time_euler = time.time()
    func(*argcs)
    return time.time() - start_time_euler


def comparaison_over_time(func, *args, block_func=None):
    """ Measures the execution time of the given function and ensures that the execution is complete."""
    
    start_time = time.time()
    result = func(*args)
    
    # Ensure execution is complete, if a blocking function is given
    if block_func is not None:
        block_func(result)
    
    return time.time() - start_time


def plot_execution_times_with_subplots(methods, execution_times, save_dir="./plots"):
    """
    Plot the execution time comparison with subplots based on the number of sublists in the 'methods' list.
    
    Parameters:
    - methods: A list of lists containing method names.
    - execution_times: A list of lists containing corresponding execution times for each method.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    num_plots = len(methods)
    
    fig, axs = plt.subplots(num_plots, 1, figsize=(8, num_plots * 4))  # Adjust figsize to make each plot visible

    if num_plots == 1:
        axs = [axs]

    # Loop through each method sublist and its corresponding execution times
    for i, (method_group, time_group) in enumerate(zip(methods, execution_times)):
        axs[i].bar(method_group, time_group, color=['blue', 'green', 'orange', 'red'])
        axs[i].set_ylabel('Execution Time (seconds)')
        axs[i].set_title(f'Execution Time Comparison: Plot {i + 1}')

    plt.tight_layout() 
    plt.savefig(f"{save_dir}/execution_times__curves.png")
    plt.show()

    
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
    
    """Measure execution time"""
    
    euler_time_without_scan = comparaison_over_time(
        solve_pendulum_euler, system_ode, y0, t_span, dt, b, m, l, g, 
        block_func=block_until_ready  
    )
    
    euler_time_with_scan = comparaison_over_time(
        solve_pendulum_euler_with_scan, system_ode, y0, t_span, dt, b, m, l, g, 
        block_func=block_until_ready  
    )
    
    start_time_euler_with_jit = time.time()
    t = jnp.arange(t_span[0], t_span[1], dt)
    # Check parameters before calling the JIT function
    check_parameters(m, l, g)
    _, y_euler_jit = solve_pendulum_euler_with_jit( y0, t, dt, b, m, l, g)
    block_until_ready(y_euler_jit) # Ensure execution is complete 
    euler_time_with_jit = time.time() - start_time_euler_with_jit
    
    start_time_euler_with_scan_and_jit = time.time()
    check_parameters(m, l, g)
    t = jnp.arange(t_span[0], t_span[1], dt)
    _, y_euler_scan_and_jit = solve_pendulum_euler_with_scan_and_jit( y0, t, dt, b, m, l, g)
    block_until_ready(y_euler_scan_and_jit) 
    euler_time_with_scan_and_jit = time.time() - start_time_euler_with_scan_and_jit
    
    print(f"Euler method execution time (with for loop) : {euler_time_without_scan:.6f} seconds")
    print(f"Euler method execution time (with JAX scan) : {euler_time_with_scan:.6f} seconds")
    print(f"Euler method execution time (with JIT) : {euler_time_with_jit:.6f} seconds")
    print(f"Euler method execution time (with JIT and JAX Scan) : {euler_time_with_scan_and_jit:.6f} seconds")
    
    # Runge–Kutta Method
    t_rk, y_rk = runge_kutta_4(system_ode, y0, jnp.array(t_span), dt, b, m, l, g)
    
    """Measure execution time"""
    rk_time_without_scan = comparaison_over_time(
        runge_kutta_4, system_ode, y0, t_span, dt, b, m, l, g, 
        block_func=block_until_ready  
    )
    
    rk_time_with_scan = comparaison_over_time(
        runge_kutta_4_with_scan, system_ode, y0, t_span, dt, b, m, l, g, 
        block_func=block_until_ready  
    )
    
    start_time_rk_jit = time.time()
    # check_parameters(m, l, g)
    t = jnp.arange(t_span[0], t_span[1], dt)
    _, y_rk_jit = runge_kutta_4_with_jit( y0, t, dt, b, m, l, g)
    block_until_ready(y_rk_jit) 
    rk_time_with_jit = time.time() - start_time_rk_jit
    
    start_time_rk_jit_scan = time.time()
    # check_parameters(m, l, g)
    t = jnp.arange(t_span[0], t_span[1], dt)
    _, y_rk_jit_and_jit = runge_kutta_4_with_scan_and_jit(y0, t, dt, b, m, l, g)
    block_until_ready(y_rk_jit_and_jit) 
    rk_time_with_jit_scan = time.time() - start_time_rk_jit_scan
    
    print(f"Runge-Kutta method execution time (with for loop) : {rk_time_without_scan:.6f} seconds")
    print(f"Runge-Kutta method execution time (with JAX scan) : {rk_time_with_scan:.6f} seconds")
    print(f"Runge-Kutta method execution time (with JIT): {rk_time_with_jit:.6f} seconds")
    print(f"Runge-Kutta method execution time (with JIT and JAX Scan): {rk_time_with_jit_scan:.6f} seconds")
        
    # Plot the solutions
    plot_solution(t_euler, y_euler, y_rk, labels=["Theta (rad)", "Omega (rad/s)"])
    
    # Generate data 
    data_time_without_jit = comparaison_over_time(
        runge_kutta_4_with_scan, system_ode, y0, t_span, dt, b, m, l, g, 
        block_func=block_until_ready  
    )
    start_time_d = time.time()
    t_train, y_train, t_test, y_test = gen_data(t_rk, y_rk)
    block_until_ready(y_test) # Ensure execution is complete
    data_time_without_jit = time.time() -start_time_d
    
    
    start_time_gen = time.time()
    _, _, _, y_test_jit = gen_data_with_jit(t_rk, y_rk)
    block_until_ready(y_test_jit) # Ensure execution is complete
    data_time_with_jit = time.time() - start_time_gen
    
    print(f"Data generation execution time (without JIT-compiled) : {data_time_without_jit:.6f} seconds")
    print(f"Data generation execution time (with JIT-compiled) : {data_time_with_jit:.6f} seconds")
    
    methods = [['Generat Data Without JIT', 'Generat Data  With JIT'], ['Euler With Loop', 'Euler With Scan' , 'Euler With JIT', 'Euler With Scan and JIT'], ['RK With Loop', 'RK With Scan', 'RK With JIT', 'RK With Scan and JIT']]
    execution_times = [[data_time_without_jit, data_time_with_jit],[ euler_time_without_scan, euler_time_with_scan, euler_time_with_jit, rk_time_with_jit_scan], [rk_time_without_scan, rk_time_with_scan, rk_time_with_jit, rk_time_with_jit_scan] ]

    plot_execution_times_with_subplots(methods, execution_times)


    
    
    