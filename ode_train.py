import os
import time
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, block_until_ready
import matplotlib.pyplot as plt
import optax

from train import MLP, create_train_state, train
from data_generator import gen_data, runge_kutta_4, system_ode


def ode_loss(params, apply_fn, batch, ode_params=(0.3, 1.0, 1.0, 9.81)):

    t, _ = batch
    b, m, l, g = ode_params
    # TODO: Complete this function
    t_resized = t.reshape(-1, 1)
        
    mlp_pred = apply_fn({'params': params}, t_resized)[0, 0]
    
    def theta(specific_time):
        return apply_fn({'params': params}, specific_time.reshape(-1, 1))[0, 0]
    
    d_mlp_dt = vmap(lambda specific_time: grad(theta)(specific_time) )(t)
    
    def d2theta_dt2(specific_time):
        return grad(theta)(specific_time)
    
    d2_mlp_dt2 = vmap(lambda specific_time: grad(d2theta_dt2)(specific_time))(t)
    
    ode_residual = d2_mlp_dt2 + (b / m) * d_mlp_dt + (g / l) * jnp.sin(mlp_pred)
    
    initial_condition_angle = (mlp_pred - 2 * jnp.pi / 3) ** 2
    initial_condition_velocity = (d_mlp_dt[0]) ** 2
    
    total_loss = jnp.mean(ode_residual ** 2)  + initial_condition_angle + initial_condition_velocity

    return total_loss

@jax.jit
def ode_train_step(state, batch):
    """A train step using the ode_loss."""
    
    # TODO: Complete this function
    params = state.params
    loss_fn = lambda p: ode_loss(p, state.apply_fn, batch)

    # Compute gradients
    grads = jax.grad(loss_fn)(params)
    state = state.apply_gradients(grads=grads) 
    
    loss = loss_fn(params)

    return state, loss

# TODO: Write a function for training the model using the ODE loss.
def train_ode_model(key, batch, model, epochs, learning_rate):
    """Train the model for a set number of epochs."""

    input_shape = (1, 1)
    init_key, _ = jax.random.split(key)
    state = create_train_state(model, init_key, learning_rate, input_shape)
    
    losses = []

    for epoch in range(epochs):
        # Perform one ODE train step
        state, loss = ode_train_step(state, batch)
        losses.append(loss)

        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return state, losses

@jit
def train_ode_model_with_jit(key, batch, model_params, epochs, learning_rate):
    """Train the model for a set number of epochs, accelerated with JIT."""
    input_shape = (1, 1)
    model = MLP(model_params)
    init_key, _ = jax.random.split(key)
    state = create_train_state(model, init_key, learning_rate, input_shape)

    losses = []

    for epoch in range(epochs):
        # Perform one ODE train step with JIT acceleration
        state, loss = ode_train_step(state, batch)
        losses.append(loss)

        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    return state, losses

def plot_ode_loss(ode_metrics_history, save_dir):
    """
        Plot the ODE loss over epochs.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join( save_dir , "loss_ode_curves.png")
    
    plt.plot(ode_metrics_history, label="ODE loss")
    plt.title('ODE Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()  
    plt.grid(True)  
    plt.savefig(save_path)
    plt.show()

def plot_metrics(data_driven_metrics, scientific_metrics, labels, save_dir):
    """Plot and save training and test loss curves."""
    
    os.makedirs(save_dir, exist_ok=True)

    train_losses_1 = [m for m in data_driven_metrics]
    train_losses_2 = [ m for m in scientific_metrics ]

    plt.plot(train_losses_1, label=labels[0])
    plt.plot(train_losses_2, label=labels[1])
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/comparison_curves.png")
    plt.show()
    

if __name__ == "__main__":
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    data = runge_kutta_4(system_ode, y0, t_span, dt, b, m, l, g) #  Output from the KR method function.
    # data = gen_data(t, y)  
    key = jax.random.PRNGKey(0)
    model_params = [16, 16, 16]
    model = MLP([16, 16, 16])
    learning_rate = 1e-3
    epochs = 100_000
    

    
    # Using the JIT-accelerated version
    # start_time_train_jit = time.time()
    # _, ode_metrics_history_jit = train_ode_model_with_jit(key, (data[0], None), model_params, epochs, learning_rate)
    # block_until_ready(ode_metrics_history_jit) # Ensure execution is complete
    # train_time_with_jit = time.time() - start_time_train_jit
    
    start_time_train = time.time()
    state, ode_metrics_history = train_ode_model(key, (data[0], None), model, epochs, learning_rate) # Output from the ode training function.
    block_until_ready(state) # Ensure execution is complete
    train_time_without_jit = time.time() - start_time_train
    
    print(f"Training ODE execution time (without JIT-compiled) : { train_time_without_jit:.6f} seconds")
    # print(f"Training ODE execution time (with JIT-compiled) : {train_time_with_jit:.6f} seconds")

    # TODO: Add plotting functionality
    plot_ode_loss(ode_metrics_history, save_dir="./plots")
    
    data = gen_data(data[0], data[1])
    _, train_metrics_history, _ = train(model, data, learning_rate, epochs, (1,), key)  # The output from the train function.
    
    plot_metrics(train_metrics_history, ode_metrics_history, labels=["Loss with data driven approch", "ODE Loss over Epochs"], save_dir="./plots")
 
    
    
    
    
    
    
