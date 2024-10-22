import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state
import matplotlib.pyplot as plt
import optax
import os

from data_generator import gen_data, runge_kutta_4, system_ode


class MLP(nn.Module):
    """Multilayer Perceptron model."""

    # TODO: Complete this class
    layers: tuple
    
    @nn.compact
    def __call__(self, x):
        for units in self.layers:
            x = nn.Dense(units)(x)
            x = nn.relu(x)  
        x = nn.Dense(1)(x)  # Output layer (1 output)
        return x
    
    
class TrainState(train_state.TrainState):
    metrics: dict


def create_train_state(model, init_key, learning_rate, input_shape):

    # TODO: Complete this function
    """Initialize the train state."""
    
    params = model.init(init_key, jnp.ones(input_shape))["params"]
    tx = optax.adam(learning_rate) # Define optimizer
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx, metrics={})


def mse_loss(params, apply_fn, batch):
    # TODO: Complete this function
    
    """Calculate Mean Squared Error loss."""
    inputs, targets = batch
    predictions = apply_fn({"params": params}, inputs)
    return jnp.mean((predictions - targets) ** 2)


@jax.jit
def compute_metrics(state, batch):
    # TODO: Complete this function
    
    """Compute loss metrics."""
    loss = mse_loss(state.params, state.apply_fn, batch)
    return {"loss": loss}


@jax.jit
def train_step(state, batch):
    # TODO: Complete this function
    
    """Single training step."""

    grad_fn = jax.grad(mse_loss)
    grads = grad_fn(state.params, state.apply_fn, batch)
    new_state = state.apply_gradients(grads=grads)
    return new_state, compute_metrics(state, batch)


@jax.jit
def val_step(state, batch):
    # TODO: Complete this function
    
    """Validation step to calculate metrics."""
    
    metrics = compute_metrics(state, batch)
    return metrics


# TODO: Write a train function to train a given model on a given data using a specified learning rate.
def train(model, data, learning_rate, epochs, input_shape, key):
    
    """Train the model and return the training state and metrics history."""
    
    init_key, _ = jax.random.split(key)
    train_state = create_train_state(model, init_key, learning_rate, input_shape)

    t_train, y_train, t_test, y_test = data  # Unpack the train and test data  
    
    # Reshape if necessary
    t_train = t_train.reshape(-1, 1)  # Shape (8, 1)
    y_train = y_train.reshape(-1, 1)  
    t_test = t_test.reshape(-1, 1)  
    y_test = y_test.reshape(-1, 1)  

    # Prepare the training and validation data
    train_data = (t_train, y_train)
    val_data = (t_test, y_test)
    
    train_metrics_history = []
    val_metrics_history = []

    for epoch in range(epochs):
        train_state, train_metrics = train_step(train_state, train_data)
        val_metrics = val_step(train_state, val_data)
        
        train_metrics_history.append(train_metrics['loss'])
        val_metrics_history.append(val_metrics)

        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Train Loss: {train_metrics['loss']}, Validation Loss: {val_metrics['loss']:.6f}")

    return train_state,  train_metrics_history, val_metrics_history

def plot_metrics(train_metrics, test_metrics, save_dir):
    """Plot and save training and test loss curves."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    train_losses = [m for m in train_metrics]
    test_losses = [m["loss"] for m in test_metrics]

    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/loss_mse_curves.png")
    plt.show()
    

if __name__ == "__main__":
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    t_span = (0, 20)
    dt = 0.01
    b = 0.3
    m = 1.0
    l = 1.0
    g = 9.81
    t, y = runge_kutta_4(system_ode, y0, t_span, dt, b, m, l, g)  #  Output from the KR method function.
    data = gen_data(t, y)

    key = jax.random.PRNGKey(0)
    model = MLP([16, 16, 16])
    learning_rate = 1e-3
    epochs = 100_000

    input_shape = (1,) 
    state, train_metrics_history, test_metrics_history = train(model, data, learning_rate, epochs, input_shape, key)  # The output from the train function.
    # TODO: Add plotting functionality
    plot_metrics(train_metrics_history, test_metrics_history, save_dir="./plots")
