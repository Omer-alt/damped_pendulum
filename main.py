import jax
import jax.numpy as jnp

import hydra
from omegaconf import DictConfig

from ode_train import plot_metrics, plot_ode_loss, train_ode_model
from train import MLP, train
from data_generator import gen_data, runge_kutta_4, system_ode

# Hydra entry point
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    y0 = jnp.array([2 * jnp.pi / 3, 0.0])
    
    t_span = (cfg.solver.t_start, cfg.solver.t_end)
    dt =  cfg.solver.dt
    b = cfg.prop.b
    m = cfg.prop.m
    l = cfg.prop.l
    g = cfg.prop.g
    data = runge_kutta_4(system_ode, y0, t_span, dt, b, m, l, g) #  Output from the KR method function.
      
    key = jax.random.PRNGKey(0)
    model_params = cfg.model.hidden_layers
    model = MLP(model_params)
    learning_rate = cfg.training.learning_rate
    epochs = cfg.training.epochs
    _, ode_metrics_history = train_ode_model(key, (data[0], None), model, epochs, learning_rate) # Output from the ode training function.
    
    # TODO: Add plotting functionality
    plot_ode_loss(ode_metrics_history, save_dir="./plots")
    
    data = gen_data(data[0], data[1])
    _, train_metrics_history, _ = train(model, data, learning_rate, epochs, (1,), key)  # The output from the train function.
    
    plot_metrics(train_metrics_history, ode_metrics_history, labels=["Loss with data driven approch", "ODE Loss over Epochs"], save_dir="./plots")
    
if __name__ == "__main__":
    main()