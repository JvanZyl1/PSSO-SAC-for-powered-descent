import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
from flax.core.frozen_dict import freeze
from src.agents.functions.networks import GaussianActor, ClassicalActor

def load_pso_weights(flight_phase):
    results_df = pd.read_csv(f'data/pso_saves/{flight_phase}/particle_subswarm_optimisation_results.csv')
    pso_row = results_df[results_df['Algorithm'] == 'Particle Subswarm Optimisation']
    if pso_row.empty:
        raise ValueError("Particle Swarm Optimisation results not found in CSV")
    
    weight_columns = [col for col in pso_row.columns if 'weight' in col]
    bias_columns = [col for col in pso_row.columns if 'bias' in col]
    
    layer_weights = {}
    layer_biases = {}
    
    for col in weight_columns:
        parts = col.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            layer_idx = int(parts[0])
            if layer_idx not in layer_weights:
                layer_weights[layer_idx] = []
            layer_weights[layer_idx].append((col, pso_row[col].values[0]))
    
    for col in bias_columns:
        parts = col.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            layer_idx = int(parts[0])
            if layer_idx not in layer_biases:
                layer_biases[layer_idx] = []
            layer_biases[layer_idx].append((col, pso_row[col].values[0]))
    
    all_layer_indices = sorted(set(list(layer_weights.keys()) + list(layer_biases.keys())))
    
    first_layer_weights = sorted(layer_weights[0], key=lambda x: int(x[0].split('_')[-1]))
    num_weights_first_layer = len(first_layer_weights)

    # Check if we have biases for the first layer
    if 0 in layer_biases and layer_biases[0]:
        first_layer_biases = sorted(layer_biases[0], key=lambda x: int(x[0].split('_')[-1]))
        hidden_dim = len(first_layer_biases)
    else:
        for idx in all_layer_indices[1:]:
            if idx in layer_biases and layer_biases[idx]:
                hidden_dim = len(layer_biases[idx])
                break
        else:
            raise ValueError("Debug")
    
    input_dim = num_weights_first_layer // hidden_dim
    
    output_layer_idx = all_layer_indices[-1]
    output_dim = len(layer_biases[output_layer_idx])
    
    params = {'params': {}}
    
    for i, layer_idx in enumerate(all_layer_indices):
        if layer_idx not in layer_weights:
            continue
            
        layer_weight_values = [w[1] for w in sorted(layer_weights[layer_idx], 
                                                   key=lambda x: int(x[0].split('_')[-1]))]
        
        if layer_idx in layer_biases:
            layer_bias_values = [b[1] for b in sorted(layer_biases[layer_idx], 
                                                     key=lambda x: int(x[0].split('_')[-1]))]
        else:
            if i == 0:
                layer_bias_values = [0.0] * hidden_dim
            elif i == len(all_layer_indices) - 1:
                layer_bias_values = [0.0] * output_dim
            else:
                layer_bias_values = [0.0] * hidden_dim
        
        if i == 0:  # Input layer
            weights = np.array(layer_weight_values).reshape(hidden_dim, input_dim).T
        elif i == len(all_layer_indices) - 1:  # Output layer
            weights = np.array(layer_weight_values).reshape(output_dim, hidden_dim).T
        else:  # Hidden layers
            weights = np.array(layer_weight_values).reshape(hidden_dim, hidden_dim).T
        
        params['params'][f'Dense_{i}'] = {
            'kernel': jnp.array(weights),
            'bias': jnp.array(layer_bias_values),
        }
    
    return freeze(params)

def load_pso_actor(flight_phase,
                   rl_type: str):
    assert rl_type in ['sac', 'td3'] , "rl_type must be either 'sac' or 'td3'"
    params = load_pso_weights(flight_phase)
    
    state_dim = params['params']['Dense_0']['kernel'].shape[0]  
    hidden_dim = params['params']['Dense_0']['kernel'].shape[1]  # 10
    action_dim = params['params'][f'Dense_{len(params["params"])-1}']['bias'].shape[0]  # 3
    number_of_hidden_layers = len(params['params']) - 2

    if rl_type == 'sac':
        network = GaussianActor(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            number_of_hidden_layers=number_of_hidden_layers
        )
    else:
        network = ClassicalActor(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            number_of_hidden_layers=number_of_hidden_layers
        )
    
    key = jax.random.PRNGKey(0)
    sample_state = jnp.zeros(state_dim)
    new_params = network.init(key, sample_state)
    
    for i in range(len(params['params'])):
        layer_name = f'Dense_{i}'
        new_params['params'][layer_name]['bias'] = params['params'][layer_name]['bias']
        new_params['params'][layer_name]['kernel'] = params['params'][layer_name]['kernel']
    
    return new_params, hidden_dim, number_of_hidden_layers