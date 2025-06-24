import jax
import jax.numpy as jnp
import pickle

from src.agents.functions.networks import GaussianActor, ClassicalActor
from src.envs.universal_physics_plotter import universal_physics_plotter
from src.envs.supervisory.env_wrapped_supervisory import supervisory_wrapper
from src.envs.utils.input_normalisation import find_input_normalisation_vals

# Function to load the model parameters
def load_supervisory_weights(flight_phase='subsonic'):
    # As supervisory learning is always done with a TD3 agent, add extra std layer
    filename=f'data/agent_saves/SupervisoryLearning/{flight_phase}/supervisory_network.pkl'
    with open(filename, 'rb') as f:
        params = pickle.load(f)

    hidden_layers = -2 # input and output (x2) layers
    for key in params:
        hidden_layers += 1
    hidden_dim = len(params["Dense_0"]["bias"])

    loaded_actor_params_clean = {}
    loaded_actor_params_clean['params'] = params
    return loaded_actor_params_clean, hidden_dim, hidden_layers

def load_supervisory_actor(flight_phase,
                   rl_type: str):
    assert rl_type in ['sac', 'td3'] , f"rl_type must be either 'sac' or 'td3', not {rl_type}"
    assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']

    if flight_phase == 'subsonic':
        action_dim_needed = 2
    elif flight_phase == 'supersonic':
        action_dim_needed = 2
    elif flight_phase == 'flip_over_boostbackburn':
        action_dim_needed = 1
    elif flight_phase == 'ballistic_arc_descent':
        action_dim_needed = 1
    elif flight_phase == 'landing_burn':
        action_dim_needed = 4
    elif flight_phase == 'landing_burn_pure_throttle':
        action_dim_needed = 1
    elif flight_phase == 'landing_burn_pure_throttle_Pcontrol':
        action_dim_needed = 1
    else:
        raise ValueError(f'Invalid flight phase: {flight_phase}')

    params, hidden_dim, number_of_hidden_layers = load_supervisory_weights(flight_phase)
    
    state_dim = params['params']['Dense_0']['kernel'].shape[0]  
    hidden_dim = params['params']['Dense_0']['kernel'].shape[1]  # 10
    action_dim = params['params'][f'Dense_{len(params["params"])-1}']['bias'].shape[0]  # 3
    assert action_dim == action_dim_needed, f"Action dimension mismatch: {action_dim} != {action_dim_needed}"

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
    
    return network, new_params, hidden_dim, number_of_hidden_layers

class Agent_Supervisory_Learnt:
    def __init__(self,
                 flight_phase='subsonic',
                 rl_type: str = 'sac'):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']
        assert rl_type in ['sac', 'td3'] , "rl_type must be either 'sac' or 'td3'"
        self.flight_phase = flight_phase
        self.rl_type = rl_type
        self.actor, self.actor_params, _, _ = load_supervisory_actor(flight_phase=flight_phase, rl_type=rl_type)

    def select_actions_no_stochastic(self, state):
        if self.rl_type == 'sac':
            mean, std = self.actor.apply(self.actor_params, state)
            return mean
        elif self.rl_type == 'td3':
            action = self.actor.apply(self.actor_params, state)
            return action
    
def plot_trajectory_supervisory(flight_phase='subsonic',
                                enable_wind = False,
                                stochastic_wind = False,
                                horiontal_wind_percentile = 95):
    input_normalisation_values = find_input_normalisation_vals(flight_phase=flight_phase)
    
    env = supervisory_wrapper(input_normalisation_values = input_normalisation_values,
                              flight_phase=flight_phase,
                              enable_wind = enable_wind,
                              stochastic_wind = stochastic_wind,
                              horiontal_wind_percentile = horiontal_wind_percentile)
    agent = Agent_Supervisory_Learnt(flight_phase=flight_phase, rl_type='td3')
    save_path = f'results/SupervisoryLearning/{flight_phase}/'
    if flight_phase in ['landing_burn', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
        reward_total, y_array = universal_physics_plotter(env,
                                                          agent,
                                                          save_path,
                                                          flight_phase = flight_phase,
                                                          type='supervisory')
    else:
        universal_physics_plotter(env,
                                  agent,
                                  save_path,
                                  flight_phase = flight_phase,
                                  type='supervisory')