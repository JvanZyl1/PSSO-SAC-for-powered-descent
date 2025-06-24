import math
import torch.nn as nn
import numpy as np
import torch
import pickle

from src.envs.utils.input_normalisation import find_input_normalisation_vals
from src.envs.base_environment import rocket_environment_pre_wrap
from src.envs.universal_physics_plotter import universal_physics_plotter
'''
class model:
    def __init__(self):
    def objective_function(individual):        
    def reset(self):
    def individual_update_model(self, individual):
    def plot_results(self):
'''
class simple_actor:
    def __init__(self,
                 number_of_hidden_layers = 15,
                 hidden_dim = 10,
                 output_dim = 2,
                 input_dim = 7,
                 flight_phase = 'subsonic'):
        self.number_of_hidden_layers = number_of_hidden_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(number_of_hidden_layers)],
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        self.number_of_network_parameters = sum(p.numel() for p in self.network.parameters())


    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        action = self.network(state)
        return action
        
    def update_individiual(self, individual):
        remaining_elements = individual
        param_index = 0
        for name, param in self.network.named_parameters():
            param_size = param.numel()
            if param_index + param_size <= len(remaining_elements):
                param_data = torch.tensor(
                    remaining_elements[param_index:param_index+param_size], 
                    dtype=torch.float32
                ).view(param.shape)
                param.data = param_data
                param_index += param_size
            else:
                print(f"Warning: Not enough elements in individual for parameter {name}")

    def return_setup_vals(self):
        # returns dict with the params
        mock_individual_dictionary = {}
        bounds = []
        bound_scale = 1.5
        
        param_index = 0
        for name, param in self.network.named_parameters():
            flat_param = param.data.flatten().tolist()
            for j, val in enumerate(flat_param):
                mock_individual_dictionary[f'{name.replace(".", "_")}_{j}'] = val
                bounds.append((-bound_scale, bound_scale))
                param_index += 1
        
        return mock_individual_dictionary, bounds

class pso_wrapper:
    def __init__(self,
                 flight_phase = 'subsonic',
                 enable_wind = False,
                 stochastic_wind = False,
                 horiontal_wind_percentile = 95):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn_pure_throttle', 'landing_burn']
        self.flight_phase = flight_phase
        self.enable_wind = enable_wind
        self.env = rocket_environment_pre_wrap(type = 'pso',
                                               flight_phase = self.flight_phase,
                                               enable_wind = enable_wind,
                                               stochastic_wind = stochastic_wind,
                                               horiontal_wind_percentile = horiontal_wind_percentile)
        self.initial_mass = self.env.reset()[-2]
        self.input_normalisation_vals = find_input_normalisation_vals(self.flight_phase)
        
    def truncation_id(self):
        return self.env.truncation_id

    def augment_state(self, state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if self.flight_phase in ['subsonic', 'supersonic']:
            action_state = np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])
            action_state /= self.input_normalisation_vals
        elif self.flight_phase == 'flip_over_boostbackburn':
            action_state = np.array([theta, theta_dot])
            action_state /= self.input_normalisation_vals
        elif self.flight_phase == 'ballistic_arc_descent':
            action_state = np.array([theta, theta_dot, gamma, alpha])
            action_state /= self.input_normalisation_vals
        elif self.flight_phase == 'landing_burn_pure_throttle':
            y = y/self.input_normalisation_vals[0]
            vy = vy/self.input_normalisation_vals[1]
            action_state = np.array([y, vy])
        elif self.flight_phase == 'landing_burn':
            y = y/self.input_normalisation_vals[0]
            vy = vy/self.input_normalisation_vals[1]

            theta_deviation_max_guess = math.radians(25)
            k_theta = float(np.arctanh(0.75)/theta_deviation_max_guess)
            theta = math.tanh(k_theta*(theta - math.pi/2))

            x = x/self.input_normalisation_vals[-2]
            vx = vx/self.input_normalisation_vals[-1]
            action_state = np.array([x, y, vx, vy, theta])
        return action_state
    
    def step(self, action):
        action_detached = action.detach().numpy()
        state, reward, done, truncated, info = self.env.step(action_detached)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        return state
        

class pso_wrapped_env:
    def __init__(self,
                 flight_phase = 'subsonic',
                 enable_wind = False,
                 stochastic_wind = False,
                 horiontal_wind_percentile = 50):
        self.enable_wind = enable_wind
        self.env = pso_wrapper(flight_phase = flight_phase,
                               enable_wind = enable_wind,
                               stochastic_wind = stochastic_wind,
                               horiontal_wind_percentile = horiontal_wind_percentile)
        
        # Initialise the network with correct input dimension (7 for x, y, vx, vy, theta, theta_dot, alpha, mass)
        if flight_phase == 'subsonic':
            self.actor = simple_actor(input_dim=8,
                                      output_dim=2,
                                      number_of_hidden_layers = 14,
                                      hidden_dim = 50,
                                      flight_phase = flight_phase) # 2 actions: u0, u1
        elif flight_phase == 'supersonic':
            self.actor = simple_actor(input_dim=8,
                                      output_dim=2,
                                      number_of_hidden_layers = 10,
                                      hidden_dim = 50,
                                      flight_phase = flight_phase) # 2 actions: u0, u1
        elif flight_phase == 'flip_over_boostbackburn':
            self.actor = simple_actor(input_dim=2,
                                      output_dim=1,
                                      number_of_hidden_layers = 10,
                                      hidden_dim = 8,
                                      flight_phase = flight_phase) # 1 actions: u0
        elif flight_phase == 'ballistic_arc_descent':
            self.actor = simple_actor(input_dim=4,
                                      output_dim=1,
                                      number_of_hidden_layers = 10,
                                      hidden_dim = 8,
                                      flight_phase = flight_phase) # 1 actions: u0
        elif flight_phase == 'landing_burn_pure_throttle':
            self.actor = simple_actor(input_dim=2,
                                      output_dim=1,
                                      number_of_hidden_layers = 3,
                                      hidden_dim = 8,
                                      flight_phase = flight_phase) # 1 actions: u0
        elif flight_phase == 'landing_burn':
            self.actor = simple_actor(input_dim=5,
                                      output_dim=4, # 4 actions: u0, u1, u2, u3 : throttle, gimbal, acs left, acs right
                                      number_of_hidden_layers = 4,
                                      hidden_dim = 8,
                                      flight_phase = flight_phase) # 1 actions: u0
        self.flight_phase = flight_phase
        self.mock_dictionary_of_opt_params, self.bounds = self.actor.return_setup_vals()
        self.experience_buffer = []
        self.save_interval_experience_buffer = 1000       # So saves roughly every generation of a 1000 particle swarm.
        self.episode_idx = 0


    def individual_update_model(self, individual):
        self.actor.update_individiual(individual)

    def reset(self):
        _ = self.env.reset()
        self.experience_buffer = []

    def objective_function(self, individual):
        self.individual_update_model(individual)
        state = self.env.reset()

        done_or_truncated = False
        episode_reward = 0
        previous_action = None
        previous_state = state
        previous_reward = None
        while not done_or_truncated:
            action = self.actor.forward(state)
            state, reward, done, truncated, info = self.env.step(action)
            done_or_truncated = done or truncated
            episode_reward -= reward # As minimisation problem
            if previous_action is not None:
                self.experience_buffer.append((previous_state, previous_action, previous_reward, state, action))
            previous_action = action
            previous_state = state  
            previous_reward = reward

        self.episode_idx += 1

        return episode_reward
    
    def plot_results(self, individual, save_path):
        self.individual_update_model(individual)
        universal_physics_plotter(self.env,
                                  self.actor,
                                  save_path,
                                  flight_phase = self.flight_phase,
                                  type = 'pso')