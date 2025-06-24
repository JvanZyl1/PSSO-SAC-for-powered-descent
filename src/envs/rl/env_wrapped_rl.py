import jax
import math
import numpy as np
import gymnasium as gym
import jax.numpy as jnp

from src.envs.base_environment import rocket_environment_pre_wrap
from src.envs.utils.input_normalisation import find_input_normalisation_vals
from src.envs.load_initial_states import load_landing_burn_initial_state

class GymnasiumWrapper:
    def __init__(self,
                 env: gym.Env):
        self.env = env
    def reset(self):
        state  = self.env.reset()
        processed_state = self._process_state(state)  
        return processed_state
    
    def augment_action(self, action): # child
        return action
    
    def augment_state(self, state): # child
        return state

    def step(self, action):
        if isinstance(action, jnp.ndarray):
            action = np.array(jax.device_get(action))  # JAX -> numpy
        action = self.augment_action(action)
        if action.ndim == 2:
            action = action[0]
        state, reward, done, truncated, info = self.env.step(action)
        processed_state = self._process_state(state)
        return processed_state, float(reward), bool(done), bool(truncated), info

    def _process_state(self, state):
        if isinstance(state, tuple):
            state = state[0]
        state = jnp.asarray(state, dtype=jnp.float32).reshape(-1)
        state = self.augment_state(state)
        return state

    def render(self):
        pass

    def close(self):
        pass
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model

def maximum_velocity(y, vy):
    air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(float(y))
    if speed_of_sound != 0:
        v_max = math.sqrt(2*atmospheric_pressure/air_density)
    else:
        v_max = vy
    return v_max

class rl_wrapped_env(GymnasiumWrapper):
    def __init__(self,
                 flight_phase: str = 'subsonic',
                 enable_wind: bool = False,
                 stochastic_wind: bool = True,
                 horiontal_wind_percentile: int = 50,
                 trajectory_length: int = None,
                 discount_factor: float = None):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']
        self.flight_phase = flight_phase
        env = rocket_environment_pre_wrap(type = 'rl',
                                          flight_phase = flight_phase,
                                          enable_wind = enable_wind,
                                          stochastic_wind = stochastic_wind,
                                          horiontal_wind_percentile = horiontal_wind_percentile,
                                          trajectory_length = trajectory_length,
                                          discount_factor = discount_factor)
        self.enable_wind = enable_wind
        # State : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
        if self.flight_phase in ['subsonic', 'supersonic']:
            self.state_dim = 8
            self.action_dim = 2
        elif self.flight_phase == 'flip_over_boostbackburn':
            self.state_dim = 2
            self.action_dim = 1
        elif self.flight_phase == 'ballistic_arc_descent':
            self.state_dim = 4
            self.action_dim = 1
        elif self.flight_phase == 'landing_burn':
            self.state_dim = 5
            self.action_dim = 4
        elif self.flight_phase == 'landing_burn_ACS':
            self.state_dim = 5
            self.action_dim = 3
        elif self.flight_phase == 'landing_burn_pure_throttle':
            self.state_dim = 2
            self.action_dim = 1
        elif self.flight_phase == 'landing_burn_pure_throttle_Pcontrol':
            self.state_dim = 1
            self.action_dim = 1

        self.input_normalisation_vals = find_input_normalisation_vals(flight_phase)

        if flight_phase == 'landing_burn_pure_throttle_Pcontrol':
            initial_state = load_landing_burn_initial_state()
            x0, y0, vx0, vy0, _, _, _, _, _, _, _ = initial_state
            self.speed0 = math.sqrt(vx0**2 + vy0**2)
        super().__init__(env)
    
    def truncation_id(self):
        return self.env.truncation_id

    def augment_action(self, actions):
        if self.flight_phase == 'landing_burn':
            '''
            u0 is gimbal angle norm from -1 to 1
            u1 is non nominal throttle from -1 to 1
            u2 is left deflection command norm from -1 to 1
            u3 is right deflection command norm from -1 to 1
            For gimbal angle a logarithmic-like scalling is used.
            Which is: smooth, monotomic and numerically safe.
            Works for:
                a) Penalise overaction.
                b) Large actions cause instability.
                c) Majority of control around equilibrium.
            As such using a' = sign(a) * log(1 + c * |a|)/log(1+c)
                - Is still bounded between [-1, 1]
                - Can use Desmos graphing calculator to tune it: https://www.desmos.com/calculator
                - c is essentially the compression factor, larger c -> sharper supression.
                - For absolute actions:
                a       c=5     c= 10
                0.0     0.0     0.0
                0.1     0.26    0.41
                0.5     0.7     0.80
                1.0     1.0     1.0
            '''
            if actions.ndim == 2:
                u0, u1, u2, u3 = actions[0]
            else:
                u0, u1, u2, u3 = actions
            c_gimbal = 10
            u0_aug = math.copysign(math.log(1 + c_gimbal * abs(u0))/math.log(1+c_gimbal),u0)
            u1_aug = u1 # No scalling is needed
            c_deflection = 5
            u2_aug = math.copysign(math.log(1 + c_deflection * abs(u2))/math.log(1+c_deflection), u2)
            u3_aug = math.copysign(math.log(1 + c_deflection * abs(u3))/math.log(1+c_deflection), u3)
            if actions.ndim == 2:
                actions = np.array([[u0_aug, u1_aug, u2_aug, u3_aug]])
            else:
                actions = np.array([u0_aug, u1_aug, u2_aug, u3_aug])
        if self.flight_phase == 'landing_burn_pure_throttle_Pcontrol':
            if actions.ndim == 2:
                u0 = actions[0]
            else:
                u0 = actions
            u0_aug = (u0 + 1)/2 * self.speed0
            actions = np.array([u0_aug])
        return actions
    
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
        elif self.flight_phase in ['landing_burn', 'landing_burn_ACS']:
            y = y/self.input_normalisation_vals[0]
            vy = vy/self.input_normalisation_vals[1]

            theta_deviation_max_guess = math.radians(5)
            k_theta = float(np.arctanh(0.75)/theta_deviation_max_guess)
            theta = math.tanh(k_theta*(theta - math.pi/2))

            theta_dot_max_guess = 0.01 # may have outlier so set k so tanh(k*theta_dot_max_guess) = 0.75
            k_theta_dot = float(np.arctanh(0.75)/theta_dot_max_guess)
            theta_dot = math.tanh(k_theta_dot*theta_dot)

            gamma_deviation_max_guess = math.radians(5)
            k_gamma = float(np.arctanh(0.75)/gamma_deviation_max_guess)
            gamma = math.tanh(k_gamma*(gamma - 3/2 * math.pi))
            action_state = np.array([y, vy, theta, theta_dot, gamma])

        elif self.flight_phase == 'landing_burn_pure_throttle':
            y = (1-y/self.input_normalisation_vals[0])*2-1
            vy = (1-vy/self.input_normalisation_vals[1])*2-1
            action_state  = np.array([y, vy])
        elif self.flight_phase == 'landing_burn_pure_throttle_Pcontrol':
            y = (1-y/self.input_normalisation_vals[0])*2-1
            action_state = np.array([y])
        return action_state

    def close(self): # check
        pass
