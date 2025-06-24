import numpy as np
import math
from src.envs.base_environment import rocket_environment_pre_wrap
from src.envs.load_initial_states import load_landing_burn_initial_state

class supervisory_wrapper:
    def __init__(self,
                 input_normalisation_values,
                 flight_phase = 'subsonic',
                 enable_wind = False,
                 stochastic_wind = False,
                 horiontal_wind_percentile = 95):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']
        self.flight_phase = flight_phase
        self.input_normalisation_values = input_normalisation_values
        if flight_phase == 'landing_burn_pure_throttle':
            self.input_normalisation_values = input_normalisation_values[:2]
        self.enable_wind = False
        self.stochastic_wind = False
        self.horiontal_wind_percentile = 95
        self.env = rocket_environment_pre_wrap(type = 'supervisory',
                                               flight_phase = self.flight_phase,
                                               enable_wind = self.enable_wind,
                                               stochastic_wind = self.stochastic_wind,
                                               horiontal_wind_percentile = self.horiontal_wind_percentile)
        self.initial_mass = self.env.reset()[-2]
        if flight_phase == 'landing_burn_pure_throttle_Pcontrol':
            initial_state = load_landing_burn_initial_state()
            x0, y0, vx0, vy0, _, _, _, _, _, _, _ = initial_state
            self.speed0 = math.sqrt(vx0**2 + vy0**2)

    def truncation_id(self):
        return self.env.truncation_id

    def augment_state(self, state):# state used will be: [x, y, vx, vy, theta, theta_dot, alpha, mass]
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if self.flight_phase in ['subsonic', 'supersonic']:
            de_normalised_state = np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])
            action_state = de_normalised_state / self.input_normalisation_values
        elif self.flight_phase == 'flip_over_boostbackburn':
            de_normalised_state = np.array([theta, theta_dot])
            action_state = de_normalised_state / self.input_normalisation_values
        elif self.flight_phase == 'ballistic_arc_descent':
            de_normalised_state = np.array([theta, theta_dot, gamma, alpha])
            action_state = de_normalised_state / self.input_normalisation_values
        elif self.flight_phase == 'landing_burn':
            de_normalised_state = np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])
            action_state = de_normalised_state / self.input_normalisation_values
        elif self.flight_phase == 'landing_burn_pure_throttle':
            de_normalised_state = np.array([y, vy])
            y = (1-y/self.input_normalisation_values[0])*2-1
            vy = (1-vy/self.input_normalisation_values[1])*2-1
            action_state  = np.array([y, vy])
        elif self.flight_phase == 'landing_burn_pure_throttle_Pcontrol':
            y = (1-y/self.input_normalisation_values[0])*2-1
            action_state = np.array([y])
        return action_state
    
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
            u1_aug = u1
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
    
    def step(self, action):
        action_numpy = np.array(action)
        action = self.augment_action(action_numpy)
        state, reward, done, truncated, info = self.env.step(action)
        state = self.augment_state(state)
        return state, reward, done, truncated, info
    
    def reset(self):
        state = self.env.reset()
        state = self.augment_state(state)
        return state
