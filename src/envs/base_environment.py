import math

from src.envs.universal_physics_plotter import universal_physics_plotter
from src.envs.rockets_physics import compile_physics
from src.envs.rl.rtd_rl import compile_rtd_rl
from src.envs.pso.rtd_pso import compile_rtd_pso
from src.envs.supervisory.rtd_supervisory_mock import compile_rtd_supervisory_test
from src.RocketSizing.main_sizing import size_rocket
from src.envs.wind.full_wind_model import WindModel
from src.envs.load_initial_states import load_subsonic_initial_state, load_supersonic_initial_state, load_flip_over_initial_state, load_high_altitude_ballistic_arc_initial_state, load_landing_burn_initial_state

class rocket_environment_pre_wrap:
    def __init__(self,
                 type = 'rl',
                 flight_phase = 'subsonic',
                 enable_wind = True,
                 stochastic_wind = True,
                 horiontal_wind_percentile = 50,
                 trajectory_length = 100,
                 discount_factor = 0.99):
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']
        self.flight_phase = flight_phase
        self.dt = 0.1
        if flight_phase == 'subsonic':
            self.state_initial = load_subsonic_initial_state()
        elif flight_phase == 'supersonic':
            self.state_initial = load_supersonic_initial_state()
        elif flight_phase == 'flip_over_boostbackburn':
            self.state_initial = load_flip_over_initial_state()
            self.gimbal_angle_deg = 0.0
        elif flight_phase == 'ballistic_arc_descent':
            self.state_initial = load_high_altitude_ballistic_arc_initial_state()
        elif flight_phase == 'landing_burn':
            self.state_initial = load_landing_burn_initial_state()
            self.gimbal_angle_deg_prev = 0.0
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
        elif flight_phase == 'landing_burn_ACS':
            self.state_initial = load_landing_burn_initial_state()
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
        elif flight_phase == 'landing_burn_pure_throttle':
            self.state_initial = load_landing_burn_initial_state()
        elif flight_phase == 'landing_burn_pure_throttle_Pcontrol':
            self.state_initial = load_landing_burn_initial_state()
        # windd
        self.enable_wind = enable_wind
        if enable_wind:
            self.horiontal_wind_percentile = horiontal_wind_percentile
            self.wind_generator = WindModel(self.dt, stochastic_wind = stochastic_wind, given_percentile = self.horiontal_wind_percentile)
        else:
            self.wind_generator = None
        
        self.physics_step = compile_physics(self.dt,
                                            flight_phase=flight_phase)
        
        self.state = self.state_initial
        self.type = type

        assert type in ['rl', 'pso', 'supervisory']

        if type == 'rl':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_rl(flight_phase = flight_phase,
                                                                                   trajectory_length = trajectory_length,
                                                                                   discount_factor = discount_factor,
                                                                                   dt = self.dt)
        elif type == 'pso':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_pso(flight_phase = flight_phase)
        elif type == 'supervisory':
            self.reward_func, self.truncated_func, self.done_func = compile_rtd_supervisory_test(flight_phase = flight_phase)

        self.reset()
        if type == 'pso':
            self.truncation_id = 0
        self.previous_state = self.state
        self.g_loads_window = []
        self.g_load_window_time = 1.0
        self.g_load_window_length = int(self.g_load_window_time/self.dt)

    def reset(self):
        self.state = self.state_initial
        self.previous_state = self.state
        if self.type == 'pso':
            self.truncation_id = 0
        if self.flight_phase == 'flip_over_boostbackburn':
            self.gimbal_angle_deg = 0.0
        elif self.flight_phase == 'landing_burn':
            self.gimbal_angle_deg_prev = 0.0
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
        elif self.flight_phase == 'landing_burn_ACS':
            self.delta_command_left_rad_prev = 0.0
            self.delta_command_right_rad_prev = 0.0
        if self.enable_wind:
            self.wind_generator.reset()
        self.g_loads_window = []
        return self.state

    def step(self, actions):
        # Physics step
        if self.flight_phase in ['subsonic', 'supersonic', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
            self.state, info = self.physics_step(self.state,
                                                    actions,
                                                    wind_generator=self.wind_generator)
        elif self.flight_phase == 'flip_over_boostbackburn':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 self.gimbal_angle_deg,
                                                 wind_generator=self.wind_generator)
            self.gimbal_angle_deg = info['action_info']['gimbal_angle_deg']
        elif self.flight_phase == 'ballistic_arc_descent':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 wind_generator=self.wind_generator)
        elif self.flight_phase == 'landing_burn':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 self.gimbal_angle_deg_prev,
                                                 self.delta_command_left_rad_prev,
                                                 self.delta_command_right_rad_prev,
                                                 wind_generator=self.wind_generator)
            self.gimbal_angle_deg_prev = info['action_info']['gimbal_angle_deg']
            self.delta_command_left_rad_prev = info['action_info']['delta_command_left_rad']
            self.delta_command_right_rad_prev = info['action_info']['delta_command_right_rad']
        elif self.flight_phase == 'landing_burn_ACS':
            self.state, info = self.physics_step(self.state,
                                                 actions,
                                                 self.delta_command_left_rad_prev,
                                                 self.delta_command_right_rad_prev,
                                                 wind_generator=self.wind_generator)
            self.delta_command_left_rad_prev = info['action_info']['delta_command_left_rad']
            self.delta_command_right_rad_prev = info['action_info']['delta_command_right_rad']
            
        info['state'] = self.state
        info['actions'] = actions
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = self.state
        xp, yp, vxp, vyp, thetap, theta_dotp, gammamp, alphap, massp, mass_propellantp, timep = self.previous_state
        v = math.sqrt(vx**2 + vy**2)
        v_p = math.sqrt(vxp**2 + vyp**2)
        v_diff = abs(v - v_p)
        g_load = v_diff/self.dt * 1/9.81
        # Add gload untill full window
        if len(self.g_loads_window) < self.g_load_window_length:
            self.g_loads_window.append(g_load)
        else:
            self.g_loads_window.pop(0)
            self.g_loads_window.append(g_load)
        # Window mean
        info['g_load_1_sec_window'] = sum(self.g_loads_window)/self.g_load_window_length
        truncated, self.truncation_id = self.truncated_func(self.state, self.previous_state, info)
        done = self.done_func(self.state)
        reward = self.reward_func(self.state, done, truncated, actions, self.previous_state, info)
        self.previous_state = self.state
        return self.state, reward, done, truncated, info
    
    def run_test_physics(self):
        universal_physics_plotter(env = self,
                                  agent = None,
                                  save_path = f'results/physics_test/',
                                  flight_phase = self.flight_phase,
                                  type = 'physics')
        self.reset()

    def physics_step_test(self, actions, target_altitude):
        terminated = False
        self.state, info = self.physics_step(self.state, actions)
        info['state'] = self.state
        info['actions'] = actions
        altitude = self.state[1]
        propellant_mass = self.state[-2]
        if altitude >= target_altitude:
            terminated = True
        elif propellant_mass <= 0:
            terminated = True
        else:
            terminated = False

        return self.state, terminated, info
    

