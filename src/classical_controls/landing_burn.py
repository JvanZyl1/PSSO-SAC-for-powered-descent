import os
import csv
import math
import numpy as np
import pandas as pd
from pyswarm import pso
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.envs.load_initial_states import load_landing_burn_initial_state
from src.envs.rockets_physics import compile_physics
from src.classical_controls.utils import PD_controller_single_step

#  -------- REFERENCE TRAJECTORY --------
# a(t) = T/m(t) * tau(t) - g_0 + 0.5 * rho(y(t)) * v(t)^2 * C_n_0 * S
# m(t) = m_0 - mdot * int_0^t tau(t) dt
# v(t) = v_0 + int_0^t a(t) dt
# y(t) = y_0 + int_0^t v(t) dt
# Constraints : m > ms, v(t) < v_max(y(t))
# Initial conditions : m(0) = m_0, v(0) = v_0, y(0) = y_0
# Final conditions : v(t_f) = 0, y(t_f) = 0
# tau(t) = (0,1)

class reference_landing_trajectory:
    def __init__(self):
        # Constants
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        Te = float(sizing_results['Thrust engine stage 1'])
        ne = 12
        self.T = Te * ne
        v_ex =  float(sizing_results['Exhaust velocity stage 1'])
        self.mdot_max = self.T/v_ex
        self.g_0 = 9.80665
        self.C_n_0 = 3
        self.S_grid_fin = 2
        self.n_grid_fin = 4
        self.dt = 0.1
        self.max_q = 28000 # [Pa]
        self.m_s = float(sizing_results['Structural mass stage 1 (descent)'])*1000
        
        # Logging
        self.a_vals = []
        self.m_vals = []
        self.y_vals = []
        self.v_vals = []
        self.tau_vals = []
        self.time_vals = []
        self.time = 0.0

        self.load_initial_conditions()
        self.find_dynamic_pressure_limited_velocities()
        self.compute_optimal_trajectory()
        self.post_process_results()
        
    def load_initial_conditions(self):
        state_initial = load_landing_burn_initial_state()
        self.y_0 = state_initial[1]
        v_x_0 = state_initial[2]
        self.v_y_0 = state_initial[3]
        self.v_0 = np.sqrt(v_x_0**2 + self.v_y_0**2)
        self.y_refs = np.linspace(self.y_0, 0, 100)
        # Initial conditions
        self.m = state_initial[8]
        self.y = self.y_0
        self.v = self.v_0

    def find_dynamic_pressure_limited_velocities(self):
        air_densities = np.zeros(len(self.y_refs))
        max_v_s = np.zeros(len(self.y_refs))
        no_thrust_velocities = np.zeros(len(self.y_refs))
        for i, y_ref in enumerate(self.y_refs):
            air_densities[i], atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y_ref)
            max_v_s[i] = np.sqrt(2 * self.max_q /air_densities[i])

        self.v_max_fcn = np.poly1d(np.polyfit(self.y_refs, max_v_s, 4))


    # v(y) = a*y^2 + b*y + c
    # Initial conditions: v(0) = 0, v(y_0) = abs(v_y_0)
    # Using v(0) = 0 => c = 0, so v(y) = a*y^2 + b*y
    def compute_optimal_trajectory(self):
        y_samples = np.linspace(0, self.y_0, 200)
        def v_opt_profile(y, params):
            a, b = params
            return a * y**2 + b * y
            
        def objective(params):
            a, b = params
            return -((a/3) * self.y_0**3 + (b/2) * self.y_0**2) # maximise area under velocity curve
        
        def constraint_initial_velocity(params):
            return v_opt_profile(self.y_0, params) - abs(self.v_y_0)
        
        def constraint_velocity_limit(params):
            return self.v_max_fcn(y_samples) - v_opt_profile(y_samples, params) # vy < vy_lim
        
        result = minimize(
            objective,
            [0, abs(self.v_y_0) / self.y_0], # linear profile from (0,0) to (y_0, abs(v_y_0))
            constraints=[{'type': 'eq', 'fun': constraint_initial_velocity},
                        {'type': 'ineq', 'fun': constraint_velocity_limit}],
            method='trust-constr',
            options={'disp': True}
        )
        a_opt, b_opt = result.x
        print(f"Optimal velocity profile: v(y) = {a_opt:.6e}*y^2 + {b_opt:.6e}*y")
        v_opt = lambda y: a_opt * y**2 + b_opt * y
        return v_opt

    def post_process_results(self):
        v_opt_fn = self.compute_optimal_trajectory()
        # ------ PLOTTING ------
        self.y_vals_plot = np.linspace(min(self.y_refs), max(self.y_refs), 1500)
        v_opt_plot = v_opt_fn(self.y_vals_plot)
        self.a_opt_plot = np.gradient(v_opt_plot, self.y_vals_plot) * v_opt_plot # Chain rule dv/dt = dv/dy * dy/dt = dv/dy * v
        self.a_max_v_s = np.gradient(self.v_max_fcn(self.y_vals_plot), self.y_vals_plot) * self.v_max_fcn(self.y_vals_plot)

        plt.figure(figsize=(20, 10))
        plt.suptitle('Optimal Feasible Landing Trajectory', fontsize=22)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.y_vals_plot/1000, self.v_max_fcn(self.y_vals_plot)/1000, color='blue', linewidth=4, label='Max Velocity Limit')
        ax1.plot(self.y_vals_plot/1000, self.v_max_fcn(self.y_vals_plot)/1000, linestyle = '--', color='grey', label='Polyfit (Limit)')
        ax1.plot(self.y_vals_plot/1000, v_opt_plot/1000, color = 'red', linewidth=3, label='Initial Guess Trajectory')
        ax1.scatter(self.y_0/1000, abs(self.v_0)/1000, color='red', s=100, label='Initial Velocity Magnitude')
        ax1.scatter(self.y_0/1000, abs(self.v_y_0)/1000, color='green', s=100, marker='x', label='Initial Vertical Velocity')
        ax1.scatter(0, 0, color='magenta', s=100, marker='x', label='Target')
        ax1.set_ylabel(r'v [$km/s$]', fontsize=20)
        ax1.set_ylim(0, 2)
        ax1.set_title('Max Velocity vs. Altitude with Optimal 2nd Order Trajectory', fontsize=20)
        ax1.grid(True)
        ax1.legend(fontsize=20)
        ax1.tick_params(labelsize=16)

        ax2 = plt.subplot(gs[1])
        ax2.plot(self.y_vals_plot/1000, self.a_opt_plot/self.g_0, 'r-', linewidth=3, label='Initial Guess Acceleration')
        ax2.plot(self.y_vals_plot/1000, self.a_max_v_s/self.g_0, 'b--', linewidth=3, label='For maximum dynamic pressure')
        ax2.set_xlabel(r'y [$km$]', fontsize=20)
        ax2.set_ylabel(r'a [$g_0$]', fontsize=20)
        ax2.set_title('True Acceleration (dv/dt) vs. Altitude', fontsize=20)
        ax2.grid(True)
        ax2.legend(fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.set_ylim(0, 5)

        plt.savefig('results/classical_controllers/landing_initial_velocity_profile_guess.png')
        plt.close()

        df_reference = pd.DataFrame({
            'altitude': self.y_vals_plot,
            'velocity': v_opt_plot,
            'acceleration': self.a_opt_plot
        })
        df_reference.to_csv('data/reference_trajectory/landing_burn_controls/landing_initial_guess_reference_profile.csv', index=False)

#  -------- CONTROLLERS --------

def throttle_controllers(v_y, v_ref, previous_error, previous_derivative, dt):
    Kp_throttle = 0.6
    Kd_throttle = 0.0
    N_throttle = 10
    error = abs(v_y) - v_ref
    throttle, new_derivative = PD_controller_single_step(Kp=Kp_throttle,
                                                                         Kd=Kd_throttle,
                                                                         N=N_throttle,
                                                                         error=error,
                                                                         previous_error=previous_error,
                                                                         previous_derivative=previous_derivative,
                                                                         dt=dt)
    throttle = np.clip(throttle, 0.0, 1.0)
    return throttle, error, new_derivative

def re_entry_burn_pitch_controller(flight_path_angle_rad,
                                   pitch_angle_rad,
                                   pitch_rate_rad_s,
                                   previous_pitch_rate_error,
                                   previous_pitch_rate_derivative,
                                   dt,
                                   Kp_pitch,
                                   Kp_pitch_rate,
                                   Kd_pitch_rate,
                                   N_pitch_rate):
    M_max = 0.75e9*6/50 # reduced by 6/50 due to 6 engines
    alpha_effective_rad = flight_path_angle_rad - pitch_angle_rad - math.pi
    
    # Outer loop: Effective angle controller
    pitch_rate_command = Kp_pitch * alpha_effective_rad
    
    # Inner loop: Pitch rate controller
    pitch_rate_error = pitch_rate_command - pitch_rate_rad_s
    Mz_command_norm, new_pitch_rate_derivative = PD_controller_single_step(Kp=Kp_pitch_rate,
                                                                          Kd=Kd_pitch_rate,
                                                                          N=N_pitch_rate,
                                                                          error=pitch_rate_error,
                                                                          previous_error=previous_pitch_rate_error,
                                                                          previous_derivative=previous_pitch_rate_derivative,
                                                                          dt=dt)
    
    Mz = np.clip(Mz_command_norm, -1, 1) * M_max
    return Mz, pitch_rate_error, new_pitch_rate_derivative

def gimbal_determination(Mz,
                         non_nominal_throttle,
                         atmospheric_pressure,
                         d_thrust_cg,
                         number_of_engines_gimballed,
                         thrust_per_engine_no_losses,
                         nozzle_exit_pressure,
                         nozzle_exit_area,
                         max_gimbal_angle_rad,
                         nominal_throttle):
    
    throttle = non_nominal_throttle * (1 - nominal_throttle) + nominal_throttle

    thrust_engine_with_losses_full_throttle = (thrust_per_engine_no_losses + (nozzle_exit_pressure - atmospheric_pressure) * nozzle_exit_area)
    thrust_gimballed = thrust_engine_with_losses_full_throttle * number_of_engines_gimballed * throttle

    ratio = -Mz / (thrust_gimballed * d_thrust_cg)
    if ratio > 1:
        gimbal_angle_rad = math.asin(1)
    elif ratio < -1:
        gimbal_angle_rad = math.asin(-1)
    else:
        gimbal_angle_rad = math.asin(ratio)

    gimbal_angle_rad = np.clip(gimbal_angle_rad, -max_gimbal_angle_rad, max_gimbal_angle_rad)

    return gimbal_angle_rad

def augment_actions_re_entry_burn(gimbal_angle_rad, non_nominal_throttle, max_gimbal_angle_rad):
    u0 = float(gimbal_angle_rad) / float(max_gimbal_angle_rad)
    u1 = float(2 * non_nominal_throttle - 1)
    u3, u4 = 0.0, 0.0
    actions = (u0, u1, u3, u4)
    return actions

def action_determination(v_y, v_ref, previous_error_v, previous_derivative_v,
                         theta, theta_dot, gamma,
                         previous_pitch_rate_error, previous_pitch_rate_derivative,
                         atmospheric_pressure, d_thrust_cg,
                         dt, Kp_pitch, Kp_pitch_rate, Kd_pitch_rate, N_pitch_rate, nominal_throttle,
                         number_of_engines_gimballed, thrust_per_engine_no_losses, nozzle_exit_pressure,
                         nozzle_exit_area, max_gimbal_angle_rad):
    non_nominal_throttle, error_v, new_derivative_v = throttle_controllers(v_y, v_ref, previous_error_v, previous_derivative_v, dt)
    Mz, pitch_rate_error, new_pitch_rate_derivative = re_entry_burn_pitch_controller(gamma,
                                                                                     theta,
                                                                                     theta_dot,
                                                                                     previous_pitch_rate_error,
                                                                                     previous_pitch_rate_derivative,
                                                                                     dt,
                                                                                     Kp_pitch,
                                                                                     Kp_pitch_rate,
                                                                                     Kd_pitch_rate,
                                                                                     N_pitch_rate)
    
    gimbal_angle_rad = gimbal_determination(Mz,
                         non_nominal_throttle,
                         atmospheric_pressure,
                         d_thrust_cg,
                         number_of_engines_gimballed,
                         thrust_per_engine_no_losses,
                         nozzle_exit_pressure,
                         nozzle_exit_area,
                         max_gimbal_angle_rad,
                         nominal_throttle)
    
    actions = augment_actions_re_entry_burn(gimbal_angle_rad, non_nominal_throttle, max_gimbal_angle_rad)
    return actions, error_v, new_derivative_v, pitch_rate_error, new_pitch_rate_derivative

class LandingBurn:
    def __init__(self, individual=None):
        self.max_q = 35e3 # [Pa]
        self.dt = 0.1
        df_reference = pd.read_csv('data/reference_trajectory/landing_burn_controls/landing_initial_guess_reference_profile.csv')
        self.v_opt_fcn = scipy.interpolate.interp1d(df_reference['altitude'], df_reference['velocity'], kind='cubic', fill_value='extrapolate')

        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'landing_burn')
        
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]

        # Set default values
        self.N_pitch_rate = 5
        self.post_process_results = True
                
        if individual is not None:
            self.Kp_pitch_rate, self.Kd_pitch_rate, self.Kp_pitch = individual
            self.post_process_results = False
        # else load from file via pandas
        else:
            df_gains = pd.read_csv('data/reference_trajectory/landing_burn_controls/controller_gains.csv')
            self.Kp_pitch_rate = df_gains['Kp_pitch_rate'].values[0]
            self.Kd_pitch_rate = df_gains['Kd_pitch_rate'].values[0]
            self.Kp_pitch = df_gains['Kp_pitch'].values[0]
            # 3.0, 2.0, 0.0
        
        number_of_engines_min = 3
        minimum_engine_throttle = 0.4
        self.nominal_throttle = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])
        self.number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1'])
        self.thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1'])
        self.nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1'])
        self.nozzle_exit_area = float(sizing_results['Nozzle exit area'])
        self.max_gimbal_angle_rad = math.radians(5)
        
        self.max_deflection_angle_deg = 60
        self._update_controller_func()
        self.initial_conditions()
        self.initialise_logging()
    
    def _update_controller_func(self):
        self.controller_func = lambda v_y, v_ref, previous_error_v, previous_derivative_v,\
                         theta, theta_dot, gamma,\
                         previous_pitch_rate_error, previous_pitch_rate_derivative,\
                         atmospheric_pressure, d_thrust_cg : \
                        action_determination(v_y, v_ref, previous_error_v, previous_derivative_v,\
                         theta, theta_dot, gamma,\
                         previous_pitch_rate_error, previous_pitch_rate_derivative,\
                         atmospheric_pressure, d_thrust_cg,
                         self.dt, self.Kp_pitch, 
                         self.Kp_pitch_rate, self.Kd_pitch_rate, self.N_pitch_rate, 
                         self.nominal_throttle, self.number_of_engines_gimballed, 
                         self.thrust_per_engine_no_losses, self.nozzle_exit_pressure,
                         self.nozzle_exit_area, self.max_gimbal_angle_rad)
    
    def update_controller_params(self, Kp_pitch_rate=None, Kd_pitch_rate=None, Kp_pitch=None, Kd_pitch=None):
        if Kp_pitch_rate is not None:
            self.Kp_pitch_rate = Kp_pitch_rate
        if Kd_pitch_rate is not None:
            self.Kd_pitch_rate = Kd_pitch_rate
        if Kp_pitch is not None:
            self.Kp_pitch = Kp_pitch
        if Kd_pitch is not None:
            self.Kd_pitch = Kd_pitch
            
        self._update_controller_func()

    def reset(self):
        self.initial_conditions()
        self.initialise_logging()

    def initial_conditions(self):
        self.state = load_landing_burn_initial_state()
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.previous_v_error = self.v_opt_fcn(self.y) - self.vy
        self.previous_v_derivative = 0.0
        self.alpha_effective = self.gamma - self.theta - math.pi
        
        self.previous_pitch_rate_error = self.Kp_pitch_rate * self.alpha_effective - self.theta_dot
        self.previous_pitch_rate_derivative = 0.0

        self.gimbal_angle_deg_prev = 0.0
        self.delta_command_left_rad_prev = 0.0
        self.delta_command_right_rad_prev = 0.0
        self.wind_generator = None

        self.dynamic_pressure= 0.0

        state, info = self.simulation_step_lambda(self.state, (0.0, 0.0, 0.0, 0.0),
                                                self.gimbal_angle_deg_prev, self.delta_command_left_rad_prev,
                                                self.delta_command_right_rad_prev, self.wind_generator)
        self.atmospheric_pressure = info['atmospheric_pressure']
        self.d_thrust_cg = info['d_thrust_cg']

    
    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.theta_vals = []
        self.theta_dot_vals = []
        self.gamma_vals = []
        self.alpha_vals = []
        self.mass_vals = []
        self.m_prop_vals = []
        self.time_vals = []
        self.Mz_acs_vals = []

        self.u0_vals = []
        self.u1_vals = []
        self.u2_vals = []
        self.u3_vals = []

        self.vy_ref_vals = []
        self.delta_L_vals = []
        self.delta_R_vals = []
        self.throttle_vals = []

        self.alpha_effective_vals = []

        self.dynamic_pressure_vals = []
        self.F_n_L_vals= []
        self.F_n_R_vals = []
        self.F_a_L_vals = []
        self.F_a_R_vals = []

        self.aero_moments = []
        self.control_moments = []
        self.total_moments = []

        self.gimbal_angles_deg = []
        
        self.pitch_rate_command_vals = []
        self.pitch_rate_error_vals = []

        self.tau_vals = []

    def closed_loop_step(self):
        v_ref = self.v_opt_fcn(self.y)
        actions, self.previous_v_error, self.previous_v_derivative,\
              self.previous_pitch_rate_error, self.previous_pitch_rate_derivative = self.controller_func(
                  self.vy, v_ref, self.previous_v_error, self.previous_v_derivative,\
                  self.theta, self.theta_dot, self.gamma,\
                  self.previous_pitch_rate_error, self.previous_pitch_rate_derivative,\
                  self.atmospheric_pressure, self.d_thrust_cg)

        self.state, info = self.simulation_step_lambda(self.state, actions,
                                                self.gimbal_angle_deg_prev, self.delta_command_left_rad_prev,
                                                self.delta_command_right_rad_prev, self.wind_generator)
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.alpha_effective = self.gamma - self.theta - math.pi

        self.gimbal_angle_deg_prev = info['action_info']['gimbal_angle_deg']
        self.delta_command_left_rad_prev = info['action_info']['delta_command_left_rad']
        self.delta_command_right_rad_prev = info['action_info']['delta_command_right_rad']
        self.atmospheric_pressure = info['atmospheric_pressure']
        self.d_thrust_cg = info['d_thrust_cg']

        self.x_vals.append(self.x)
        self.y_vals.append(self.y)
        self.vx_vals.append(self.vx)
        self.vy_vals.append(self.vy)
        self.theta_vals.append(self.theta)
        self.theta_dot_vals.append(self.theta_dot)
        self.gamma_vals.append(self.gamma)
        self.alpha_vals.append(self.alpha)
        self.mass_vals.append(self.mass)
        self.m_prop_vals.append(self.mass_propellant)
        self.time_vals.append(self.time)
        self.u0_vals.append(actions[0])
        self.u1_vals.append(actions[1])
        self.u2_vals.append(actions[2])
        self.u3_vals.append(actions[3])

        self.vy_ref_vals.append(v_ref)
        self.delta_L_vals.append(self.delta_command_left_rad_prev)
        self.delta_R_vals.append(self.delta_command_right_rad_prev)
        self.throttle_vals.append(info['action_info']['throttle'])

        self.alpha_effective_vals.append(self.alpha_effective)
        self.Mz_acs_vals.append(info['action_info']['acs_info']['Mz'])
        self.F_n_L_vals.append(info['action_info']['acs_info']['F_n_L'])
        self.F_n_R_vals.append(info['action_info']['acs_info']['F_n_R'])
        self.F_a_L_vals.append(info['action_info']['acs_info']['F_a_L'])
        self.F_a_R_vals.append(info['action_info']['acs_info']['F_a_R'])
        self.dynamic_pressure_vals.append(info['dynamic_pressure'])
        self.dynamic_pressure = info['dynamic_pressure']

        self.aero_moments.append(info['moment_dict']['aero_moment_z'])
        self.control_moments.append(info['moment_dict']['control_moment_z'])
        self.total_moments.append(info['moment_dict']['moments_z'])

        self.gimbal_angles_deg.append(self.gimbal_angle_deg_prev)

        self.pitch_rate_error_vals.append(self.previous_pitch_rate_error)
        self.tau_vals.append(info['action_info']['throttle'])

    def performance_metrics(self):
        alpha_effective_error = np.abs(np.array(self.alpha_effective_vals))
        pitch_rate_error = np.abs(np.array(self.pitch_rate_error_vals))
        reward = (-np.sum(alpha_effective_error) - np.sum(pitch_rate_error))/len(self.time_vals)
        return reward
    
    def run_closed_loop(self):
        success = True
        simulation_steps = 0
        max_steps = 50000  # Prevent infinite loops
        
        while self.mass_propellant > 0 and self.y > 1 and self.dynamic_pressure < self.max_q and simulation_steps < max_steps:
            self.closed_loop_step()
            simulation_steps += 1
            
        if self.mass_propellant < 0:
            print('Landing burn failed, out of propellant, stopped at altitude: ', self.y)
            success = False
        elif self.y < 1:
            print('Landing burn failed, out of altitude, stopped at altitude: ', self.y)
            success = False
        elif self.dynamic_pressure > self.max_q:
            print('Landing burn failed, out of dynamic pressure, stopped at altitude: ', self.y)
            success = False
        elif simulation_steps >= max_steps:
            print('Landing burn failed, max steps reached, stopped at altitude: ', self.y)
            success = False
        else:
            print('Landing burn successful')
            print(f'Mass propellant {self.mass_propellant}')
            print(f'Dynamic pressure {self.dynamic_pressure}')
            print(f'Altitude {self.y}')
            print(f'Steps {simulation_steps}')
            success = True
            
        if self.post_process_results:
            self.save_results()
            self.plot_results()
        
        return success

    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/landing_burn_controls/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_landing_burn_control.csv')

        data = {
            't[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'mass[kg]': self.mass_vals,
            'tau[-]': self.tau_vals
        }
        pd.DataFrame(data).to_csv(full_trajectory_path, index=False)
        state_action_data = {
            'time[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'theta[rad]': self.theta_vals,
            'theta_dot[rad/s]': self.theta_dot_vals,
            'alpha[rad]': self.alpha_vals,
            'gamma[rad]': self.gamma_vals,
            'mass[kg]': self.mass_vals,
            'masspropellant[kg]': self.m_prop_vals,
            'u0': self.u0_vals,
            'u1': self.u1_vals,
            'u2': self.u2_vals,
            'u3': self.u3_vals
        }
        state_action_path = os.path.join(save_folder, 'state_action_landing_burn_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)
    
    def plot_results(self):
        V_abs = np.sqrt(np.array(self.vx_vals)**2 + np.array(self.vy_vals)**2)
        
        plt.figure(figsize=(20,15))
        plt.suptitle('Landing Burn (Initial Guess) Control', fontsize=24)
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], hspace=0.3, wspace=0.3)

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(np.array(self.x_vals)/1e3, np.array(self.y_vals)/1e3, linewidth=4, color = 'blue')
        ax1.set_xlabel(r'x [$km$]', fontsize=20)
        ax1.set_ylabel(r'y [$km$]', fontsize=20)
        ax1.set_title('Trajectory', fontsize=22)
        ax1.grid(True)
        ax1.tick_params(labelsize=16)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, np.array(self.m_prop_vals)/1e3, linewidth=4, color = 'blue')
        ax2.set_xlabel(r'Time [$s$]', fontsize=20)
        ax2.set_ylabel(r'Mass [$t$]', fontsize=20)
        ax2.set_title('Propellant mass', fontsize=22)
        ax2.grid(True)
        ax2.tick_params(labelsize=16)

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(self.time_vals, V_abs, linewidth=4, color = 'blue', label='Actual')
        ax3.plot(self.time_vals, np.array(self.vy_ref_vals), linewidth=2, color = 'red', linestyle='--', label='Reference')
        ax3.plot(self.time_vals, np.abs(np.array(self.vx_vals)), linewidth=2, color = 'green', linestyle='--', label='Horizontal velocity')
        ax3.plot(self.time_vals, np.abs(np.array(self.vy_vals)), linewidth=2, color = 'purple', linestyle='--', label='Vertical velocity')
        ax3.set_xlabel(r'Time [$s$]', fontsize=20)
        ax3.set_ylabel(r'Velocity (Absolute) [$m/s$]', fontsize=20)
        ax3.set_title('Velocity', fontsize=22)
        ax3.grid(True)
        ax3.tick_params(labelsize=16)
        ax3.legend(fontsize=20)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, self.throttle_vals, linewidth=4, color = 'blue')
        ax4.set_xlabel(r'Time [$s$]', fontsize=20)
        ax4.set_ylabel(r'Throttle', fontsize=20)
        ax4.set_title('Throttle', fontsize=22)
        ax4.grid(True)
        ax4.tick_params(labelsize=16)

        ax5 = plt.subplot(gs[2, 0])
        ax5.plot(self.time_vals, np.rad2deg(np.array(self.alpha_effective_vals)), linewidth=4, color = 'blue')
        ax5.set_xlabel(r'Time [$s$]', fontsize=20)
        ax5.set_ylabel(r'Effective angle of attack [$^{\circ}$]', fontsize=20)
        ax5.set_title('Effective angle of attack', fontsize=22)
        ax5.grid(True)
        ax5.tick_params(labelsize=16)

        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(self.time_vals, self.gimbal_angles_deg, linewidth=4, color = 'blue')
        ax6.set_xlabel(r'Time [$s$]', fontsize=20)
        ax6.set_ylabel(r'Gimbal angle [$^{\circ}$]', fontsize=20)
        ax6.set_title('Gimbal angle', fontsize=22)
        ax6.grid(True)
        ax6.tick_params(labelsize=16)

        plt.savefig('results/classical_controllers/landing_burn_control_initial_guess.png')
        plt.close()

        plt.figure(figsize=(20,15))
        plt.suptitle('Landing Burn (Initial Guess) Control', fontsize=24)
        # Deflection angles, Effective angle of attack, Mz_acs
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.2)

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.time_vals, self.gimbal_angles_deg, linewidth=4, color = 'blue')
        ax1.set_xlabel(r'Time [$s$]', fontsize=20)
        ax1.set_ylabel(r'Gimbal angle [$^{\circ}$]', fontsize=20)
        ax1.set_title('Gimbal angle', fontsize=22)
        ax1.grid(True)
        ax1.tick_params(labelsize=16)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, np.rad2deg(np.array(self.alpha_effective_vals)), linewidth=4, color = 'blue')
        ax2.set_xlabel(r'Time [$s$]', fontsize=20)
        ax2.set_ylabel(r'Effective angle of attack [$^{\circ}$]', fontsize=20)
        ax2.set_title('Effective angle of attack', fontsize=22)
        ax2.grid(True)
        ax2.tick_params(labelsize=16)

        ax3 = plt.subplot(gs[1, 0])
        if max(np.array(self.total_moments)) > 1e6:
            ax3.plot(self.time_vals, np.array(self.Mz_acs_vals)/1e6, linewidth=4, color = 'green', label='ACS')
            ax3.plot(self.time_vals, np.array(self.aero_moments)/1e6, linewidth=3, color = 'red', linestyle='--', label='Aero')
            ax3.plot(self.time_vals, np.array(self.control_moments)/1e6, linewidth=3, color = 'blue', linestyle='--', label='Control')
            ax3.plot(self.time_vals, np.array(self.total_moments)/1e6, linewidth=2, color = 'black', linestyle='--', label='Total')
            ax3.set_ylabel(r'Moment [$MNm$]', fontsize=20)
        elif max(np.array(self.total_moments)) > 1e3:
            ax3.plot(self.time_vals, np.array(self.Mz_acs_vals)/1e3, linewidth=4, color = 'green', label='ACS')
            ax3.plot(self.time_vals, np.array(self.aero_moments)/1e3, linewidth=3, color = 'red', linestyle='--', label='Aero')
            ax3.plot(self.time_vals, np.array(self.control_moments)/1e3, linewidth=3, color = 'blue', linestyle='--', label='Control')
            ax3.plot(self.time_vals, np.array(self.total_moments)/1e3, linewidth=2, color = 'black', linestyle='--', label='Total')
            ax3.set_ylabel(r'Moment [$kNm$]', fontsize=20)
        else:
            ax3.plot(self.time_vals, np.array(self.Mz_acs_vals), linewidth=4, color = 'green', label='ACS')
            ax3.plot(self.time_vals, np.array(self.aero_moments), linewidth=2, color = 'red', linestyle='--', label='Aero')
            ax3.plot(self.time_vals, np.array(self.control_moments), linewidth=2, color = 'blue', linestyle='--', label='Control')
            ax3.plot(self.time_vals, np.array(self.total_moments), linewidth=2, color = 'black', linestyle='--', label='Total')
            ax3.set_ylabel(r'Moment [$Nm$]', fontsize=20)
        ax3.set_xlabel(r'Time [$s$]', fontsize=20)
        ax3.set_title('Moment', fontsize=22)
        ax3.grid(True)
        ax3.tick_params(labelsize=16)
        ax3.legend(fontsize=20)

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, np.array(self.dynamic_pressure_vals)/1000, linewidth=4, color = 'red', label='Dynamic pressure')
        ax4.set_xlabel(r'Time [$s$]', fontsize=20)
        ax4.set_ylabel(r'Dynamic pressure [$kPa$]', fontsize=20)
        ax4.set_title('Dynamic pressure', fontsize=22)
        ax4.grid(True)
        ax4.tick_params(labelsize=16)

        ax5 = plt.subplot(gs[2, 0])
        ax5.plot(self.time_vals, np.rad2deg(np.array(self.gamma_vals)), linewidth=4, color = 'blue', label='Flight path')
        ax5.plot(self.time_vals, np.rad2deg(np.array(self.theta_vals) + math.pi), linewidth=4, color = 'red', label='Pitch')
        ax5.set_xlabel(r'Time [$s$]', fontsize=20)
        ax5.set_ylabel(r'Angle [$^{\circ}$]', fontsize=20)
        ax5.set_title('Angles', fontsize=22)
        ax5.grid(True)
        ax5.tick_params(labelsize=16)
        ax5.legend(fontsize=20)
        
        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(self.time_vals, np.array(self.F_a_L_vals)/1e3, linewidth=4, color = 'magenta', label='Left')
        ax6.plot(self.time_vals, np.array(self.F_a_R_vals)/1e3, linewidth=4, color = 'cyan', label='Right')
        ax6.set_xlabel(r'Time [$s$]', fontsize=20)
        ax6.set_ylabel(r'Axial force [$kN$]', fontsize=20)
        ax6.set_title('Axial force', fontsize=22)
        ax6.grid(True)
        ax6.tick_params(labelsize=16)
        plt.savefig('results/classical_controllers/landing_burn_control_initial_guess_angular_controls.png')
        plt.close()
        
        # Add a plot for pitch rate error
        plt.figure(figsize=(10,6))
        plt.plot(self.time_vals, self.pitch_rate_error_vals, linewidth=4, color='purple')
        plt.xlabel(r'Time [$s$]', fontsize=20)
        plt.ylabel(r'Pitch rate error [$rad/s$]', fontsize=20)
        plt.title('Pitch Rate Error', fontsize=22)
        plt.grid(True)
        plt.tick_params(labelsize=16)
        plt.savefig('results/classical_controllers/landing_burn_control_initial_guess_pitch_rate.png')
        plt.close()

        # Positions with time
        plt.figure(figsize=(10,6))
        plt.plot(self.time_vals, np.array(self.y_vals)/1e3, linewidth=4, color='blue', label='Altitude')
        plt.xlabel(r'Time [$s$]', fontsize=20)
        plt.ylabel(r'Altitude [$km$]', fontsize=20)
        plt.title('Altitude', fontsize=22)
        plt.grid(True)
        plt.tick_params(labelsize=16)
        plt.savefig('results/classical_controllers/landing_burn_control_initial_guess_altitude.png')
        plt.close()

# --------- OPTIMISATION OF CONTROLLERS --------


def objective_func(individual):
    landing_burn = LandingBurn(individual)
    landing_burn.run_closed_loop()
    obj = -landing_burn.performance_metrics()
    return obj

def save_gains(xopt):
    xopt = np.array(xopt)
    print(f"Best parameters found: {xopt}")
    gains = pd.DataFrame({
        'Kp_pitch_rate': [xopt[0]],
        'Kd_pitch_rate': [xopt[1]],
        'Kp_pitch': [xopt[2]]
    })
    gains.to_csv('data/reference_trajectory/landing_burn_controls/controller_gains.csv', index=False)

def tune_landing_burn():
    # Parameter bounds: [Kp_pitch_rate, Kd_pitch_rate, Kp_pitch]
    lb = [1.0, 0.0, 0.0]  # Lower bounds
    ub = [3.0, 2.0, 1.0]  # Upper bounds
    
    print("Starting PSO optimization for controller tuning...")
    xopt, fopt = pso(
        objective_func,
        lb, 
        ub,
        swarmsize=20,       # Number of particles
        omega=0.5,          # Particle velocity scaling factor
        phip=0.5,           # Scaling factor for particle's best known position
        phig=0.5,           # Scaling factor for swarm's best known position
        maxiter=5,         # Maximum iterations
        minstep=1e-6,       # Minimum step size before search termination
        minfunc=1e-6,       # Minimum change in obj value before termination
        debug=True,         # Print progress statements
    )
    
    save_gains(xopt)
    best_landing_burn = LandingBurn(xopt)
    best_landing_burn.post_process_results = True
    best_landing_burn.run_closed_loop()
    
    return xopt, fopt


def run_landing_burn(tune_bool = False):
    # Generate reference trajectory
    ref = reference_landing_trajectory()
    if tune_bool:
        tune_landing_burn()
        
    # Run landing burn
    landing_burn = LandingBurn()
    landing_burn.run_closed_loop()
    

    
