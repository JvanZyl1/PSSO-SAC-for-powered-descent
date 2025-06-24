import os
import csv
import math
import numpy as np
import pandas as pd
import dill
from pyswarm import pso
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.envs.load_initial_states import load_landing_burn_initial_state
from src.envs.rockets_physics import compile_physics
from src.classical_controls.utils import PD_controller_single_step
from src.envs.wind.full_wind_model import WindModel
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
        self.max_q = 50000 # [Pa]
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

        for i, y_ref in enumerate(self.y_refs):
            air_densities[i], _, _ = endo_atmospheric_model(y_ref)
            max_v_s[i] = np.sqrt(2.0 * self.max_q / air_densities[i])

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
            return -((a/3) * self.y_0**3 + (b/2) * self.y_0**2) # maximise area under curve
        
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

        with open('data/reference_trajectory/landing_burn_controls/landing_initial_velocity_profile_guess.pkl', 'wb') as f:
            dill.dump(v_opt, f)

        with open('data/reference_trajectory/landing_burn_controls/landing_initial_velocity_profile_guess.txt', 'w') as f:
            f.write(f"v(y) = {a_opt:.6e}*y^2 + {b_opt:.6e}*y\n")

        return v_opt

    def post_process_results(self):
        v_opt_fn = self.compute_optimal_trajectory()
        # ------ PLOTTING ------
        self.y_vals_plot = np.linspace(min(self.y_refs), max(self.y_refs), 500)
        v_opt_plot = v_opt_fn(self.y_vals_plot)
        self.a_opt_plot = np.gradient(v_opt_plot, self.y_vals_plot) * v_opt_plot # Chain rule dv/dt = dv/dy * dy/dt = dv/dy * v
        self.a_max_v_s = np.gradient(self.v_max_fcn(self.y_vals_plot), self.y_vals_plot) * self.v_max_fcn(self.y_vals_plot)

        fig = plt.figure(figsize=(20, 10))
        plt.suptitle('Optimal Feasible Landing Trajectory', fontsize=22)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.y_vals_plot/1000, self.v_max_fcn(self.y_vals_plot)/1000, color='blue', linewidth=4, label='Max Velocity Limit')
        ax1.plot(self.y_vals_plot/1000, self.v_max_fcn(self.y_vals_plot)/1000, linestyle = '--', color='grey', label='Polyfit (Limit)')
        ax1.plot(self.y_vals_plot/1000, v_opt_plot/1000, color = 'red', linewidth=3, label='Initial Guess Trajectory')
        ax1.scatter(self.y_0/1000, abs(self.v_0)/1000, color='red', s=100, label='Initial Velocity Magnitude')
        ax1.scatter(self.y_0/1000, abs(self.v_y_0)/1000, color='green', s=100, marker='x', label='Initial Vertical Velocity')
        ax1.scatter(0, 0, color='magenta', s=100, marker='x', label='Target')
        ax1.set_ylabel(r'v [$km/s$]', fontsize=20)
        ax1.set_title('Max Velocity vs. Altitude with Optimal 2nd Order Trajectory', fontsize=20)
        ax1.grid(True)
        ax1.legend(fontsize=20)
        ax1.tick_params(labelsize=16)

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(self.y_vals_plot/1000, self.a_opt_plot/self.g_0, 'r-', linewidth=3, label='Initial Guess Acceleration')
        ax2.plot(self.y_vals_plot/1000, self.a_max_v_s/self.g_0, 'b--', linewidth=3, label='For maximum dynamic pressure')
        ax2.set_xlabel(r'y [$km$]', fontsize=20)
        ax2.set_ylabel(r'a [$g_0$]', fontsize=20)
        ax2.grid(True)
        ax2.legend(fontsize=20)
        ax2.tick_params(labelsize=16)
        ax2.set_ylim(0, 8)
        
        plt.savefig('results/classical_controllers/landing_initial_velocity_profile_guess.png', dpi=300)

        df_reference = pd.DataFrame({
            'altitude': self.y_vals_plot,
            'velocity': v_opt_plot,
            'acceleration': self.a_opt_plot,
        })
        df_reference.to_csv('data/reference_trajectory/landing_burn_controls/landing_initial_guess_reference_profile.csv', index=False)

class LandingBurn:
    def __init__(self, test_case = 'control'):
        self.test_case = test_case
        self.std_max_stochastic = 0.2
        self.std_max_stochastic_v_ref = 0.2
        assert self.test_case in ['control', 'stochastic', 'stochastic_v_ref', 'wind'], 'test_case must be either control or stochastic'
        self.max_q = 65e3 # [Pa]
        self.dt = 0.1

        with open('data/reference_trajectory/landing_burn_controls/landing_initial_velocity_profile_guess.pkl', 'rb') as f:
            self.v_opt_fcn = dill.load(f)

        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'landing_burn_pure_throttle')
        
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]

        if self.test_case == 'wind':
            self.wind_model = WindModel(self.dt, stochastic_wind = True, given_percentile = 50)
        else:
            self.wind_model = None
        
        number_of_engines_min = 0
        minimum_engine_throttle = 0.4
        self.nominal_throttle = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])
        self.number_of_engines_gimballed = int(sizing_results['Number of engines gimballed stage 1'])
        self.thrust_per_engine_no_losses = float(sizing_results['Thrust engine stage 1'])
        self.nozzle_exit_pressure = float(sizing_results['Nozzle exit pressure stage 1'])
        self.nozzle_exit_area = float(sizing_results['Nozzle exit area'])

        self.Kp_throttle = -0.10
        self.Kd_throttle = 0.0
        self.N_throttle = 10.0
        
        self.initial_conditions()
        self.initialise_logging()

    def reset(self):
        self.initial_conditions()
        self.initialise_logging()

    def initial_conditions(self):
        self.state = load_landing_burn_initial_state()
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.previous_velocity_error = self.v_opt_fcn(self.y) - self.vy
        self.previous_velocity_error_derivative = 0.0
        self.dynamic_pressure= 0.0
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        self.speed0 = self.speed
        self.alpha_effective = self.gamma - self.theta - math.pi
    
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
        self.u1_vref_vals = []
        self.vy_ref_vals = []
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
        
        self.CL_vals = []
        self.CD_vals = []
        self.drag_vals = []
        self.lift_vals = []

        self.tau_vals = []

        self.mach_number_vals = []

    def throttle_control(self):
        if self.test_case == 'stochastic_v_ref':
            speed_ref = self.v_opt_fcn(self.y)
            u1 = speed_ref/self.speed0 * 2 - 1
            u1_aug = u1 * (1 + np.random.uniform(-1.0, 1.0) * self.std_max_stochastic_v_ref) # how well responds to SAC
            self.speed_ref = (u1_aug + 1)/2 * self.speed0
        else:
            self.speed_ref = self.v_opt_fcn(self.y)
        error = self.speed_ref - self.speed
        non_nominal_throttle, self.previous_velocity_error_derivative = PD_controller_single_step(Kp=self.Kp_throttle,
                                                             Kd=self.Kd_throttle,
                                                             N=self.N_throttle,
                                                             error=error,
                                                             previous_error=self.previous_velocity_error,
                                                             previous_derivative=self.previous_velocity_error_derivative,
                                                             dt=self.dt)
        non_nominal_throttle = np.clip(non_nominal_throttle, 0.0, 1.0)
        throttle = non_nominal_throttle * (1 - self.nominal_throttle) + self.nominal_throttle
        self.previous_velocity_error = error

        # non nominal throttle (0.0, 1.0) -> u0 (-1.0, 1.0)
        u0 = 2 * (non_nominal_throttle - 0.5)
        if self.test_case == 'stochastic':
            u0 = u0 * (1 + np.random.uniform(-1.0, 1.0) * self.std_max_stochastic) # how well responds to SAC
        return throttle, u0

    def closed_loop_step(self):
        v_ref = self.v_opt_fcn(self.y)
        u1 = v_ref/self.speed0 * 2 - 1
        self.u1_vref_vals.append(u1)
        throttle, u0 = self.throttle_control()

        self.state, info = self.simulation_step_lambda(self.state, np.array([[u0]]), self.wind_model)
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.speed = np.sqrt(self.vx**2 + self.vy**2)
        self.alpha_effective = self.gamma - self.theta - math.pi
        
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
        self.u0_vals.append(u0)

        self.vy_ref_vals.append(v_ref)
        self.throttle_vals.append(info['action_info']['throttle'])

        self.alpha_effective_vals.append(self.alpha_effective)
        self.Mz_acs_vals.append(info['action_info']['acs_info']['Mz'])
        self.F_n_L_vals.append(info['action_info']['acs_info']['F_n_L'])
        self.F_n_R_vals.append(info['action_info']['acs_info']['F_n_R'])
        self.F_a_L_vals.append(info['action_info']['acs_info']['F_a_L'])
        self.F_a_R_vals.append(info['action_info']['acs_info']['F_a_R'])
        self.dynamic_pressure_vals.append(info['dynamic_pressure'])
        self.dynamic_pressure = info['dynamic_pressure']
        self.mach_number_vals.append(info['mach_number'])
        self.aero_moments.append(info['moment_dict']['aero_moment_z'])
        self.control_moments.append(info['moment_dict']['control_moment_z'])
        self.total_moments.append(info['moment_dict']['moments_z'])
        self.tau_vals.append(info['action_info']['throttle'])
        self.CL_vals.append(info['CL'])
        self.CD_vals.append(info['CD'])
        self.drag_vals.append(info['drag'])
        self.lift_vals.append(info['lift'])
    
    def run_closed_loop(self):
        simulation_steps = 0
        max_steps = 50000  # Prevent infinite loops
        
        while self.mass_propellant > 0 and self.y > 1 and self.dynamic_pressure < self.max_q and simulation_steps < max_steps and self.vy < 0\
            and self.alpha_effective < math.degrees(5):
            self.closed_loop_step()
            simulation_steps += 1

        speed_vals = np.sqrt(np.array(self.vx_vals)**2 + np.array(self.vy_vals)**2)
        acceleration_vals = np.gradient(speed_vals, self.dt)
        max_acceleration = max(abs(acceleration_vals))
        print(f'Max acceleration: {abs(max_acceleration/9.81)} g')
        plt.figure(figsize=(20,10))
        plt.plot(self.time_vals, abs(acceleration_vals/9.81), color = 'blue', linewidth=2)
        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel('Acceleration [g]', fontsize=20)
        if self.test_case == 'control':
            plt.title('Acceleration', fontsize=22)
        elif self.test_case == 'stochastic':
            plt.title(f'Acceleration, uniform noise with max rand {self.std_max_stochastic}', fontsize=22)
        elif self.test_case == 'stochastic_v_ref':
            plt.title(f'Acceleration, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=22)
        elif self.test_case == 'wind':
            plt.title(f'Acceleration, with wind!', fontsize=22)
        plt.axhline(y=6.0, color='red', linestyle='--', linewidth=2, label='Maximum')
        plt.grid(True)
        plt.tick_params(labelsize=16)
        if self.test_case == 'control':
            plt.savefig('results/classical_controllers/landing_burn_control_pure_throttle_acceleration.png')
        elif self.test_case == 'stochastic':
            plt.savefig(f'results/classical_controllers/landing_burn_stochastic/landing_burn_control_pure_throttle_acceleration_max_rand_{self.std_max_stochastic}.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig(f'results/classical_controllers/landing_burn_stochastic_v_ref/landing_burn_control_pure_throttle_acceleration_max_rand_{self.std_max_stochastic_v_ref}.png')
        elif self.test_case == 'wind':
            plt.savefig(f'results/classical_controllers/landing_burn_pure_throttle_wind/landing_burn_control_pure_throttle_acceleration_wind.png')
        plt.close()
        if self.mass_propellant < 0:
            print('Landing burn failed, out of propellant, stopped at altitude: ', self.y)
        elif self.y < 1:
            print('Landing burn failed, out of altitude, stopped at altitude: ', self.y)
        elif self.dynamic_pressure > self.max_q:
            print('Landing burn failed, out of dynamic pressure, stopped at altitude: ', self.y)
        elif simulation_steps >= max_steps:
            print('Landing burn failed, max steps reached, stopped at altitude: ', self.y)
        elif self.vy > 0:
            print('Landing burn failed, vertical velocity is positive, stopped at altitude: ', self.y)
        else:
            print('Landing burn successful')
            print(f'Mass propellant {self.mass_propellant}')
            print(f'Dynamic pressure {self.dynamic_pressure}')
            print(f'Altitude {self.y}')
            print(f'Steps {simulation_steps}')
        
        if self.test_case == 'control':
            self.save_results()
        
        self.plot_results()

    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/landing_burn_controls_pure_throttle/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_landing_burn_pure_throttle_control.csv')

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
            'u1_vref': self.u1_vref_vals
        }
        state_action_path = os.path.join(save_folder, 'state_action_landing_burn_pure_throttle_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)
    
    def plot_results(self):
        V_abs = np.sqrt(np.array(self.vx_vals)**2 + np.array(self.vy_vals)**2)
        
        plt.figure(figsize=(20,15))
        if self.test_case == 'control':
            plt.suptitle('Landing Burn (Initial Guess) Pure Throttle Control', fontsize=24)
        elif self.test_case == 'stochastic':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Stochastic, uniform noise with max rand {self.std_max_stochastic}', fontsize=24)
        elif self.test_case == 'stochastic_v_ref':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Stochastic v_ref, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=24)
        elif self.test_case == 'wind':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Wind', fontsize=24)
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
        ax3.set_ylim(0, V_abs[0])
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
        ax5.set_ylim(-5, 5)
        ax6 = plt.subplot(gs[2, 1])
        ax6.plot(self.time_vals, np.array(self.dynamic_pressure_vals)/1000, linewidth=4, color = 'blue')
        ax6.set_xlabel(r'Time [$s$]', fontsize=20)
        ax6.set_ylabel(r'Dynamic pressure [$kPa$]', fontsize=20)
        ax6.set_title('Dynamic pressure', fontsize=22)
        ax6.grid(True)
        ax6.tick_params(labelsize=16)

        if self.test_case == 'control':
            plt.savefig('results/classical_controllers/landing_burn_control_pure_throttle.png')
        elif self.test_case == 'stochastic':
            plt.savefig('results/classical_controllers/landing_burn_stochastic/landing_burn_control_pure_throttle.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig('results/classical_controllers/landing_burn_stochastic_v_ref/landing_burn_control_pure_throttle.png')
        elif self.test_case == 'wind':
            plt.savefig('results/classical_controllers/landing_burn_pure_throttle_wind/landing_burn_control_pure_throttle.png')
        plt.close()

        plt.figure(figsize=(20,15))
        if self.test_case == 'control':
            plt.suptitle('Landing Burn (Initial Guess) Pure Throttle Angular Control', fontsize=24)
        elif self.test_case == 'stochastic_v_ref':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Angular Control, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=24)
        elif self.test_case == 'stochastic':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Angular Control, uniform noise with max rand {self.std_max_stochastic}', fontsize=24)
        elif self.test_case == 'wind':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Angular Control, with wind!', fontsize=24)
        # Deflection angles, Effective angle of attack, Mz_acs
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.2)

        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.time_vals, self.mach_number_vals, linewidth=4, color = 'blue')
        ax1.set_xlabel(r'Time [$s$]', fontsize=20)
        ax1.set_ylabel(r'Mach number', fontsize=20)
        ax1.set_title('Mach number', fontsize=22)
        ax1.grid(True)
        ax1.tick_params(labelsize=16)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, np.rad2deg(np.array(self.alpha_effective_vals)), linewidth=4, color = 'blue')
        ax2.set_xlabel(r'Time [$s$]', fontsize=20)
        ax2.set_ylabel(r'Effective angle of attack [$^{\circ}$]', fontsize=20)
        ax2.set_title('Effective angle of attack', fontsize=22)
        ax2.grid(True)
        ax2.tick_params(labelsize=16)
        ax2.set_ylim(-10, 10)
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

        if self.test_case == 'control':
            plt.savefig('results/classical_controllers/landing_burn_control_pure_throttle_angular_controls.png')
        elif self.test_case == 'stochastic':
            plt.savefig('results/classical_controllers/landing_burn_stochastic/landing_burn_control_pure_throttle_angular_controls.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig('results/classical_controllers/landing_burn_stochastic_v_ref/landing_burn_control_pure_throttle_angular_controls.png')
        elif self.test_case == 'wind':
            plt.savefig('results/classical_controllers/landing_burn_pure_throttle_wind/landing_burn_control_pure_throttle_angular_controls.png')
        plt.close()

        # if stochastic or stochastic_v_ref, plot the reference velocity
        if self.test_case == 'stochastic' or self.test_case == 'stochastic_v_ref':
            plt.figure(figsize=(20,15))
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Reference Velocity, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=24)
            plt.plot(self.time_vals, self.vy_ref_vals, linewidth=4, color = 'blue')
            plt.xlabel(r'Time [$s$]', fontsize=20)
            plt.ylabel(r'Velocity [$m/s$]', fontsize=20)
            plt.title('Reference velocity', fontsize=22)
            plt.grid(True)
            plt.tick_params(labelsize=16)
            if self.test_case == 'stochastic':
                plt.savefig(f'results/classical_controllers/landing_burn_stochastic/landing_burn_control_pure_throttle_reference_velocity_max_rand_{self.std_max_stochastic}.png')
            elif self.test_case == 'stochastic_v_ref':
                plt.savefig(f'results/classical_controllers/landing_burn_stochastic_v_ref/landing_burn_control_pure_throttle_reference_velocity_max_rand_{self.std_max_stochastic_v_ref}.png')
            plt.close()

        # Plot the acceleration over 1 second in gs through interpolation
        speed_vals = np.sqrt(np.array(self.vx_vals)**2 + np.array(self.vy_vals)**2)
        # as dt = 0.1, every 10th point is 1 second
        speed_vals = speed_vals[::10]
        time_vals = self.time_vals[::10]
        acceleration_vals = np.gradient(speed_vals, time_vals)
        plt.figure(figsize=(20, 10))
        plt.plot(time_vals, abs(acceleration_vals/9.81), color='blue', linewidth=2)
        plt.xlabel('Time [s]', fontsize=20)
        plt.ylabel('Acceleration [g]', fontsize=20)
        plt.title('Acceleration (0.5-second sliding window)', fontsize=22)
        plt.axhline(y=6.0, color='red', linestyle='--', linewidth=2, label='Maximum Limit')
        plt.grid(True)
        plt.tick_params(labelsize=16)
        plt.legend(fontsize=16)
        
        if self.test_case == 'control':
            plt.savefig('results/classical_controllers/landing_burn_control_pure_throttle_acceleration_1s_window.png')
        elif self.test_case == 'stochastic':
            plt.savefig(f'results/classical_controllers/landing_burn_stochastic/landing_burn_control_pure_throttle_acceleration_1s_window_max_rand_{self.std_max_stochastic}.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig(f'results/classical_controllers/landing_burn_stochastic_v_ref/landing_burn_control_pure_throttle_acceleration_1s_window_max_rand_{self.std_max_stochastic_v_ref}.png')
        elif self.test_case == 'wind':
            plt.savefig('results/classical_controllers/landing_burn_pure_throttle_wind/landing_burn_control_pure_throttle_acceleration_1s_window.png')
        plt.close()
        if self.test_case == 'wind':
            self.wind_model.plot_wind_model(save_path = 'results/classical_controllers/landing_burn_pure_throttle_wind/wind_model.png')

        plt.figure(figsize=(20,10))
        # Plot CL, CD, drag, lift
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
        
        # Plot CL
        ax1.plot(self.time_vals, self.CL_vals, linewidth=3, color='blue')
        ax1.set_ylabel('CL', fontsize=20)
        ax1.set_title('Lift Coefficient', fontsize=22)
        ax1.grid(True)
        ax1.tick_params(labelsize=16)
        
        # Plot CD
        ax2.plot(self.time_vals, self.CD_vals, linewidth=3, color='red')
        ax2.set_ylabel('CD', fontsize=20)
        ax2.set_title('Drag Coefficient', fontsize=22)
        ax2.grid(True)
        ax2.tick_params(labelsize=16)
        
        # Plot lift force
        ax3.plot(self.time_vals, np.array(self.lift_vals)/1e3, linewidth=3, color='blue')
        ax3.set_xlabel('Time [s]', fontsize=20)
        ax3.set_ylabel('Lift [kN]', fontsize=20)
        ax3.set_title('Lift Force', fontsize=22)
        ax3.grid(True)
        ax3.tick_params(labelsize=16)
        
        # Plot drag force
        ax4.plot(self.time_vals, np.array(self.drag_vals)/1e3, linewidth=3, color='red')
        ax4.set_xlabel('Time [s]', fontsize=20)
        ax4.set_ylabel('Drag [kN]', fontsize=20)
        ax4.set_title('Drag Force', fontsize=22)
        ax4.grid(True)
        ax4.tick_params(labelsize=16)
        
        plt.tight_layout()
        fig.suptitle('Aerodynamic Properties', fontsize=24, y=1.02)
        
        if self.test_case == 'control':
            plt.savefig('results/classical_controllers/landing_burn_control_pure_throttle_aerodynamics.png')
        elif self.test_case == 'stochastic':
            plt.savefig(f'results/classical_controllers/landing_burn_stochastic/landing_burn_control_pure_throttle_aerodynamics_max_rand_{self.std_max_stochastic}.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig(f'results/classical_controllers/landing_burn_stochastic_v_ref/landing_burn_control_pure_throttle_aerodynamics_max_rand_{self.std_max_stochastic_v_ref}.png')
        elif self.test_case == 'wind':
            plt.savefig('results/classical_controllers/landing_burn_pure_throttle_wind/landing_burn_control_pure_throttle_aerodynamics.png')
        plt.close()