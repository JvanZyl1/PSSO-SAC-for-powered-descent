import os
import dill
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.envs.load_initial_states import load_landing_burn_initial_state
from src.envs.rockets_physics import compile_physics

class LandingBurn_PDcontrol:
    def __init__(self, test_case = 'control'):
        self.dt = 0.1
        self.max_q = 65e3 # [Pa]
        self.std_max_stochastic_v_ref = 0.5
        self.test_case = test_case
        assert self.test_case in ['control', 'stochastic_v_ref'], 'Invalid test case'
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                    flight_phase = 'landing_burn_pure_throttle_Pcontrol')
        
        self.v_opt_fcn = dill.load(open('data/reference_trajectory/landing_burn_controls/landing_initial_velocity_profile_guess.pkl', 'rb'))

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

        self.tau_vals = []

        self.mach_number_vals = []

    def v_ref_fcn(self, y):
        if self.test_case == 'control':
            v_ref = self.v_opt_fcn(y)
            u1 = v_ref/self.speed0 * 2 - 1
            return u1, v_ref
        elif self.test_case == 'stochastic_v_ref':
            v_ref = self.v_opt_fcn(y)
            u1 = v_ref/self.speed0 * 2 - 1
            u1_aug = u1 * (1 + np.random.uniform(0.0, self.std_max_stochastic_v_ref))
            v_ref_aug = (u1_aug + 1)/2 * self.speed0
            return u1_aug, v_ref_aug
        
    def closed_loop_step(self):
        u1, v_ref = self.v_ref_fcn(self.y)
        self.u1_vref_vals.append(u1)

        self.state, info = self.simulation_step_lambda(self.state, np.array([[v_ref]]), None) # Takes in augmented action
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
    
    def run_closed_loop(self):
        simulation_steps = 0
        max_steps = 50000
        
        while self.mass_propellant > 0 and self.y > 1 and self.dynamic_pressure < self.max_q and simulation_steps < max_steps and self.vy < 0:
            self.closed_loop_step()
            simulation_steps += 1
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
        elif self.test_case == 'stochastic_v_ref':
            plt.title(f'Acceleration, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=22)
        plt.axhline(y=6.0, color='red', linestyle='--', linewidth=2, label='Maximum')
        plt.grid(True)
        plt.tick_params(labelsize=16)
        if self.test_case == 'control':
            plt.savefig('results/classical_controllers/landing_burn_v_ref_control/landing_burn_control_pure_throttle_acceleration.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig(f'results/classical_controllers/landing_burn_v_ref_control_stochastic/landing_burn_control_pure_throttle_acceleration_max_rand_{self.std_max_stochastic_v_ref}.png')
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
        save_folder = f'data/reference_trajectory/landing_burn_v_ref_control/'
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
            'u1_vref': self.u1_vref_vals
        }
        state_action_path = os.path.join(save_folder, 'state_action_landing_burn_pure_throttle_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)

    def plot_results(self):
        V_abs = np.sqrt(np.array(self.vx_vals)**2 + np.array(self.vy_vals)**2)
        
        plt.figure(figsize=(20,15))
        if self.test_case == 'control':
            plt.suptitle('Landing Burn (Initial Guess) Pure Throttle Control', fontsize=24)
        elif self.test_case == 'stochastic_v_ref':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Stochastic v_ref, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=24)
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
        ax6.plot(self.time_vals, np.array(self.dynamic_pressure_vals)/1000, linewidth=4, color = 'blue')
        ax6.set_xlabel(r'Time [$s$]', fontsize=20)
        ax6.set_ylabel(r'Dynamic pressure [$kPa$]', fontsize=20)
        ax6.set_title('Dynamic pressure', fontsize=22)
        ax6.grid(True)
        ax6.tick_params(labelsize=16)

        if self.test_case == 'control':
            plt.savefig('results/classical_controllers/landing_burn_v_ref_control/landing_burn_control_pure_throttle.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig('results/classical_controllers/landing_burn_v_ref_control_stochastic/landing_burn_control_pure_throttle.png')
        plt.close()

        plt.figure(figsize=(20,15))
        if self.test_case == 'control':
            plt.suptitle('Landing Burn (Initial Guess) Pure Throttle Angular Control', fontsize=24)
        elif self.test_case == 'stochastic_v_ref':
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Angular Control, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=24)
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
            plt.savefig('results/classical_controllers/landing_burn_v_ref_control/landing_burn_control_pure_throttle_angular_controls.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig('results/classical_controllers/landing_burn_v_ref_control_stochastic/landing_burn_control_pure_throttle_angular_controls.png')
        plt.close()

        # if stochastic_v_ref, plot the reference velocity
        if self.test_case == 'stochastic_v_ref':
            plt.figure(figsize=(20,15))
            plt.suptitle(f'Landing Burn (Initial Guess) Pure Throttle Reference Velocity, stochastic v_ref with max rand {self.std_max_stochastic_v_ref}', fontsize=24)
            plt.plot(self.time_vals, self.vy_ref_vals, linewidth=4, color = 'blue')
            plt.xlabel(r'Time [$s$]', fontsize=20)
            plt.ylabel(r'Velocity [$m/s$]', fontsize=20)
            plt.title('Reference velocity', fontsize=22)
            plt.grid(True)
            plt.tick_params(labelsize=16)
            plt.savefig(f'results/classical_controllers/landing_burn_v_ref_control_stochastic/landing_burn_control_pure_throttle_reference_velocity_max_rand_{self.std_max_stochastic_v_ref}.png')
            plt.close()

        # Plot the acceleration over 1 second in gs through interpolation
        
        speed_vals = np.sqrt(np.array(self.vx_vals)**2 + np.array(self.vy_vals)**2)
        # as dt = 0.1, every 10th point is 1 second
        speed_vals = speed_vals[::10]
        time_vals = self.time_vals[::10]
        # Calculate acceleration as the gradient of velocity
        acceleration_vals = np.gradient(speed_vals, time_vals)
        # Plot the results
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
            plt.savefig('results/classical_controllers/landing_burn_v_ref_control/landing_burn_control_pure_throttle_acceleration_1s_window.png')
        elif self.test_case == 'stochastic_v_ref':
            plt.savefig(f'results/classical_controllers/landing_burn_v_ref_control_stochastic/landing_burn_control_pure_throttle_acceleration_1s_window_max_rand_{self.std_max_stochastic_v_ref}.png')
        plt.close()