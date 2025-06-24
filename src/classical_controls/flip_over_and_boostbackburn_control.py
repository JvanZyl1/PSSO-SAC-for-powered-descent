import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyswarm import pso

from src.envs.base_environment import load_flip_over_initial_state
from src.envs.rockets_physics import compile_physics
from src.classical_controls.utils import PD_controller_single_step

optimization_history = {
    'iterations': [],
    'best_score': [],
    'parameters': []
}


# PSO : -22.0, -4.80
# Normal : -18.0, -5.0

def flip_over_pitch_control(pitch_angle_rad, max_gimbal_angle_deg, previous_pitch_angle_error_rad, previous_derivative, dt, flip_over_pitch_reference_deg, Kp_theta_flip=None, Kd_theta_flip=None):
    if Kp_theta_flip is None:
        Kp_theta_flip = -18.0
    if Kd_theta_flip is None:
        Kd_theta_flip = -5.0
    N_theta_flip = 14

    pitch_angle_error_rad = math.radians(flip_over_pitch_reference_deg) - pitch_angle_rad
    gimbal_angle_command_deg, new_derivative = PD_controller_single_step(Kp=Kp_theta_flip,
                                                                         Kd=Kd_theta_flip,
                                                                         N=N_theta_flip,
                                                                         error=pitch_angle_error_rad,
                                                                         previous_error=previous_pitch_angle_error_rad,
                                                                         previous_derivative=previous_derivative,
                                                                         dt=dt)

    gimbal_angle_command_deg = np.clip(gimbal_angle_command_deg, -max_gimbal_angle_deg, max_gimbal_angle_deg)

    return gimbal_angle_command_deg, pitch_angle_error_rad, new_derivative

def augment_action_flip_over(action,
                             max_gimbal_angle_deg):
    return action/max_gimbal_angle_deg

class FlipOverandBoostbackBurnControl:
    def __init__(self,
                 pitch_tuning_bool : bool = False,
                 Kp_theta_flip=None,
                 Kd_theta_flip=None):
        self.dt = 0.1
        self.max_gimbal_angle_deg = 10
        self.final_pitch_error_deg = 2
        self.flip_over_pitch_reference_deg = 184
        self.vx_terminal = -215
        self.pitch_tuning_bool = pitch_tuning_bool

        self.pitch_controller_lambda = lambda pitch_angle_rad, previous_pitch_angle_error_rad, previous_derivative : flip_over_pitch_control(pitch_angle_rad = pitch_angle_rad,
                                                                                                                                    max_gimbal_angle_deg = self.max_gimbal_angle_deg,
                                                                                                                                    previous_pitch_angle_error_rad = previous_pitch_angle_error_rad,
                                                                                                                                    previous_derivative = previous_derivative,
                                                                                                                                    dt = self.dt,
                                                                                                                                    flip_over_pitch_reference_deg = self.flip_over_pitch_reference_deg,
                                                                                                                                    Kp_theta_flip=Kp_theta_flip,
                                                                                                                                    Kd_theta_flip=Kd_theta_flip)
        
        self.state = load_flip_over_initial_state()
        self.simulation_step_lambda = compile_physics(dt = self.dt,
                                                      flight_phase = 'flip_over_boostbackburn')
        self.initialise_logging()
        self.initial_conditions()
        
    def initialise_logging(self):
        self.x_vals = []
        self.y_vals = []
        self.pitch_angle_deg_vals = []
        self.pitch_angle_reference_deg_vals = []
        self.time_vals = []
        self.flight_path_angle_deg_vals = []
        self.mach_number_vals = []
        self.angle_of_attack_deg_vals = []
        self.gimbal_angle_commanded_deg_vals = []
        self.u0_vals = []

        self.pitch_rate_deg_vals = []
        self.mass_vals = []
        self.mass_propellant_vals = []
        self.vx_vals = []
        self.vy_vals = []
        self.gimbal_angle_deg_vals = []
        self.flight_path_angle_rad_vals = []
        self.aero_moment_z_vals = []
        self.dynamic_pressure_vals = []
    def initial_conditions(self):
        self.gimbal_angle = 0.0
        self.previous_pitch_angle_error_rad = math.radians(self.flip_over_pitch_reference_deg) - self.state[4]
        self.pitch_angle_previous_derivative = 0.0

    def reset(self):
        self.state = load_flip_over_initial_state()
        self.previous_pitch_angle_error_rad = math.radians(self.flip_over_pitch_reference_deg) - self.state[4]
        self.pitch_angle_previous_derivative = 0.0
        self.initialise_logging()

    def closed_loop_step(self):
        gimbal_angle_command_deg, self.previous_pitch_angle_error_rad, self.pitch_angle_previous_derivative = self.pitch_controller_lambda(pitch_angle_rad = self.state[4],
                                                                                                      previous_pitch_angle_error_rad = self.previous_pitch_angle_error_rad,
                                                                                                      previous_derivative = self.pitch_angle_previous_derivative)
        action = augment_action_flip_over(action = gimbal_angle_command_deg,
                                          max_gimbal_angle_deg = self.max_gimbal_angle_deg)
        
        self.state, info = self.simulation_step_lambda(self.state, action, self.gimbal_angle, None)
        self.gimbal_angle = info['action_info']['gimbal_angle_deg']
        aero_moment_z = info['moment_dict']['aero_moment_z']
        self.aero_moment_z_vals.append(aero_moment_z)
        # state : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        self.x_vals.append(self.state[0])
        self.y_vals.append(self.state[1])
        self.pitch_angle_deg_vals.append(math.degrees(self.state[4]))
        self.pitch_angle_reference_deg_vals.append(math.degrees(math.radians(self.flip_over_pitch_reference_deg)))
        self.time_vals.append(self.state[-1])
        self.flight_path_angle_deg_vals.append(math.degrees(self.state[6]))
        self.mach_number_vals.append(info['mach_number'])
        self.angle_of_attack_deg_vals.append(math.degrees(self.state[7]))
        self.gimbal_angle_commanded_deg_vals.append(gimbal_angle_command_deg)
        self.gimbal_angle_deg_vals.append(self.gimbal_angle)
        self.u0_vals.append(action)
        self.pitch_rate_deg_vals.append(math.degrees(self.state[5]))
        self.mass_vals.append(self.state[8])
        self.mass_propellant_vals.append(self.state[9])
        self.vx_vals.append(self.state[2])
        self.vy_vals.append(self.state[3])
        self.flight_path_angle_rad_vals.append(self.state[6])
        self.dynamic_pressure_vals.append(info['dynamic_pressure'])
    def save_results(self):
        # t[s],x[m],y[m],vx[m/s],vy[m/s],mass[kg]
        save_folder = f'data/reference_trajectory/flip_over_and_boostbackburn_controls/'
        full_trajectory_path = os.path.join(save_folder, 'reference_trajectory_flip_over_and_boostbackburn_control.csv')

        data = {
            't[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'mass[kg]': self.mass_vals
        }
        pd.DataFrame(data).to_csv(full_trajectory_path, index=False)

        pitch_angle_rad_vals = np.array(self.pitch_angle_deg_vals) * np.pi / 180
        pitch_rate_rad_vals = np.array(self.pitch_rate_deg_vals) * np.pi / 180
        angle_of_attack_rad_vals = np.array(self.angle_of_attack_deg_vals) * np.pi / 180
        state_action_data = {
            'time[s]': self.time_vals,
            'x[m]': self.x_vals,
            'y[m]': self.y_vals,
            'vx[m/s]': self.vx_vals,
            'vy[m/s]': self.vy_vals,
            'theta[rad]': pitch_angle_rad_vals,
            'theta_dot[rad/s]': pitch_rate_rad_vals,
            'alpha[rad]': angle_of_attack_rad_vals,
            'gamma[rad]': self.flight_path_angle_rad_vals,
            'mass[kg]': self.mass_vals,
            'mass_propellant[kg]': self.mass_propellant_vals,
            'time[s]': self.time_vals,
            'gimbalanglecommanded[deg]': self.gimbal_angle_commanded_deg_vals,
            'u0': self.u0_vals
        }
        state_action_path = os.path.join(save_folder, 'state_action_flip_over_and_boostbackburn_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)

    def plot_results(self):
        plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
        plt.suptitle('Flip Over and Boostback Burn Control', fontsize = 32)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(np.array(self.x_vals)/1000, np.array(self.y_vals)/1000, linewidth = 4, color = 'blue')
        ax1.set_xlabel('x [km]', fontsize = 20)
        ax1.set_ylabel('y [km]', fontsize = 20)
        ax1.set_title('Flight Path', fontsize = 22)
        ax1.tick_params(axis='both', which='major', labelsize=16)
        ax1.grid()

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_vals, self.vx_vals, linewidth = 4, color = 'blue')
        ax2.set_xlabel('Time [s]', fontsize = 20)
        ax2.set_ylabel('Velocity [m/s]', fontsize = 20)
        ax2.set_title('Horizontal Velocity', fontsize = 22)
        ax2.tick_params(axis='both', which='major', labelsize=16)
        ax2.grid()

        ax3 = plt.subplot(gs[1, 0])
        ax3.plot(self.time_vals, self.pitch_angle_deg_vals, linewidth = 4, color = 'blue', label = 'Pitch Angle')
        ax3.plot(self.time_vals, self.pitch_angle_reference_deg_vals, linewidth = 3, linestyle = '--', color = 'red', label = 'Pitch Angle Reference')
        ax3.plot(self.time_vals, self.flight_path_angle_deg_vals, linewidth = 4, color = 'green', label = 'Flight Path Angle')
        ax3.set_xlabel('Time [s]', fontsize = 20)
        ax3.set_ylabel(r'Angle [$^\circ$]', fontsize = 20)
        ax3.set_title('Pitch and Flight Path Angles', fontsize = 22)
        ax3.tick_params(axis='both', which='major', labelsize=16)
        ax3.grid()
        ax3.legend(fontsize = 16, loc = 'upper right')

        ax4 = plt.subplot(gs[1, 1])
        ax4.plot(self.time_vals, self.gimbal_angle_commanded_deg_vals, linewidth = 3, color = 'pink', linestyle = '--', label = 'Commanded')
        ax4.plot(self.time_vals, self.gimbal_angle_deg_vals, linewidth = 4, color = 'blue', label = 'Actual')
        ax4.set_xlabel('Time [s]', fontsize = 20)
        ax4.set_ylabel(r'Angle [$^\circ$]', fontsize = 20)
        ax4.set_title('Gimbal Angle (Commanded and Actual)', fontsize = 22)
        ax4.tick_params(axis='both', which='major', labelsize=16)
        ax4.grid()
        ax4.legend(fontsize = 16)
        if self.pitch_tuning_bool:
            plt.savefig(f'results/classical_controllers/flip_over_and_boostbackburn_pitch_tuning.png')
        else:
            plt.savefig(f'results/classical_controllers/flip_over_and_boostbackburn.png')
        plt.close()

    def run_closed_loop(self):
        # vx < - 150 m/s
        if self.pitch_tuning_bool:
            pitch_angle_error_deg = self.flip_over_pitch_reference_deg - math.degrees(self.state[4])
            time_settled = 0.0
            time_ran = 0.0
            while time_settled < 1 and time_ran < 200:
                self.closed_loop_step()
                pitch_angle_error_deg = abs(self.flip_over_pitch_reference_deg - math.degrees(self.state[4]))
                time_ran += self.dt
                if pitch_angle_error_deg < self.final_pitch_error_deg:
                    time_settled += self.dt
                else:
                    time_settled = 0.0
        else:
            time_ran = 0.0
            vx = self.state[2]
            while vx > self.vx_terminal and time_ran < 280:
                self.closed_loop_step()
                vx = self.state[2]
                time_ran += self.dt
        self.plot_results()
        self.save_results()

    def performance_metrics(self):
        reward = 0
        for i in range(len(self.pitch_angle_deg_vals)):
            pitch_error = abs(self.flip_over_pitch_reference_deg - self.pitch_angle_deg_vals[i])
            reward -= pitch_error/10
            
            reward -= abs(self.gimbal_angle_commanded_deg_vals[i])/20
            
            reward -= abs(self.pitch_rate_deg_vals[i])/5
            
            if i > 0:
                current_pitch_error = self.flip_over_pitch_reference_deg - self.pitch_angle_deg_vals[i]
                pitch_rate_change = self.pitch_rate_deg_vals[i] - self.pitch_rate_deg_vals[i-1]
                if (current_pitch_error > 0 and pitch_rate_change < 0) or (current_pitch_error < 0 and pitch_rate_change > 0):
                    reward -= abs(pitch_rate_change)/2
            
        final_pitch_error = abs(self.flip_over_pitch_reference_deg - self.pitch_angle_deg_vals[-1])
        reward -= final_pitch_error * 10
        
        return reward

def objective_func_lambda(individual):
    flip_over = FlipOverandBoostbackBurnControl(pitch_tuning_bool=True, Kp_theta_flip=individual[0], Kd_theta_flip=individual[1])
    flip_over.run_closed_loop()
    obj = -flip_over.performance_metrics()
    
    if hasattr(objective_func_lambda, 'iteration'):
        objective_func_lambda.iteration += 1
    else:
        objective_func_lambda.iteration = 1
    
    if not optimization_history['best_score'] or obj < min(optimization_history['best_score']):
        optimization_history['iterations'].append(objective_func_lambda.iteration)
        optimization_history['best_score'].append(obj)
        optimization_history['parameters'].append(individual.copy())
    
    return obj

def save_gains(xopt):
    xopt = np.array(xopt)
    print("Optimal parameters:", xopt)
    gains = pd.DataFrame({'Kp_theta_flip': [xopt[0]], 'Kd_theta_flip': [xopt[1]]})
    gains.to_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/gains.csv', index=False)

def plot_optimization_progress():    
    plt.figure(figsize=(12, 8))
    plt.plot(optimization_history['iterations'], optimization_history['best_score'], 'o-', linewidth=4)
    plt.title('Optimization Progress - Flip Over and Boostback Burn Controller', fontsize=24)
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Best Objective Value (lower is better)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)
    plt.savefig('results/classical_controllers/flip_over_optimization_progress.png')
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    param_history = np.array(optimization_history['parameters'])
    ax.plot(optimization_history['iterations'], param_history[:, 0], 'g-', linewidth=4, label='Kp_theta_flip')
    ax.plot(optimization_history['iterations'], param_history[:, 1], 'm-', linewidth=4, label='Kd_theta_flip')
    ax.set_title('Parameter Convergence - Flip Over and Boostback Burn Controller', fontsize=24)
    ax.set_xlabel('Iteration', fontsize=20)
    ax.set_ylabel('Parameter Value', fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True)
    plt.savefig('results/classical_controllers/flip_over_parameter_convergence.png')
    plt.close()

def tune_flip_over_and_boostbackburn():
    global optimization_history
    optimization_history = {
        'iterations': [],
        'best_score': [],
        'parameters': []
    }
    
    if hasattr(objective_func_lambda, 'iteration'):
        delattr(objective_func_lambda, 'iteration')
    
    lb = [-22.0, -8.0]
    ub = [-10.0, -2.0]
    
    xopt, fopt = pso(
        objective_func_lambda,
        lb, 
        ub,
        swarmsize=20,
        omega=0.5,
        phip=0.5,
        phig=0.5,
        maxiter=30,
        minstep=1e-6,
        minfunc=1e-6,
        debug=True,
    )
    
    plot_optimization_progress()
    save_gains(xopt)
    return FlipOverandBoostbackBurnControl(Kp_theta_flip=xopt[0], Kd_theta_flip=xopt[1])

class FlipOverandBoostbackBurnTuning:
    def __init__(self, tune_bool=False):
        if tune_bool:
            self.env = tune_flip_over_and_boostbackburn()
        else:
            self.env = FlipOverandBoostbackBurnControl()

    def run_closed_loop(self):
        self.env.run_closed_loop()

    def plot_results(self):
        self.env.plot_results()

    def save_results(self):
        self.env.save_results()