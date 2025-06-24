import csv
import math
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from src.envs.utils.aerodynamic_coefficients import rocket_CD_compiler, rocket_CL_compiler
from src.envs.load_initial_states import load_landing_burn_initial_state
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.classical_controls.utils import PD_controller_single_step
from src.envs.utils.grid_fin_aerodynamics import compile_grid_fin_Ca, compile_grid_fin_Cn

def max_v_allowed():
    max_q = 28000
    def v_allowed(y):
        air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        v_max = math.sqrt(2 * max_q / (air_density))
        return v_max
    return v_allowed

v_allowed = max_v_allowed()                                                                                     

class AerodynamicStabilityDescent:
    def __init__(self, ACS_enabled_bool = False):
        self.state = load_landing_burn_initial_state()
        self.x, self.y, self.vx, self.vy, self.theta, self.theta_dot, self.gamma, self.alpha, self.mass, self.mass_propellant, self.time = self.state
        self.CD_func = lambda M, alpha_rad: rocket_CD_compiler()(M, math.degrees(alpha_rad)) # Mach, alpha [deg]
        self.CL_func = lambda M, alpha_rad: rocket_CL_compiler()(M, math.degrees(alpha_rad)) # Mach, alpha [deg]
        self.Ca_func = compile_grid_fin_Ca() # Mach
        self.Cn_func = compile_grid_fin_Cn() # Mach, alpha [rad]
        # Read sizing results
        sizing_results = {}
        with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                sizing_results[row[0]] = row[2]
        self.m_burn_out = float(sizing_results['Structural mass stage 1 (ascent)'])*1000
        self.T_e = float(sizing_results['Thrust engine stage 1'])
        self.v_ex = float(sizing_results['Exhaust velocity stage 1'])
        self.n_e = float(sizing_results['Number of engines gimballed stage 1'])
        self.mdot = self.T_e/self.v_ex * self.n_e
        self.frontal_area = float(sizing_results['Rocket frontal area'])
        self.x_cog = 20
        self.x_cop = 0.75 * float(sizing_results['Stage 1 height '])
        self.x_gf = float(sizing_results['Stage 1 height ']) - 2.0
        self.rocket_radius = float(sizing_results['Rocket Radius'])

        self.dt = 0.1
        self.inertia = 1e9
        self.alpha_effective = self.gamma - self.theta - math.pi
        self.initialise_logging()
        if ACS_enabled_bool:
            self.initialise_ACS_logging()
        self.ACS_enabled_bool = ACS_enabled_bool

        # Thrust control: velocity reference
        df_reference = pd.read_csv('data/reference_trajectory/landing_burn_controls/landing_initial_guess_reference_profile.csv')
        self.v_opt_fcn = scipy.interpolate.interp1d(df_reference['altitude'], df_reference['velocity'], kind='cubic', fill_value='extrapolate')
        #self.v_opt_fcn = max_v_allowed()
        self.Kp_throttle = -0.5
        self.Kd_throttle = 0.0
        self.N_throttle = 10.0
        self.speed = math.sqrt(self.vx**2 + self.vy**2)
        self.previous_velocity_error = self.v_opt_fcn(self.y) - self.speed
        self.previous_velocity_error_derivative = 0.0
        number_of_engines_min = 3
        minimum_engine_throttle = 0.4
        self.nominal_throttle = (number_of_engines_min * minimum_engine_throttle) / int(sizing_results['Number of engines gimballed stage 1'])

        self.grid_fin_area = float(sizing_results['S_grid_fins'])

    def __call__(self):
        self.run_closed_loop()

    def initialise_logging(self):
        self.x_log = []
        self.y_log = []
        self.vx_log = []
        self.vy_log = []
        self.theta_log = []
        self.theta_dot_log = []
        self.gamma_log = []
        self.alpha_effective_log = []
        self.aero_x_log = []
        self.aero_y_log = []
        self.aero_moments_z_log = []
        self.lift_log = []
        self.drag_log = []
        self.alpha_log = []
        self.mass_log = []
        self.mass_propellant_log = []
        self.time_log = []
        self.mach_number_log = []
        self.dynamic_pressure_log = []
        self.CL_log = []
        self.CD_log = []
        self.throttle_log = []
        self.speed_log = []
        self.speed_ref_log = []
        self.aero_force_parallel_log = []
        self.aero_force_perpendicular_log = []

    def log_data(self):
        self.x_log.append(self.x)
        self.y_log.append(self.y)
        self.vx_log.append(self.vx)
        self.vy_log.append(self.vy)
        self.theta_log.append(self.theta)
        self.theta_dot_log.append(self.theta_dot)
        self.gamma_log.append(self.gamma)
        self.alpha_effective_log.append(self.alpha_effective)
        self.aero_x_log.append(self.aero_x)
        self.aero_y_log.append(self.aero_y)
        self.aero_moments_z_log.append(self.aero_moments_z)
        self.lift_log.append(self.lift)
        self.drag_log.append(self.drag)
        self.alpha_log.append(self.alpha)
        self.mass_log.append(self.mass)
        self.mass_propellant_log.append(self.mass_propellant)
        self.time_log.append(self.time)
        self.dynamic_pressure_log.append(self.dynamic_pressure)    
        self.CL_log.append(self.C_L)
        self.CD_log.append(self.C_D)  
        self.mach_number_log.append(self.mach_number)
        self.throttle_log.append(self.throttle)
        self.speed_log.append(self.speed)
        self.speed_ref_log.append(self.speed_ref)
        self.aero_force_parallel_log.append(self.aero_force_parallel)
        self.aero_force_perpendicular_log.append(self.aero_force_perpendicular)
        self.u0_log.append(self.u0)
    def initialise_ACS_logging(self):
        self.alpha_local_left_vals = []
        self.alpha_local_right_vals = []
        self.Ca_vals = []
        self.Cn_L_vals = []
        self.Cn_R_vals = []
        self.gf_force_parallel_vals = []
        self.gf_force_perpendicular_vals = []
        self.gf_moment_z_vals = []
        self.gf_force_x_vals = []
        self.gf_force_y_vals = []
        self.u0_log = []
    def log_ACS_data(self):
        self.alpha_local_left_vals.append(self.alpha_local_left)
        self.alpha_local_right_vals.append(self.alpha_local_right)
        self.Ca_vals.append(self.Ca)
        self.Cn_L_vals.append(self.Cn_L)
        self.Cn_R_vals.append(self.Cn_R)
        self.gf_force_parallel_vals.append(self.gf_force_parallel)
        self.gf_force_perpendicular_vals.append(self.gf_force_perpendicular)
        self.gf_moment_z_vals.append(self.gf_moment_z)
        self.gf_force_x_vals.append(self.gf_force_x)
        self.gf_force_y_vals.append(self.gf_force_y)

    def throttle_control(self):
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
        u0 = 2 * (throttle - 0.5)
        return throttle, u0
    
    def grid_fin_physics(self, delta_left, delta_right):
        qS = self.dynamic_pressure * self.grid_fin_area
        self.alpha_local_left = self.alpha_effective - delta_left
        self.alpha_local_right = self.alpha_effective - delta_right
        self.Ca = self.Ca_func(self.mach_number)
        self.Cn_L = self.Cn_func(self.mach_number, self.alpha_local_left)
        self.Cn_R = self.Cn_func(self.mach_number, self.alpha_local_right)
        Ca_total = self.Ca * 4 # four grid fins indep local alpha
        self.Fa_gf_total = Ca_total * qS

        gf_N_L = self.Cn_L * qS
        gf_N_R = self.Cn_R * qS
        self.Fn_gf_total = gf_N_L + gf_N_R

        # F_para = F_a (fixed) + F_a * (cos(delta_L) + cos(delta_R)) + F_n_L * sin(delta_L) + F_n_R * sin(delta_R)
        self.gf_force_parallel = qS * (self.Ca * (2 + math.cos(delta_left) + math.cos(delta_right)) +
                                  self.Cn_L * math.sin(delta_left) + self.Cn_R * math.sin(delta_right))
        self.gf_force_perpendicular = qS * (-self.Ca * (math.sin(delta_left) + math.sin(delta_right)) +
                                        self.Cn_L * math.cos(delta_left) +
                                         self.Cn_R * math.cos(delta_right))
        
        self.gf_moment_z = -(self.x_gf - self.x_cog) * self.gf_force_perpendicular +\
                    self.rocket_radius * qS * (
                        self.Ca * (math.cos(delta_right) - math.cos(delta_left))
                        - self.Cn_L * math.sin(delta_left)
                        + self.Cn_R * math.sin(delta_right)
                    )
        
        self.gf_force_x = self.gf_force_parallel * math.cos(self.theta) + self.gf_force_perpendicular * math.sin(self.theta)
        self.gf_force_y = self.gf_force_parallel * math.sin(self.theta) - self.gf_force_perpendicular * math.cos(self.theta)
        return self.gf_force_x, self.gf_force_y, self.gf_moment_z

    def step(self):
        self.alpha_effective = self.gamma - (self.theta + math.pi)
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(self.y)
        self.speed = math.sqrt(self.vx**2 + self.vy**2)
        self.dynamic_pressure = 0.5 * density * self.speed**2
        self.throttle, self.u0 = self.throttle_control()
        if speed_of_sound != 0.0:
            self.mach_number = self.speed/speed_of_sound
            self.C_L = self.CL_func(self.mach_number, self.alpha_effective) # Mach, alpha [rad]
            self.C_D = self.CD_func(self.mach_number, self.alpha_effective) # Mach, alpha [rad]
        else:
            self.mach_number = 0.0
            self.C_L = 0.0
            self.C_D = 0.0
        self.drag = self.dynamic_pressure * self.C_D * self.frontal_area
        self.lift = self.dynamic_pressure * self.C_L * self.frontal_area

        self.aero_force_parallel = -self.drag * math.cos(self.alpha_effective) - self.lift * math.sin(self.alpha_effective)
        self.aero_force_perpendicular = -self.drag * math.sin(self.alpha_effective) - self.lift * math.cos(self.alpha_effective)
        d_cp_cg = self.x_cog - self.x_cop
        self.aero_x = self.aero_force_parallel * math.cos(self.theta) + self.aero_force_perpendicular * math.sin(self.theta)
        self.aero_y = self.aero_force_parallel * math.sin(self.theta) - self.aero_force_perpendicular * math.cos(self.theta)
        self.aero_moments_z = self.aero_force_perpendicular * d_cp_cg

        # ACS
        if self.ACS_enabled_bool:
            acs_force_x, acs_force_y, acs_moment_z = self.grid_fin_physics(math.radians(0.0), math.radians(0.0))
        else:
            acs_force_x = 0.0
            acs_force_y = 0.0
            acs_moment_z = 0.0

        # Thrust
        T_parallel = self.T_e * self.n_e * self.throttle
        T_x = T_parallel * math.cos(self.theta)
        T_y = T_parallel * math.sin(self.theta)
        
        # Acceleration linear
        
        a_x = (self.aero_x + T_x + acs_force_x)/self.mass
        a_y = (self.aero_y + T_y + acs_force_y)/self.mass

        # Velocity update
        self.vx += a_x * self.dt
        self.vy += a_y * self.dt

        # Position update
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        # Moments update
        theta_dot_dot = (self.aero_moments_z + acs_moment_z) / self.inertia
        self.theta_dot += theta_dot_dot * self.dt
        self.theta += self.theta_dot * self.dt
        self.gamma = math.atan2(self.vy, self.vx)

        if self.theta > 2 * math.pi:
            self.theta -= 2 * math.pi
        if self.gamma < 0:
            self.gamma = 2 * math.pi + self.gamma

        self.alpha = self.theta - self.gamma
        
        # Mass update
        self.mass -= self.mdot * self.dt * self.throttle
        self.mass_propellant -= self.mdot * self.dt * self.throttle

        # Time update
        self.time += self.dt

    def save_results(self):
        # Create save directory if it doesn't exist
        save_folder = 'data/reference_trajectory/landing_burn_controls/'
        os.makedirs(save_folder, exist_ok=True)
        
        # Create a DataFrame from the collected data
        trajectory_data = {
            't[s]': self.time_log,
            'x[m]': self.x_log,
            'y[m]': self.y_log,
            'vx[m/s]': self.vx_log,
            'vy[m/s]': self.vy_log,
            'mass[kg]': self.mass_log
        }
        
        # Save the trajectory data
        trajectory_path = os.path.join(save_folder, 'reference_trajectory_landing_burn_control.csv')
        pd.DataFrame(trajectory_data).to_csv(trajectory_path, index=False)

        # Save state-action data
        state_action_data = {
            'time[s]': self.time_log,
            'x[m]': self.x_log,
            'y[m]': self.y_log,
            'vx[m/s]': self.vx_log,
            'vy[m/s]': self.vy_log,
            'theta[rad]': self.theta_log,
            'theta_dot[rad/s]': self.theta_dot_log,
            'gamma[rad]': self.gamma_log,
            'alpha[rad]': self.alpha_log,
            'mass[kg]': self.mass_log,
            'mass_propellant[kg]': self.mass_propellant_log,
            'time[s]': self.time_log,
            'u0': self.u0_log
        }
        
        state_action_path = os.path.join(save_folder, 'state_action_landing_burn_control.csv')
        pd.DataFrame(state_action_data).to_csv(state_action_path, index=False)
        


    def run_closed_loop(self):
        while self.mass_propellant > 0.0 and self.y > 0.0 and abs(self.alpha_effective) < math.radians(5) and self.speed > 20:
            self.step()
            self.log_data()
            if self.ACS_enabled_bool:
                self.log_ACS_data()
        if self.mass_propellant < 0.0:
            print('Propellant depleted')
        elif self.y < 0.0:
            print('Landed')
        elif abs(self.alpha_effective) > math.radians(5):
            print('Alpha effective too high')
        elif self.speed < 20:
            print('Speed too low')
        print(f'Altitude: {self.y} m, Speed: {self.speed} m/s, Throttle: {self.throttle}, u0: {self.u0}')
        self.save_results()
        self.plot_results()
        if self.ACS_enabled_bool:
            self.plot_ACS_results()

    def plot_results(self):
        self.pitch_down_deg = np.rad2deg(np.array(self.theta_log)) + 180
        plt.figure(figsize=(20,15))
        plt.suptitle('Aerodynamic Stability Descent : No pitch damping', fontsize=16)
        # 3 x 3 plot
        # alpha effective   | Mach      | Dynamic Pressure
        # Lift              | Drag      | Moment
        # Pitch (down), FPA | Pitch Rate| CL and CD
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.time_log, np.rad2deg(np.array(self.alpha_effective_log)), linewidth=2, color='blue')
        ax1.set_ylabel(r'$\alpha_{eff}$ [$^{\circ}$]', fontsize=14)
        ax1.grid(True)
        
        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_log, np.array(self.mach_number_log), linewidth=2, color='blue')
        ax2.set_ylabel(r'$M$', fontsize=14)
        ax2.grid(True)

        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(self.time_log, np.array(self.dynamic_pressure_log)/1e3, linewidth=2, color='blue')
        ax3.set_ylabel(r'$q$ [kPa]', fontsize=14)
        ax3.grid(True)

        ax4 = plt.subplot(gs[1, 0])
        ax4.plot(self.time_log, np.array(self.lift_log)/1e3, linewidth=2, color='magenta')
        ax4.plot(self.time_log, np.array(self.drag_log)/1e3, linewidth=2, color='cyan')
        ax4.set_ylabel(r'Force [kN]', fontsize=14)
        ax4.legend(['Lift', 'Drag'], fontsize=12)
        ax4.grid(True)

        ax5 = plt.subplot(gs[1, 1])
        ax5.plot(self.time_log, np.array(self.speed_log), linewidth=2, color='blue')
        ax5.plot(self.time_log, np.array(self.speed_ref_log), linewidth = 2, linestyle = '--', color = 'red')
        ax5.set_ylabel(r'Speed [m/s]', fontsize=14)
        ax5.set_yscale('log')
        ax5.legend(['Speed', 'Speed Reference'], fontsize=12)
        ax5.grid(True)

        ax6 = plt.subplot(gs[1, 2])
        ax6.plot(self.time_log, np.array(self.aero_moments_z_log)/1e3, linewidth=2, color='blue')
        if self.ACS_enabled_bool:
            ax6.plot(self.time_log, np.array(self.gf_moment_z_vals)/1e3, linewidth=2, color='green')
            ax6.legend(['Aero', 'Grid Fins'], fontsize=12)
        ax6.set_ylabel(r'$M_z$ [kNm]', fontsize=14)
        ax6.grid(True)

        ax7 = plt.subplot(gs[2, 0])
        ax7.plot(self.time_log, np.array(self.pitch_down_deg), linewidth=2, color='blue', label='Pitch Down')
        ax7.plot(self.time_log, np.rad2deg(np.array(self.gamma_log)), linewidth=2, color='red', label='Flight Path Angle')
        ax7.set_ylabel(r'Angle [$^{\circ}$]', fontsize=14)
        ax7.set_xlabel(r'Time [s]', fontsize=14)
        ax7.grid(True)
        ax7.legend()

        ax8 = plt.subplot(gs[2, 1])
        ax8.plot(self.time_log, np.rad2deg(np.array(self.theta_dot_log)), linewidth=2, color='blue', label='Pitch Rate')
        ax8.set_ylabel(r'$\dot{\theta}$ [$^{\circ}$/s]', fontsize=14)
        ax8.set_xlabel(r'Time [s]', fontsize=14)
        ax8.grid(True)
        ax8.legend()

        ax9 = plt.subplot(gs[2, 2])
        ax9.plot(self.time_log, np.array(self.CL_log), linewidth=2, color='magenta', label='CL')
        ax9.plot(self.time_log, np.array(self.CD_log), linewidth=2, color='cyan', label='CD')
        ax9.set_ylabel(r'Coefficient', fontsize=14)
        ax9.set_xlabel(r'Time [s]', fontsize=14)
        ax9.grid(True)
        ax9.legend()

        plt.savefig('results/classical_controllers/descent_stability/landing_burn_stability_test.png')
        plt.close()

        plt.figure(figsize=(20,15))
        # plot throttle and u0
        plt.subplot(2,1,1)
        plt.plot(self.time_log, np.array(self.throttle_log), linewidth=2, color='blue', label='Throttle')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(self.time_log, np.array(self.u0_log), linewidth=2, color='red', label='u0')
        plt.legend()
        plt.savefig('results/classical_controllers/descent_stability/landing_burn_stability_test_throttle.png')
        plt.close()

    def plot_ACS_results(self):
        plt.figure(figsize=(20,15))
        plt.suptitle('Aerodynamic Stability Descent : No pitch damping', fontsize=16)
        # 3 x 3 plot
        # alpha local left & right      | Ca                        | Cn left & right
        # gf force parallel             | gf force perpendicular    | gf moment z
        # gf force x                    | gf force y                | dynamic pressure
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(self.time_log, np.rad2deg(np.array(self.alpha_local_left_vals)), linewidth=2, color='orange')
        ax1.plot(self.time_log, np.rad2deg(np.array(self.alpha_local_right_vals)), linewidth=2, color='green')
        ax1.set_ylabel(r'$\alpha_{local}$ [$^{\circ}$]', fontsize=14)
        ax1.grid(True)
        ax1.legend(['Left', 'Right'], fontsize=12)

        ax2 = plt.subplot(gs[0, 1])
        ax2.plot(self.time_log, np.array(self.Ca_vals), linewidth=2, color='blue')
        ax2.set_ylabel(r'$C_a$', fontsize=14)
        ax2.grid(True)

        ax3 = plt.subplot(gs[0, 2])
        ax3.plot(self.time_log, np.array(self.Cn_L_vals), linewidth=2, color='orange')
        ax3.plot(self.time_log, np.array(self.Cn_R_vals), linewidth=2, color='green')
        ax3.set_ylabel(r'$C_n$', fontsize=14)
        ax3.grid(True)

        ax4 = plt.subplot(gs[1, 0])
        ax4.plot(self.time_log, np.array(self.gf_force_parallel_vals)/1e3, linewidth=2, color='blue')
        ax4.plot(self.time_log, np.array(self.aero_force_parallel_log)/1e3, linewidth=2, color='orange')
        ax4.set_ylabel(r'$F_{parallel}$ [kN]', fontsize=14)
        ax4.legend(['Grid Fins', 'Aero'], fontsize=12)
        ax4.grid(True)

        ax5 = plt.subplot(gs[1, 1])
        ax5.plot(self.time_log, np.array(self.gf_force_perpendicular_vals)/1e3, linewidth=2, color='blue')
        ax5.plot(self.time_log, np.array(self.aero_force_perpendicular_log)/1e3, linewidth=2, color='orange')
        ax5.set_ylabel(r'$F_{perpendicular}$ [kN]', fontsize=14)
        ax5.legend(['Grid Fins', 'Aero'], fontsize=12)
        ax5.grid(True)

        ax6 = plt.subplot(gs[1, 2])
        ax6.plot(self.time_log, np.array(self.gf_moment_z_vals)/1e3, linewidth=2, color='blue')
        ax6.plot(self.time_log, np.array(self.aero_moments_z_log)/1e3, linewidth=2, color='orange')
        ax6.set_ylabel(r'$M_z$ [kNm]', fontsize=14)
        ax6.legend(['Grid Fins', 'Aero'], fontsize=12)
        ax6.grid(True)

        ax7 = plt.subplot(gs[2, 0])
        ax7.plot(self.time_log, np.array(self.gf_force_x_vals)/1e3, linewidth=2, color='blue')
        ax7.plot(self.time_log, np.array(self.aero_x_log)/1e3, linewidth=2, color='orange')
        ax7.set_ylabel(r'$F_x$ [kN]', fontsize=14)
        ax7.legend(['Grid Fins', 'Aero'], fontsize=12)
        ax7.grid(True)

        ax8 = plt.subplot(gs[2, 1])
        ax8.plot(self.time_log, np.array(self.gf_force_y_vals)/1e3, linewidth=2, color='blue')
        ax8.plot(self.time_log, np.array(self.aero_y_log)/1e3, linewidth=2, color='orange')
        ax8.set_ylabel(r'$F_y$ [kN]', fontsize=14)
        ax8.legend(['Grid Fins', 'Aero'], fontsize=12)
        ax8.grid(True)

        ax9 = plt.subplot(gs[2, 2])
        ax9.plot(self.time_log, np.array(self.mach_number_log), linewidth=2, color='blue')
        ax9.set_ylabel(r'$M$', fontsize=14)
        ax9.grid(True)

        plt.savefig('results/classical_controllers/descent_stability/landing_burn_stability_test_acs.png')
        plt.close()

if __name__ == '__main__':
    test_aerodynamic_stability_descent = AerodynamicStabilityDescent(ACS_enabled_bool=True)
    test_aerodynamic_stability_descent()

        
