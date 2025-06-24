import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import dill
from src.RocketSizing.functions.staging import staging_p1_reproduction
from src.RocketSizing.functions.rocket_radius_calc import new_radius_func
from src.RocketSizing.functions.rocket_dimensions import rocket_dimensions
from src.RocketSizing.functions.cop_estimation import cop_func, plot_cop_func
import csv

R_earth = 6378137 # [m]

def fix_csv():
    data = pd.read_csv('data/reference_trajectory/SizingSimulation/reference_trajectory_endo.csv')
    data.interpolate(method='linear', inplace=True)
    data.to_csv('data/reference_trajectory/SizingSimulation/reference_trajectory_endo_clean.csv', index=False)

class create_rocket_configuration:
    def __init__(self,
                 dv_a_1_loss : float = 1391.0,
                 dv_a_2_loss : float = 710.0,
                 dv_d_1_loss : float = 1401.0,
                 dv_d_1_star : float = 1709):
        
        # Constants
        self.max_dynamic_pressure = 70000 # [Pa]
        
        # Always LEO
        self.m_payload = 100e3 # kg
        self.semi_major_axis = R_earth + 200e3 # [m]
        self.number_of_stages = 2

        # Load Raptor constants
        self.load_raptor_constants()

        # stage rocket
        self.dv_a_loss = [dv_a_1_loss, dv_a_2_loss]
        self.dv_d_1_loss = dv_d_1_loss
        self.dv_d_1_star = dv_d_1_star
        self.stage_rocket(dv_a_loss = self.dv_a_loss,
                          dv_d_1_loss = self.dv_d_1_loss,
                          dv_d_1_star = self.dv_d_1_star)
        self.number_of_engines()

        # Inertia calculator
        self.inertia_calculator()
        self.inertia_graphs()        

        # CoP functions
        self.cop_functions()

        # ACS sizing
        self.acs_sizing()

        # RCS sizing
        self.rcs_sizing()

    def load_raptor_constants(self):
        self.Isp_stage_1 = 350 # [s]
        self.Isp_stage_2 = 380 # [s]

        self.v_ex_stage_1 = self.Isp_stage_1 * 9.81 # [m/s]
        self.v_ex_stage_2 = self.Isp_stage_2 * 9.81 # [m/s]

        self.T_engine_stage_1 = 2745e3 # [N]
        self.T_engine_stage_2 = 2000e3 # [N]

        self.nozzle_exit_area = 1.326 # [m^2]
        self.nozzle_exit_pressure_stage_1 = 100000 # [Pa]
        self.nozzle_exit_pressure_stage_2 = 0 # [Pa]

    def stage_rocket(self,
                     dv_a_loss : np.array,
                     dv_d_1_loss : float,
                     dv_d_1_star : float):
        # Super heavy : https://en.wikipedia.org/wiki/SpaceX_Super_Heavy
        total_mass_super_heavy = 3675           # [t] : Total mass of the super heavy := mass propellant + mass dry
        propellant_mass_super_heavy = 3400      # [t] : Propellant mass of the super heavy
        structural_mass_super_heavy = total_mass_super_heavy - propellant_mass_super_heavy
        structural_coefficient_super_heavy = structural_mass_super_heavy / (propellant_mass_super_heavy + structural_mass_super_heavy)

        # Starship : https://en.wikipedia.org/wiki/SpaceX_Starship_(spacecraft)
        structural_mass_starship = 100           # [t] : Structural mass of the starship
        propellant_mass_starship = 1500          # [t] : Propellant mass of the starship
        structural_coefficient_starship = structural_mass_starship / (propellant_mass_starship + structural_mass_starship)

        stage_dict, trace = staging_p1_reproduction(
                                             m_pay = self.m_payload,
                                             dv_a_loss = dv_a_loss,
                                             dv_d_1_loss = dv_d_1_loss,
                                             dv_d_1_star = dv_d_1_star,
                                             v_ex = np.array([self.v_ex_stage_1, self.v_ex_stage_2]),
                                             eps = np.array([structural_coefficient_super_heavy, structural_coefficient_starship]),
                                             debug = False)
        
        self.m_initial = stage_dict['initial_mass']
        self.m_stage_1_ascent_burnout = stage_dict['mass_at_stage_1_ascent_burnout']
        self.m_prop_1 = stage_dict['propellant_mass_stage_1_ascent'] + stage_dict['propellant_mass_stage_1_descent']
        self.m_prop_2 = stage_dict['propellant_mass_stage_2_ascent']
        self.m_stage_1 = stage_dict['structural_mass_stage_1_ascent'] + stage_dict['propellant_mass_stage_1_ascent']
        self.m_stage_2 = stage_dict['structural_mass_stage_2_ascent'] + stage_dict['propellant_mass_stage_2_ascent']

        with open('data/rocket_parameters/sizing_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Variable', 'Units', 'Value'])
            writer.writerow(['Semi-major axis', 'm', self.semi_major_axis])
            writer.writerow(['delta v loss stage 1 (ascent)', 'm/s', dv_a_loss[0]])
            writer.writerow(['delta v loss stage 2 (ascent)', 'm/s', dv_a_loss[1]])
            writer.writerow(['delta v loss stage 1 (descent)', 'm/s', dv_d_1_loss])
            writer.writerow(['delta v star stage 1 (descent)', 'm/s', dv_d_1_star])
            writer.writerow(['Kappa', '-', trace['kappa']])
            writer.writerow(['Optimal loss-free payload ratio stage 1', '-', trace['payload_opt_1']])
            writer.writerow(['Optimal loss-free payload ratio stage 2', '-', trace['payload_opt_2']])
            writer.writerow(['Optimal loss payload ratio stage 1', '-', trace['payload_ratio_l_1_star']])
            writer.writerow(['Optimal loss payload ratio stage 2', '-', trace['payload_ratio_l_2_star']])
            writer.writerow(['Structural coefficient ascent stage 1', '-', trace['eps_a_1']])
            writer.writerow(['Structural coefficient descent stage 1', '-', trace['eps_d_1']])
            writer.writerow(['Structural coefficient ascent stage 1 (loss)', '-', trace['eps_a_1_l']])
            writer.writerow(['Structural coefficient descent stage 1 (loss)', '-', trace['eps_d_1_l']])
            writer.writerow(['Optimal loss-free velocity increment stage 1', 'm/s', trace['dv_star_a_1']])
            writer.writerow(['Optimal loss-free velocity increment stage 2', 'm/s', trace['dv_star_2']])
            writer.writerow(['Initial mass (trace)', 'ton', trace['m0']/1e3])
            writer.writerow(['Structural mass stage 1 (trace)', 'ton', trace['ms_1']/1e3])
            writer.writerow(['Structural mass stage 2 (trace)', 'ton', trace['ms_2']/1e3])
            writer.writerow(['Propellant mass stage 1 (trace)', 'ton', trace['mp_1']/1e3])
            writer.writerow(['Propellant mass stage 2 (trace)', 'ton', trace['mp_2']/1e3])
            writer.writerow(['Structural mass stage 1 (ascent)', 'ton', stage_dict['structural_mass_stage_1_ascent']/1e3])
            writer.writerow(['Structural mass stage 2 (ascent)', 'ton', stage_dict['structural_mass_stage_2_ascent']/1e3])
            writer.writerow(['Propellant mass stage 1 (ascent)', 'ton', stage_dict['propellant_mass_stage_1_ascent']/1e3])
            writer.writerow(['Propellant mass stage 2 (ascent)', 'ton', stage_dict['propellant_mass_stage_2_ascent']/1e3])
            writer.writerow(['Structural mass stage 1 (descent)', 'ton', stage_dict['structural_mass_stage_1_descent']/1e3])
            writer.writerow(['Structural mass stage 2 (descent)', 'ton', '-'])
            writer.writerow(['Propellant mass stage 1 (descent)', 'ton', stage_dict['propellant_mass_stage_1_descent']/1e3])
            writer.writerow(['Propellant mass stage 2 (descent)', 'ton', '-'])
            writer.writerow(['Actual structural mass stage 1', 'ton', stage_dict['structural_mass_stage_1_descent'] /1e3])
            writer.writerow(['Actual structural mass stage 2', 'ton', stage_dict['structural_mass_stage_2_ascent'] /1e3])
            writer.writerow(['Stage 1 Mass', 'ton', self.m_stage_1/1e3])
            writer.writerow(['Stage 2 Mass', 'ton', self.m_stage_2/1e3])
            writer.writerow(['Actual propellant mass stage 1', 'ton', self.m_prop_1/1e3])
            writer.writerow(['Actual propellant mass stage 2', 'ton', self.m_prop_2/1e3])
            writer.writerow(['Initial mass (subrocket 0)', 'ton', self.m_initial/1e3])
            writer.writerow(['Initial mass (subrocket 1)', 'ton', stage_dict['mass_of_rocket_at_stage_1_separation']/1e3])
            writer.writerow(['Ascent burnout mass (subrocket 0)', 'ton', self.m_stage_1_ascent_burnout/1e3])
            writer.writerow(['Ascent burnout mass (subrocket 1)', 'ton', stage_dict['mass_at_stage_2_ascent_burnout']/1e3])
            writer.writerow(['Mass at stage separation (subrocket 0)', 'ton', stage_dict['mass_of_stage_1_at_separation']/1e3])
            writer.writerow(['Mass at stage separation (subrocket 1)', 'ton', stage_dict['mass_of_rocket_at_stage_1_separation']/1e3])
            writer.writerow(['Payload mass', 'ton', self.m_payload/1e3])
            writer.writerow(['Exhaust velocity stage 1', 'm/s', self.v_ex_stage_1])
            writer.writerow(['Exhaust velocity stage 2', 'm/s', self.v_ex_stage_2])
            writer.writerow(['Thrust engine stage 1', 'N', self.T_engine_stage_1])
            writer.writerow(['Thrust engine stage 2', 'N', self.T_engine_stage_2])
            writer.writerow(['Nozzle exit area', 'm^2', self.nozzle_exit_area])
            writer.writerow(['Nozzle exit pressure stage 1', 'Pa', self.nozzle_exit_pressure_stage_1])
            writer.writerow(['Nozzle exit pressure stage 2', 'Pa', self.nozzle_exit_pressure_stage_2])

    def number_of_engines(self):
        self.TWR_super_heavy = 2.51
        self.TWR_starship = 0.76
        thrust_req_stage_1 = self.m_stage_1 * 9.81 * self.TWR_super_heavy
        thrust_req_stage_2 = self.m_stage_2 * 9.81 * self.TWR_starship
        #TWR_super_heavy_with_starship_and_payload = 1.702 # 150 kg payload
        #TWR_starship_with_payload = 0.4637 # 150 kg payload
        #thrust_req_stage_1 = (self.m_stage_1 + self.m_stage_2 + self.m_payload) * 9.81 * TWR_super_heavy_with_starship_and_payload
        #thrust_req_stage_2 = (self.m_stage_2 + self.m_payload) * 9.81 * TWR_starship_with_payload


        self.n_engine_stage_1 = math.ceil(thrust_req_stage_1 / self.T_engine_stage_1)
        T_max_stage_1 = self.T_engine_stage_1 * self.n_engine_stage_1 # [N]

        self.n_engine_stage_2 = math.ceil(thrust_req_stage_2 / self.T_engine_stage_2)
        T_max_stage_2 = self.T_engine_stage_2 * self.n_engine_stage_2 # [N]

        self.m_dot_stage_1 = T_max_stage_1 / self.v_ex_stage_1 # [kg/s]
        self.m_dot_stage_2 = T_max_stage_2 / self.v_ex_stage_2 # [kg/s]

        self.t_burn_stage_1 = self.m_prop_1 / self.m_dot_stage_1 # [s]
        self.t_burn_stage_2 = self.m_prop_2 / self.m_dot_stage_2 # [s]

        self.radius_rocket, self.S_rocket, number_of_engines_per_ring = new_radius_func(self.n_engine_stage_1)
        self.stage_1_n_gimballed = number_of_engines_per_ring[0] + number_of_engines_per_ring[1]

        self.number_of_engines_per_ring = number_of_engines_per_ring
        self.T_max_stage_1 = T_max_stage_1
        self.T_max_stage_2 = T_max_stage_2

        with open('data/rocket_parameters/sizing_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Number of engines stage 1', '', self.n_engine_stage_1])
            writer.writerow(['Number of engines stage 2', '', self.n_engine_stage_2])
            writer.writerow(['Number of engines per ring stage 1', '', self.number_of_engines_per_ring])
            writer.writerow(['Number of engines gimballed stage 1', '', self.stage_1_n_gimballed])
            writer.writerow(['Rocket Radius', 'm', self.radius_rocket])
            writer.writerow(['Rocket frontal area', 'm^2', self.S_rocket])
            writer.writerow(['Maximum thrust stage 1', 'MN', self.T_max_stage_1/1e6])
            writer.writerow(['Maximum thrust stage 2', 'MN', self.T_max_stage_2/1e6])
            writer.writerow(['Burn time stage 1', 's', self.t_burn_stage_1])
            writer.writerow(['Burn time stage 2', 's', self.t_burn_stage_2])
                
    def inertia_calculator(self):
        rocket_dimensions_instance = rocket_dimensions(
            rocket_radius=self.radius_rocket,
            propellant_masses=[self.m_prop_1, self.m_prop_2],
            structural_masses=[self.m_stage_1, self.m_stage_2],
            payload_mass=self.m_payload,
            number_of_engines=[self.n_engine_stage_1, self.n_engine_stage_2]
        )
        self.x_cog_inertia_subrocket_0_lambda, self.x_cog_inertia_subrocket_1_lambda, \
            self.lengths, \
            self.d_cg_thrusters_subrocket_0_lambda, self.d_cg_thrusters_subrocket_1_lambda, \
            self.x_cog_inertia_subrocket_2_lambda, self.d_cg_thrusters_subrocket_2_lambda, self.stage_1_height = rocket_dimensions_instance()
        
    def cop_functions(self):
        self.cop_subrocket_0_lambda = lambda alpha, M: cop_func(self.lengths[0], alpha, M, d_0 = 0.25) # Stage 1 ascent
        self.cop_subrocket_1_lambda = lambda alpha, M: cop_func(self.lengths[1], alpha, M, d_0 = 0.25) # Stage 2 ascent
        self.cop_subrocket_2_lambda = lambda alpha, M: cop_func(self.lengths[2], alpha, M, d_0 = 0.75) # Stage 1 descent
        plot_cop_func()

    def inertia_graphs(self):
        fuel_consumption_percentages = np.linspace(0.01, 0.99, 100)

        x_cog_subrocket_0 = []
        x_cog_subrocket_1 = []
        x_cog_subrocket_2 = []
        inertia_subrocket_0 = []
        inertia_subrocket_1 = []
        inertia_subrocket_2 = []
        d_cg_thrusters_subrocket_0 = []
        d_cg_thrusters_subrocket_1 = []
        d_cg_thrusters_subrocket_2 = []
        for fuel_consumption_percentage in fuel_consumption_percentages:
            fill_level = 1 - fuel_consumption_percentage
            x_0, i_0 = self.x_cog_inertia_subrocket_0_lambda(fill_level)
            x_1, i_1 = self.x_cog_inertia_subrocket_1_lambda(fill_level)
            x_2, i_2 = self.x_cog_inertia_subrocket_2_lambda(fill_level)
            d_cg_0 = self.d_cg_thrusters_subrocket_0_lambda(x_0)
            d_cg_1 = self.d_cg_thrusters_subrocket_1_lambda(x_1)
            d_cg_2 = self.d_cg_thrusters_subrocket_2_lambda(x_2)
            x_cog_subrocket_0.append(x_0)
            x_cog_subrocket_1.append(x_1)
            x_cog_subrocket_2.append(x_2)
            inertia_subrocket_0.append(i_0)
            inertia_subrocket_1.append(i_1)
            inertia_subrocket_2.append(i_2)
            d_cg_thrusters_subrocket_0.append(d_cg_0)
            d_cg_thrusters_subrocket_1.append(d_cg_1)
            d_cg_thrusters_subrocket_2.append(d_cg_2)
        plt.figure(figsize=(15, 8))
        plt.suptitle('Center of Gravity of sub-rockets', fontsize=24, y=0.98)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(fuel_consumption_percentages*100, x_cog_subrocket_0, color='blue', linewidth=4)
        ax1.set_xlabel('Fuel consumption [%]', fontsize=20)
        ax1.set_ylabel('COG [m]', fontsize=20)
        ax1.set_title('Full launch vehicle', fontsize=22, pad=15)
        ax1.tick_params(labelsize=16)
        ax1.grid(True)
        ax2 = plt.subplot(gs[1])
        ax2.plot(fuel_consumption_percentages*100, x_cog_subrocket_1, color='blue', linewidth=4)
        ax2.set_xlabel('Fuel consumption [%]', fontsize=20)
        ax2.set_title('Stage 1 separated', fontsize=22, pad=15)
        ax2.tick_params(labelsize=16)
        ax2.grid(True)
        ax3 = plt.subplot(gs[2])
        ax3.plot(fuel_consumption_percentages*100, x_cog_subrocket_2, color='blue', linewidth=4)
        ax3.set_xlabel('Fuel consumption [%]', fontsize=20)
        ax3.set_title('Stage 2 separated', fontsize=22, pad=15)
        ax3.tick_params(labelsize=16)
        ax3.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('results/Sizing/center_of_gravity_graphs.png', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 8))
        plt.suptitle('Inertia of sub-rockets', fontsize=24, y=0.98)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(fuel_consumption_percentages*100, inertia_subrocket_0, color='blue', linewidth=4)
        ax1.set_xlabel('Fuel consumption [%]', fontsize=20)
        ax1.set_ylabel(r'Inertia [$kg \cdot m^2$]', fontsize=20)
        ax1.set_title('Full launch vehicle', fontsize=22, pad=15)
        ax1.tick_params(labelsize=16)
        ax1.grid(True)
        ax2 = plt.subplot(gs[1])
        ax2.plot(fuel_consumption_percentages*100, inertia_subrocket_1, color='blue', linewidth=4)
        ax2.set_xlabel('Fuel consumption [%]', fontsize=20)
        ax2.set_title('Stage 1 separated', fontsize=22, pad=15)
        ax2.tick_params(labelsize=16)
        ax2.grid(True)
        ax3 = plt.subplot(gs[2])
        ax3.plot(fuel_consumption_percentages*100, inertia_subrocket_2, color='blue', linewidth=4)
        ax3.set_xlabel('Fuel consumption [%]', fontsize=20)
        ax3.set_title('Stage 2 separated', fontsize=22, pad=15)
        ax3.tick_params(labelsize=16)
        ax3.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('results/Sizing/inertia_graphs.png', bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(15, 8))
        plt.suptitle('Thruster moment arm', fontsize=24, y=0.98)
        gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(fuel_consumption_percentages*100, d_cg_thrusters_subrocket_0, color='blue', linewidth=4)
        ax1.set_xlabel('Fuel consumption percentage [%]', fontsize=20)
        ax1.set_ylabel('Moment arm [m]', fontsize=20)
        ax1.set_title('Full launch vehicle', fontsize=22, pad=15)
        ax1.tick_params(labelsize=16)
        ax1.grid(True)
        ax2 = plt.subplot(gs[1])
        ax2.plot(fuel_consumption_percentages*100, d_cg_thrusters_subrocket_1, color='blue', linewidth=4)
        ax2.set_xlabel('Fuel consumption percentage [%]', fontsize=20)
        ax2.set_ylabel('Moment arm [m]', fontsize=20)
        ax2.set_title('Stage 2 separated', fontsize=22, pad=15)
        ax2.tick_params(labelsize=16)
        ax2.grid(True)
        ax3 = plt.subplot(gs[2])
        ax3.plot(fuel_consumption_percentages*100, d_cg_thrusters_subrocket_2, color='blue', linewidth=4)
        ax3.set_xlabel('Fuel consumption percentage [%]', fontsize=20)
        ax3.set_ylabel('Moment arm [m]', fontsize=20)
        ax3.set_title('Stage 1 separated', fontsize=22, pad=15)
        ax3.tick_params(labelsize=16)
        ax3.grid(True)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('results/Sizing/thruster_moment_arm_graphs.png', bbox_inches='tight')
        plt.close()

    def pickle_dump_funcs(self):
        # Pickle dump these functions:
        with open('data/rocket_parameters/rocket_functions.pkl', 'wb') as f:
            dill.dump({
                'x_cog_inertia_subrocket_0_lambda': self.x_cog_inertia_subrocket_0_lambda, # Stage 1 + Stage 2 + Payload (nose)
                'x_cog_inertia_subrocket_1_lambda': self.x_cog_inertia_subrocket_1_lambda, # Stage 2 + Payload (nose)
                'x_cog_inertia_subrocket_2_lambda': self.x_cog_inertia_subrocket_2_lambda, # Stage 1

                'd_cg_thrusters_subrocket_0_lambda': self.d_cg_thrusters_subrocket_0_lambda, # Stage 1 + Stage 2 + Payload (nose)
                'd_cg_thrusters_subrocket_1_lambda': self.d_cg_thrusters_subrocket_1_lambda, # Stage 2 + Payload (nose)
                'd_cg_thrusters_subrocket_2_lambda': self.d_cg_thrusters_subrocket_2_lambda, # Stage 1

                'cop_subrocket_0_lambda': self.cop_subrocket_0_lambda, # Stage 1 + Stage 2 + Payload (nose)
                'cop_subrocket_1_lambda': self.cop_subrocket_1_lambda, # Stage 2 + Payload (nose)
                'cop_subrocket_2_lambda': self.cop_subrocket_2_lambda  # Stage 1
                
            }, f)

    def acs_sizing(self):
        # Stage 1 only
        S_grid_fin = 2.5*2 # [m^2]
        d_base_grid_fin = self.stage_1_height - 1 #[m] i.e. 1m from top of stage 1 to bottom of rocket

        with open('data/rocket_parameters/sizing_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['S_grid_fins', 'm^2', S_grid_fin])
            writer.writerow(['d_base_grid_fin', 'm', d_base_grid_fin])

    def rcs_sizing(self):
        max_RCS_force_per_thruster = 5000 # [N]
        max_RCS_force_top = 2 * max_RCS_force_per_thruster # [N]
        max_RCS_force_bottom = 2 * max_RCS_force_per_thruster # [N]
        d_base_rcs_bottom = 1 # [m]
        d_base_rcs_top = self.stage_1_height - 1 # [m]

        with open('data/rocket_parameters/sizing_results.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['max_RCS_force_per_thruster', 'N', max_RCS_force_per_thruster])
            writer.writerow(['max_RCS_force_top', 'N', max_RCS_force_top])
            writer.writerow(['max_RCS_force_bottom', 'N', max_RCS_force_bottom])
            writer.writerow(['d_base_rcs_bottom', 'm', d_base_rcs_bottom])
            writer.writerow(['d_base_rcs_top', 'm', d_base_rcs_top])

def size_rocket(dv_a_1_loss : float = 1391.0,
                dv_a_2_loss : float = 710.0,
                dv_d_1_loss : float = 1401.0,
                dv_d_1_star : float = 1709):

    rocket_config = create_rocket_configuration(dv_a_1_loss = dv_a_1_loss,
                                                dv_a_2_loss = dv_a_2_loss,
                                                dv_d_1_loss = dv_d_1_loss,
                                                dv_d_1_star = dv_d_1_star)
    rocket_config.pickle_dump_funcs()  # Call the pickle dump function

if __name__ == '__main__':
    size_rocket()