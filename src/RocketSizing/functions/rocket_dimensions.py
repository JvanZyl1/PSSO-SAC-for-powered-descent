import csv
import math

rho_LOX = 1200 # [kg/m^3]
rho_LCH4 = 450 # [kg/m^3]
rho_304L = 8000 # [kg/m^3]

#### 1) TANK SIZING ####
def cylindrical_tank_dimensions(mass: float,
                                density: float,
                                radius: float):
    volume = mass / density
    height = volume / (math.pi * radius**2)
    return height, volume

def fuel_to_oxidiser_mass_calculator(propellant_mass: float,
                                     oxidiser_to_fuel_ratio: float) -> float:
    fuel_mass = propellant_mass / (1 + oxidiser_to_fuel_ratio)
    oxidiser_mass = propellant_mass - fuel_mass
    return oxidiser_mass, fuel_mass

def tank_sizing_constant_radius(propellant_mass: float,
                                density_oxidiser : float,
                                density_fuel : float,
                                oxidiser_to_fuel_ratio : float,
                                tank_radius : float):    
    oxidiser_mass, fuel_mass = fuel_to_oxidiser_mass_calculator(propellant_mass, oxidiser_to_fuel_ratio)
    fuel_tank_height, _ = cylindrical_tank_dimensions(fuel_mass, density_fuel, tank_radius)
    oxidiser_tank_height, _ = cylindrical_tank_dimensions(oxidiser_mass, density_oxidiser, tank_radius)
    return oxidiser_tank_height, fuel_tank_height, oxidiser_mass, fuel_mass

def first_stage_dry_x_cog(rocket_radius : float,
                          m_s : float,
                           m_prop : float,
                           oxidiser_to_fuel_ratio : float,
                           wall_thickness : float,
                           n_e : int,
                           m_e_integrated : float,
                           Lambda_ul : float,
                           rho_sections : float,
                           engine_height : float):
    tank_radius = rocket_radius - wall_thickness
    h_ox, h_f, m_ox, m_f = tank_sizing_constant_radius(m_prop,
                                                       rho_LOX,
                                                       rho_LCH4,
                                                       oxidiser_to_fuel_ratio,
                                                       tank_radius)
    
    m_s_tanks = math.pi * (h_ox + h_f) * (rocket_radius**2 - (rocket_radius - wall_thickness)**2) * rho_304L
    m_e_stage = n_e * m_e_integrated
    m_upper = (m_s - m_s_tanks - m_e_stage) * 1/(1 + Lambda_ul)
    m_lower = (m_s - m_s_tanks - m_e_stage) * Lambda_ul/(1 + Lambda_ul)

    h_upper = m_upper/rho_sections * 1 / (math.pi * rocket_radius**2)
    h_lower = m_lower/rho_sections * 1 / (math.pi * rocket_radius**2)

    x_dry = (-m_e_stage * (engine_height/2)
             + m_lower * (h_lower/2)
             + m_s_tanks * (h_lower + (h_ox + h_f)/2)
             + m_upper * (h_lower + h_ox + h_f + h_upper/2)
             ) / m_s
    assert m_s - m_s_tanks - m_e_stage - m_upper - m_lower == 0 # Verification
    
    I_e_stage = 1/12 * m_e_stage * engine_height**2 - \
        m_e_stage * (x_dry + engine_height/2)**2
    I_lower = 1/12 * m_lower * h_lower**2  + \
        m_lower * (h_lower/2 - x_dry)**2
    I_upper = 1/12 * m_upper * h_upper**2 + \
        m_upper * (h_lower + h_ox + h_f + h_upper/2 - x_dry)**2
    I_s_tanks = 1/12 * m_s_tanks * (h_f + h_ox)**2 + \
        m_s_tanks * (h_lower + (h_f + h_ox)/2 - x_dry)**2
    I_dry = I_e_stage + I_lower + I_s_tanks + I_upper
    return (x_dry, I_dry,
            h_ox, h_f, m_ox, m_f,
            h_lower, h_upper)

def second_stage_with_payload_dry_x_cog(rocket_radius : float,
                          m_s : float,
                           m_prop : float,
                           oxidiser_to_fuel_ratio : float,
                           wall_thickness : float,
                           n_e : int,
                           m_e_integrated : float,
                           Lambda_ul : float,
                           rho_sections : float,
                           engine_height : float,
                           rho_pay : float,
                           rho_nose : float,
                           m_pay : float,
                           t_fairing : float):

    tank_radius = rocket_radius - wall_thickness
    h_ox, h_f, m_ox, m_f = tank_sizing_constant_radius(m_prop,
                                                       rho_LOX,
                                                       rho_LCH4,
                                                       oxidiser_to_fuel_ratio,
                                                       tank_radius)
    
    m_s_tanks = math.pi * (h_ox + h_f) * (rocket_radius**2 - (rocket_radius - wall_thickness)**2) * rho_304L
    m_e_stage = n_e * m_e_integrated

    h_pay = m_pay/rho_pay * 1/(math.pi * (rocket_radius - t_fairing)**2)
    m_s_pay = math.pi * h_pay *  (rocket_radius**2 - (rocket_radius - t_fairing)**2) * rho_304L

    m_s_upper_most = t_fairing * math.pi * rocket_radius**2 * rho_304L
    m_upper = (m_s - m_s_tanks - m_e_stage - m_s_pay - m_s_upper_most) * 1/(1 + Lambda_ul)
    m_lower = (m_s - m_s_tanks - m_e_stage - m_s_pay - m_s_upper_most) * Lambda_ul/(1 + Lambda_ul)

    h_upper = m_upper/rho_sections * 1 / (math.pi * rocket_radius**2)
    h_lower = m_lower/rho_sections * 1 / (math.pi * rocket_radius**2)

    x_dry = (-m_e_stage * (engine_height/2)
             + m_lower * (h_lower/2)
             + m_s_tanks * (h_lower + (h_ox + h_f)/2)
             + m_upper * (h_lower + h_ox + h_f + h_upper/2)
             + (m_pay + m_s_pay) * (h_lower + h_ox + h_f + h_upper + h_pay/2)
             + m_s_upper_most * (h_lower + h_ox + h_f + h_upper + h_pay + t_fairing/2)
    ) / (m_s + m_pay)

    I_e_stage = 1/12 * m_e_stage * engine_height**2 - \
        m_e_stage * (x_dry + engine_height/2)**2
    I_lower = 1/12 * m_lower * h_lower**2  + \
        m_lower * (h_lower/2 - x_dry)**2
    I_upper = 1/12 * m_upper * h_upper**2 + \
        m_upper * (h_lower + h_upper/2 - x_dry)**2
    I_s_tanks = 1/12 * m_s_tanks * (h_f + h_ox)**2 + \
        m_s_tanks * (h_lower + (h_f + h_ox)/2 - x_dry)**2
    I_pay = 1/12 * (m_pay + m_s_pay) * h_pay**2 + \
        (m_pay + m_s_pay) * (h_lower + h_f + h_ox + h_upper + h_pay/2 - x_dry)**2
    I_uppermost = 1/12 * (m_s_upper_most) * t_fairing**2 + \
        (m_s_upper_most) * (h_lower + h_f + h_ox + h_upper + h_pay + t_fairing/2 - x_dry)**2
    I_dry = I_e_stage + I_lower + I_s_tanks + I_upper + I_pay + I_uppermost # around x_dry

    x_prop_initial = (
        m_ox * (h_lower + h_ox/2) + m_f * (h_lower + h_ox + h_f/2)
    ) / m_prop

    x_wet_initial = (
        x_dry * (m_pay + m_s) + x_prop_initial * m_prop
    ) / (m_pay + m_s + m_prop)

    # Round tank axis
    I_ox_initial = 1/12 * m_ox * h_ox**2 + \
        (h_lower + h_ox/2 - x_prop_initial)**2 # around x_prop_initial
    I_f_initial = 1/12 * m_f * h_f**2 + \
        (h_lower + h_ox + h_f/2 - x_prop_initial)**2 # around x_prop_initial
    
    # Around the stage axis
    I_prop_initial_stage = I_ox_initial + I_f_initial + m_prop * (x_prop_initial - x_wet_initial)**2
    I_dry_stage = I_dry + (m_s + m_pay) * (x_dry - x_wet_initial)**2
    I_initial_stage = I_prop_initial_stage + I_dry_stage
    #           -, around x_dry, around x_wet_initial
    m_dry = m_pay + m_s
    return (x_dry, I_dry, I_initial_stage, x_wet_initial,
            h_ox, h_f, m_ox, m_f, h_lower, m_dry, h_upper, h_pay, t_fairing)


def stage_inertia(h_ox : float, # initial
                  h_f : float,  # initial
                  m_ox : float,  # initial
                  m_f : float,  # initial
                  h_lower : float,
                  m_dry : float,
                  x_dry : float,
                  I_dry : float # in x_dry_fram
                  ):
    def func(fill_level):
        h_ox_tilde = h_ox * fill_level
        h_f_tilde = h_f * fill_level
        m_ox_tilde = m_ox * fill_level
        m_f_tilde = m_f * fill_level

        x_prop_tilde = (
            m_ox_tilde * (h_lower + h_ox_tilde/2) \
            + m_f_tilde * (h_lower + h_ox + h_f_tilde/2)
        ) / (m_ox_tilde + m_f_tilde)

        # Around x_prop_tilde
        I_ox_tilde = 1/12 * m_ox_tilde * h_ox_tilde**2 + \
            m_ox_tilde * (h_lower + h_ox_tilde/2 - x_prop_tilde)**2
        I_f_tilde = 1/12 * m_f_tilde * h_f_tilde**2 + \
            m_f_tilde * (h_lower + h_ox + h_f_tilde/2 - x_prop_tilde)**2
        I_prop_tilde = I_ox_tilde + I_f_tilde

        x_wet_tilde = (
            m_dry * x_dry + (m_ox_tilde + m_f_tilde) * x_prop_tilde
        ) / (m_dry + m_ox_tilde + m_f_tilde)
        
        # Around x_wet_tilde (stage X wet)
        I_dry_hat = I_dry + m_dry * (x_dry - x_wet_tilde)**2
        I_prop_hat = I_prop_tilde + (m_ox_tilde + m_f_tilde) * (x_prop_tilde - x_wet_tilde)**2

        # Around x_cog (stage X wet)
        x_cog = x_wet_tilde
        inertia = I_dry_hat + I_prop_hat
        return x_cog, inertia
    return func

def full_rocket_inertia(m_s_1 : float,
                        x_dry_1 : float,
                        I_dry_1 : float,
                        m_2 : float,
                        m_pay : float,
                        x_wet_2_initial : float,
                        I_wet_2_initial : float,
                        h_1 : float,
                        h_1_ox : float,
                        h_1_f : float,
                        m_1_ox : float,
                        m_1_f : float,
                        h_lower_1 : float):
    def func(fill_level):
        h_ox_1_tilde = h_1_ox * fill_level
        h_f_1_tilde = h_1_f * fill_level
        m_ox_1_tilde = m_1_ox * fill_level
        m_f_1_tilde = m_1_f * fill_level
        m_prop_1_tilde = m_ox_1_tilde + m_f_1_tilde

        x_prop_1_tilde = (
            m_ox_1_tilde * (h_lower_1 + h_ox_1_tilde/2) \
            + m_1_f * (h_lower_1 + h_ox_1_tilde + h_f_1_tilde/2)
        ) / (m_ox_1_tilde + m_f_1_tilde)

        # Around x_prop_1_tilde
        I_ox_1_tilde = 1/12 * m_ox_1_tilde * h_ox_1_tilde**2 + \
            m_ox_1_tilde * (h_lower_1 + h_ox_1_tilde/2 - x_prop_1_tilde)**2
        I_f_1_tilde = 1/12 * m_f_1_tilde * h_f_1_tilde**2 + \
            m_f_1_tilde * (h_lower_1 + h_ox_1_tilde + h_f_1_tilde/2 - x_prop_1_tilde)**2
        I_prop_1_tilde = I_ox_1_tilde + I_f_1_tilde
        
        x_rocket_tilde = (
            m_s_1 * x_dry_1
            + (m_2 + m_pay) * (x_wet_2_initial + h_1)
            + m_prop_1_tilde * x_prop_1_tilde
        ) / (m_s_1 + m_2 + m_pay + m_prop_1_tilde)

        I_rocket_hat = I_dry_1 + m_s_1 * (x_dry_1 - x_rocket_tilde)**2 \
            + I_wet_2_initial + m_2 * (x_wet_2_initial - x_rocket_tilde)**2 \
                + I_prop_1_tilde + m_prop_1_tilde * (x_prop_1_tilde - x_rocket_tilde)**2
        return x_rocket_tilde, I_rocket_hat
    return func

def d_cg_thrusters(x_cog,
                   engine_height):
    # Thrusters are at the bottom of the rocket
    d_cg_thrusters = x_cog + engine_height
    return d_cg_thrusters


class rocket_dimensions:
    def __init__(self,
                 rocket_radius: float,
                 propellant_masses: list,
                 structural_masses: list,
                 payload_mass: float,
                 number_of_engines: list):
        self.wall_thickness_tanks = 0.01
        self.rocket_radius = rocket_radius
        self.propellant_mass_stage_1 = propellant_masses[0]
        self.propellant_mass_stage_2 = propellant_masses[1]
        self.structural_mass_stage_1 = structural_masses[0]
        self.structural_mass_stage_2 = structural_masses[1]
        self.payload_mass = payload_mass
        self.number_of_engines_stage_1 = number_of_engines[0]
        self.number_of_engines_stage_2 = number_of_engines[1]

        self.engine_integrated_mass = 1720                #[kg]
        self.engine_height = 3.1                            #[m]
        self.payload_fairing_thickness = 0.1                #[m]
        self.payload_density = 2000                         #[kg/m^3]
        self.nose_density = 2000                         #[kg/m^3] :
        self.oxidiser_to_fuel_ratio = 3.545
        self.Lambda_ul_1 = 1
        self.Lambda_ul_2 = 1
        self.rho_sections = 8000

        self.save_path = 'data/rocket_parameters/sizing_results.csv'

        with open(self.save_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Wall thickness tanks ', 'mm', self.wall_thickness_tanks*1000])
            writer.writerow(['Payload fairing thickness ', 'cm', self.payload_fairing_thickness*100])
            writer.writerow(['Payload density ', 'kg/m^3', self.payload_density])


    def size_first_stage(self):
        x_dry, I_dry, h_ox, h_f, m_ox, m_f, h_lower, h_upper = first_stage_dry_x_cog(rocket_radius= self.rocket_radius,
                                                                                     m_s = self.structural_mass_stage_1,
                                                                                     m_prop = self.propellant_mass_stage_1,
                                                                                     oxidiser_to_fuel_ratio = self.oxidiser_to_fuel_ratio,
                                                                                     wall_thickness = self.wall_thickness_tanks,
                                                                                     n_e = self.number_of_engines_stage_1,
                                                                                     m_e_integrated = self.engine_integrated_mass,
                                                                                     Lambda_ul = self.Lambda_ul_1,
                                                                                     rho_sections= self.rho_sections,
                                                                                     engine_height = self.engine_height)
        
        with open(self.save_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Stage 1 upper section height ', 'm', h_upper])
            writer.writerow(['Stage 1 lower section height ', 'm', h_lower])
            writer.writerow(['Stage 1 height ', 'm', h_lower + h_f + h_ox + h_upper])
            writer.writerow(['Oxidiser tank height stage 1 ', 'm', h_ox])
            writer.writerow(['Fuel tank height stage 1 ', 'm', h_f])
            writer.writerow(['Oxidiser mass stage 1 ', 'kg', m_ox])
            writer.writerow(['Fuel mass stage 1 ', 'kg', m_f])
                            
        stage_1_inertia_cog_func = stage_inertia(h_ox = h_ox,
                                                 h_f = h_f,
                                                 m_ox = m_ox,
                                                 m_f = m_f,
                                                 h_lower = h_lower,
                                                 m_dry = self.structural_mass_stage_1,
                                                 x_dry = x_dry,
                                                 I_dry = I_dry)
        
        h_1 = h_lower + h_ox + h_f + h_upper
        return stage_1_inertia_cog_func, x_dry, I_dry, h_1, h_ox, h_f, m_ox,m_f, h_lower
    
    def size_second_stage(self):
        x_dry, I_dry, I_initial_stage, x_wet_initial, h_ox, h_f, m_ox, m_f, h_lower, m_dry, h_upper, h_pay, h_uppermost \
            = second_stage_with_payload_dry_x_cog(rocket_radius = self.rocket_radius,
                                                  m_s = self.structural_mass_stage_2,
                                                  m_prop = self.propellant_mass_stage_2,
                                                  oxidiser_to_fuel_ratio=self.oxidiser_to_fuel_ratio,
                                                  wall_thickness=self.wall_thickness_tanks,
                                                  n_e = self.number_of_engines_stage_2,
                                                  m_e_integrated = self.engine_integrated_mass,
                                                  Lambda_ul=self.Lambda_ul_2,
                                                  rho_sections=self.rho_sections,
                                                  engine_height=self.engine_height,
                                                  rho_pay=self.payload_density,
                                                  rho_nose = self.nose_density,
                                                  m_pay=self.payload_mass,
                                                  t_fairing=self.payload_fairing_thickness)
        
        with open(self.save_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Stage 2 upper section height ', 'm', h_upper])
            writer.writerow(['Stage 2 lower section height ', 'm', h_lower])
            writer.writerow(['Stage 2 payload height ', 'm', h_pay])
            writer.writerow(['Stage 2 uppermost section height ', 'm', h_uppermost])
            writer.writerow(['Stage 2 height ', 'm', h_lower + h_f + h_ox + h_upper + h_pay + h_uppermost])
            writer.writerow(['Oxidiser tank height stage 2 ', 'm', h_ox])
            writer.writerow(['Fuel tank height stage 2 ', 'm', h_f])
            writer.writerow(['Oxidiser mass stage 2 ', 'kg', m_ox])
            writer.writerow(['Fuel mass stage 2 ', 'kg', m_f])
        
        stage_2_inertia_cog_func = stage_inertia(h_ox = h_ox,
                                                 h_f = h_f,
                                                 m_ox = m_ox,
                                                 m_f = m_f,
                                                 h_lower = h_lower,
                                                 m_dry = m_dry, # inc. payload
                                                 x_dry = x_dry,
                                                 I_dry = I_dry)
        
        m_2 = m_dry + m_ox + m_f
        h_2 = h_lower + h_ox + h_f + h_upper + h_pay + h_uppermost
        
        return stage_2_inertia_cog_func, I_initial_stage,  x_wet_initial, m_2, h_2

    def __call__(self):
        stage_1_inertia_cog_func, x_dry_1, I_dry_1, h_1, h_ox_1, h_f_1, m_ox_1, m_f_1, h_lower_1 \
            = self.size_first_stage()
        stage_2_inertia_cog_func, I_wet_2_initial,  x_wet_2_initial, m_2, h_2 \
            = self.size_second_stage()
        
        full_rocket_inertia_cog_func = full_rocket_inertia(m_s_1 = self.structural_mass_stage_1,
                                                           x_dry_1 = x_dry_1 ,
                                                           I_dry_1 = I_dry_1,
                                                           m_2 = m_2,
                                                           m_pay = self.payload_mass,
                                                           x_wet_2_initial = x_wet_2_initial,
                                                           I_wet_2_initial = I_wet_2_initial,
                                                           h_1 = h_1,
                                                           h_1_ox = h_ox_1,
                                                           h_1_f = h_f_1,
                                                           m_1_ox = m_ox_1,
                                                           m_1_f = m_f_1,
                                                           h_lower_1 = h_lower_1)        
        lengths = [
            h_1 + h_2,
            h_2,
            h_1
        ]      

        d_cg_thrusters_full_rocket = lambda x_cog : d_cg_thrusters(x_cog, self.engine_height)
        d_cg_thrusters_stage_2 = lambda x_cog : d_cg_thrusters(x_cog, self.engine_height)
        d_cg_thrusters_stage_1 = lambda x_cog : d_cg_thrusters(x_cog, self.engine_height)
        return (full_rocket_inertia_cog_func, stage_1_inertia_cog_func, lengths, \
                d_cg_thrusters_full_rocket, d_cg_thrusters_stage_2, \
                    stage_2_inertia_cog_func, d_cg_thrusters_stage_1, h_1)

