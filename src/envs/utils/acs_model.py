import math
from src.envs.utils.grid_fin_aerodynamics import compile_grid_fin_Ca, compile_grid_fin_Cn


def first_order_low_pass_step(x, u, tau, dt):
    dx = (-x + u) / tau
    y = x + dt * dx
    return y

Ca_func = compile_grid_fin_Ca() # Mach
Cn_func = compile_grid_fin_Cn() # Mach, alpha [rad]

def ACS(alpha_effective_rad : float,
        pitch_angle : float,
        dynamic_pressure : float,
        mach_number : float,
        x_cog : float,
        deflection_command_left_deg : float,
        deflection_command_right_deg : float,
        delta_command_left_rad_prev : float,
        delta_command_right_rad_prev : float,
        dt : float,
        grid_fin_area : float,
        d_base_grid_fin : float,
        rocket_radius : float):
    # right up is positive, left down is positive
    max_deflection_angle_deg = 60
    delta_command_left_rad = math.radians(deflection_command_left_deg * max_deflection_angle_deg)
    delta_command_right_rad = math.radians(deflection_command_right_deg * max_deflection_angle_deg)

    delta_left_rad = first_order_low_pass_step(x = delta_command_left_rad_prev,
                                          u = delta_command_left_rad,
                                          tau = 0.5,
                                          dt = dt)
    delta_right_rad = first_order_low_pass_step(x = delta_command_right_rad_prev,
                                          u = delta_command_right_rad,
                                          tau = 0.5,
                                          dt = dt)
    alpha_local_left = alpha_effective_rad - delta_left_rad
    alpha_local_right = alpha_effective_rad - delta_right_rad

    qS = dynamic_pressure * grid_fin_area

    Ca = Ca_func(mach_number)
    Cn_L = Cn_func(mach_number, alpha_local_left)
    Cn_R = Cn_func(mach_number, alpha_local_right)

    # F_para = F_a (fixed) + F_a * (cos(delta_L) + cos(delta_R)) + F_n_L * sin(delta_L) + F_n_R * sin(delta_R)
    gf_force_perpendicular = qS * (Cn_R * math.cos(delta_right_rad) - Cn_L * math.cos(delta_left_rad) -
                                    Ca * (math.sin(delta_left_rad) - math.sin(delta_right_rad)))
    gf_force_parallel = qS * (Ca * (2 + math.cos(delta_left_rad) + math.cos(delta_right_rad)) -
                                Cn_L * math.sin(delta_left_rad) + Cn_R * math.sin(delta_right_rad))
    
    gf_moment_z = -(d_base_grid_fin - x_cog) * gf_force_perpendicular +\
                rocket_radius * qS * (
                    Ca * (math.sin(delta_right_rad) - math.sin(delta_left_rad))
                    - Cn_L * math.cos(delta_left_rad)
                    + Cn_R * math.cos(delta_right_rad)
                )
    gf_force_x = gf_force_parallel * math.cos(pitch_angle) + gf_force_perpendicular * math.sin(pitch_angle)
    gf_force_y = gf_force_parallel * math.sin(pitch_angle) - gf_force_perpendicular * math.cos(pitch_angle)
    acs_info = {
        'alpha_local_left_rad' : alpha_local_left,
        'alpha_local_right_rad' : alpha_local_right,
        'C_n_L' : Cn_L,
        'C_a_L' : Ca,
        'C_n_R' : Cn_R,
        'C_a_R' : Ca,
        'F_n_L' : Cn_L * qS,
        'F_a_L' : Ca * qS,
        'F_n_R' : Cn_R * qS,
        'F_a_R' : Ca * qS,
        'F_perpendicular_L' : qS * (Cn_L * math.cos(delta_left_rad) - Ca * math.sin(delta_left_rad)),
        'F_perpendicular_R' : qS * (Cn_R * math.cos(delta_right_rad) - Ca * math.sin(delta_right_rad)),
        'F_perpendicular' : gf_force_perpendicular,
        'F_parallel_L' : qS * (Ca * math.cos(delta_left_rad) + Cn_L * math.sin(delta_left_rad)),
        'F_parallel_R' : qS * (Ca * math.cos(delta_right_rad) + Cn_R * math.sin(delta_right_rad)),
        'F_parallel' : gf_force_parallel,
        'Fx' : gf_force_x,
        'Fy' : gf_force_y,
        'Mz' : gf_moment_z,
        'd_fin_cg' : d_base_grid_fin - x_cog,
        'delta_left_rad' : delta_left_rad,
        'delta_right_rad' : delta_right_rad        
    }

    return gf_force_perpendicular, gf_force_parallel, gf_moment_z, delta_command_left_rad, delta_command_right_rad, acs_info