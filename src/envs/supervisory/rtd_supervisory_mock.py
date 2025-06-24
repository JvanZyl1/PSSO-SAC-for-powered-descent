import math
import pandas as pd

from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model    

def compile_rtd_supervisory_test(flight_phase = 'subsonic'):
    assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']
    if flight_phase == 'subsonic':
        terminal_mach = 1.1
    elif flight_phase == 'supersonic':
        reference_data = pd.read_csv(f'data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv')
        terminal_altitude = reference_data['y[m]'].iloc[-1]
    
    flip_over_boostbackburn_terminal_vx = -60
    dynamic_pressure_threshold = 65000

    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)
        if flight_phase in ['subsonic']:
            mach_number = speed / speed_of_sound
            if mach_number > terminal_mach:
                return True
            else:
                return False
        elif flight_phase == 'supersonic':
            if y > terminal_altitude:
                return True
            else:
                return False
        elif flight_phase == 'flip_over_boostbackburn':
            if vx < flip_over_boostbackburn_terminal_vx:
                return True
            else:
                return False
        elif flight_phase == 'ballistic_arc_descent':
            if dynamic_pressure > dynamic_pressure_threshold and \
                abs_alpha_effective < math.radians(3):
                return True
            else:
                return False
        elif flight_phase in ['landing_burn', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
            if y < 1:
                return True
            else:
                return False
    def truncated_func_lambda(state, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)

        if flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn']:
            if mass_propellant <= 0:
                return True, 1
            else:
                return False, 0
        elif flight_phase == 'ballistic_arc_descent':
            if dynamic_pressure > 35000 and \
                abs_alpha_effective > math.radians(3):
                return True, 1
            else:
                return False, 0
        elif flight_phase in ['landing_burn', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
            air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
            xp, yp, vxp, vyp, thetap, theta_dotp, gammamp, alphap, massp, mass_propellantp, timep = previous_state
            speed = math.sqrt(vx**2 + vy**2)
            dynamic_pressure = 0.5 * air_density * speed**2
            speed_p = math.sqrt(vxp**2 + vyp**2)
            speed_diff = abs(speed - speed_p)
            dt = time - timep
            if vy < 0:
                alpha_effective = abs(gamma - theta - math.pi)
            else:
                alpha_effective = abs(theta - gamma)
            if y < -10:
                return True, 1
            elif mass_propellant <= 0:
                print(f'Truncated due to mass_propellant <= 0, mass_propellant = {mass_propellant}, y = {y}')
                return True, 2
            elif theta > math.pi + math.radians(2):
                print(f'Truncated due to theta > math.pi + math.radians(2), theta = {theta}, y = {y}')
                return True, 3
            elif dynamic_pressure > 65000:
                print(f'Truncated due to dynamic pressure > 65000, dynamic_pressure = {dynamic_pressure}, y = {y}')
                return True, 4
            elif info['g_load_1_sec_window'] > 6.0:
                print(f'Truncated due to acceleration > 6.0, acceleration = {acceleration}, y = {y}, vy = {vy}, vyp = {vyp}')
                return True, 5
            elif vy > 0.0:
                print(f'Truncated due to vy > 0.0, vy = {vy}, y = {y}')
                return True, 6
            else:
                return False, 0
    
    def reward_func_lambda(state, done, truncated, actions, previous_state, info):
        return 0
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda