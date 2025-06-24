import dill
import math
import numpy as np
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from src.envs.utils.reference_trajectory_interpolation import reference_trajectory_lambda_func_y
from src.envs.utils.atmosphere_dynamics import endo_atmospheric_model
from src.envs.load_initial_states import load_landing_burn_initial_state

def compile_rtd_rl_ascent(reference_trajectory_func_y,
                                learning_hyperparameters,
                                terminal_mach): # i.e. mach goes from 0 to 1.0, but can stop between 1.0 and 1.1
    machs = [hyperparameter[0] for hyperparameter in learning_hyperparameters]
    max_x_errors = [hyperparameter[1] for hyperparameter in learning_hyperparameters]
    max_vy_errors = [hyperparameter[2] for hyperparameter in learning_hyperparameters]
    max_vx_errors = [hyperparameter[3] for hyperparameter in learning_hyperparameters]
    max_alpha_degs = [hyperparameter[4] for hyperparameter in learning_hyperparameters]
    alpha_reward_weights = [hyperparameter[5] for hyperparameter in learning_hyperparameters]
    x_reward_weights = [hyperparameter[6] for hyperparameter in learning_hyperparameters]
    vy_reward_weights = [hyperparameter[7] for hyperparameter in learning_hyperparameters]
    vx_reward_weights = [hyperparameter[8] for hyperparameter in learning_hyperparameters]

    # Interpolate the learning hyperparameters
    f_max_x_error = interp1d(machs, max_x_errors, kind='linear', fill_value='extrapolate')
    f_max_vy_error = interp1d(machs, max_vy_errors, kind='linear', fill_value='extrapolate')
    f_max_vx_error = interp1d(machs, max_vx_errors, kind='linear', fill_value='extrapolate')
    f_max_alpha_deg = interp1d(machs, max_alpha_degs, kind='linear', fill_value='extrapolate')
    f_alpha_reward_weight = interp1d(machs, alpha_reward_weights, kind='linear', fill_value='extrapolate')
    f_x_reward_weight = interp1d(machs, x_reward_weights, kind='linear', fill_value='extrapolate')
    f_vy_reward_weight = interp1d(machs, vy_reward_weights, kind='linear', fill_value='extrapolate')
    f_vx_reward_weight = interp1d(machs, vx_reward_weights, kind='linear', fill_value='extrapolate')    
    
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if any(math.isnan(val) for val in state):
            print(f'Truncated state due to NaN: {state}')
            return False
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        if speed != 0 and speed_of_sound != 0:
            mach_number = speed / speed_of_sound
        else:
            mach_number = 0
        
        if mass_propellant >= 0 and mach_number > terminal_mach:
            return True
        else:
            return False
        
    def truncated_func_lambda(state, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if any(math.isnan(val) for val in state):
            print(f'Truncated state due to NaN: {state}')
            return True, 0
        xr, yr, vxr, vyr, m  = reference_trajectory_func_y(y)

        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        if speed != 0 and speed_of_sound != 0:
            mach_number = speed / speed_of_sound
        else:
            mach_number = 0

        # If mass is depleted, return True
        if mass_propellant <= 0:
            return True, 1
        elif mach_number > terminal_mach + 0.09:
            return True, 2
        elif abs(x - xr) > f_max_x_error(mach_number):
            return True, 3
        elif y < 0:
            return True, 4
        elif abs(alpha) > math.radians(f_max_alpha_deg(mach_number)):
            return True, 5
        elif abs(vx - vxr) > f_max_vx_error(mach_number):
            return True, 6
        elif abs(vy - vyr) > f_max_vy_error(mach_number):
            return True, 7
        else:
            return False, 0

    def reward_func_lambda(state, done, truncated, actions, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if any(math.isnan(val) for val in state):
            print(f'Truncated state due to NaN: {state}')
            return 0
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        if speed != 0 and speed_of_sound != 0:
            mach_number = speed / speed_of_sound
        else:
            mach_number = 0
        reward = 0

        # Get the reference trajectory
        xr, _, vxr, vyr, m = reference_trajectory_func_y(y)
        # Special errors
        if y < 0:
            return 0

        reward += math.exp(-4 * (vx - vxr)**2/f_max_vx_error(mach_number)**2) * f_vx_reward_weight(mach_number)
        reward += math.exp(-4 * (vy - vyr)**2/f_max_vy_error(mach_number)**2) * f_vy_reward_weight(mach_number)
        reward += math.exp(-4 * (x - xr)**2/f_max_x_error(mach_number)**2) * f_x_reward_weight(mach_number)
        reward += math.exp(-4*math.degrees(alpha)**2/f_max_alpha_deg(mach_number)**2) * f_alpha_reward_weight(mach_number)

        # Done function
        if done:
            reward += 2.5

        reward /= 10**4
        return reward

    return reward_func_lambda, truncated_func_lambda, done_func_lambda

def compile_rtd_rl_test_boostback_burn(theta_abs_error_max):
    flip_over_boostbackburn_terminal_vx = -20
    data = pd.read_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv')
    theta = data['theta[rad]'].values
    vy_vals = data['vy[m/s]'].values
    f_theta = interp1d(vy_vals, theta, kind='linear', fill_value='extrapolate')
    def theta_abs_error(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        theta_ref = f_theta(vy)
        return abs(theta_ref - theta)

    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        if vx < flip_over_boostbackburn_terminal_vx:
            return True
        else:
            return False
    
    def truncated_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state

        if mass_propellant <= 0:
            return True, 1
        elif theta_abs_error(state) > theta_abs_error_max:
            return True, 2
        else:
            return False, 0
    
    def reward_func_lambda(state, done, truncated, actions, previous_state, info):
        reward = 1 - theta_abs_error(state)/theta_abs_error_max
        if done:
            reward += 0.25
        reward /= 100
        return reward
    
    return reward_func_lambda, truncated_func_lambda, done_func_lambda

def compile_rtd_rl_ballistic_arc_descent(dynamic_pressure_threshold = 10000):
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)
        if dynamic_pressure > dynamic_pressure_threshold and \
            abs_alpha_effective < math.radians(3):
            return True
        else:
            return False
    
    def truncated_func_lambda(state, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        abs_alpha_effective = abs(gamma - theta - math.pi)
        if dynamic_pressure > dynamic_pressure_threshold - 2000 and \
            abs_alpha_effective > math.radians(5):
            return True, 1
        else:
            return False, 0
    
    def reward_func_lambda(state, done, truncated, actions, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        abs_alpha_effective = abs(gamma - theta - math.pi)

        reward = (math.pi - abs_alpha_effective)/math.pi
        if done:
            reward += 3.5
        reward /= 100
        return reward

    return reward_func_lambda, truncated_func_lambda, done_func_lambda

def compile_rtd_rl_landing_burn(trajectory_length, discount_factor, pure_throttle = False, dt = 0.1):
    sparse_bool = False
    max_alpha_effective = math.radians(20)
    x_0, y_0, vx_0, vy_0, theta_0, theta_dot_0, gamma_0, alpha_0, mass_0, mass_propellant_0, time_0 = load_landing_burn_initial_state()
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        if y > 0 and y < 1:
            if speed < 5.0:
                print(f'Done due to speed < 5.0, speed = {speed}, y = {y}')
                return True
            else:
                return False
        else:
            return False
    
    def truncated_func_lambda(state, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        xp, yp, vxp, vyp, thetap, theta_dotp, gammamp, alphap, massp, mass_propellantp, timep = previous_state
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * air_density * speed**2
        speed_p = math.sqrt(vxp**2 + vyp**2)
        speed_diff = abs(speed - speed_p)
        acceleration = speed_diff/dt * 1/9.81
        if vy < 0:
            alpha_effective = abs(gamma - theta - math.pi)
        else:
            alpha_effective = abs(theta - gamma)
        if y < -10:
            return True, 1
        elif mass_propellant <= 0:
            #print(f'Truncated due to mass_propellant <= 0, mass_propellant = {mass_propellant}, y = {y}')
            return True, 2
        elif theta > math.pi + math.radians(2):
            #print(f'Truncated due to theta > math.pi + math.radians(2), theta = {theta}, y = {y}')
            return True, 3
        elif dynamic_pressure > 65000:
            #print(f'Truncated due to dynamic pressure > 65000, dynamic_pressure = {dynamic_pressure}, y = {y}')
            return True, 4
        elif info['g_load_1_sec_window'] > 6.0:
            return True, 5
        elif vy > 0.0:
            #print(f'Truncated due to vy > 0.0, vy = {vy}, y = {y}')
            return True, 6
        elif vx > 0.01:
            return True, 7
        else:
            return False, 0
    
    if not pure_throttle and not sparse_bool:
        def reward_func_lambda(state, done, truncated, actions, previous_state, info):
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
            air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
            speed = math.sqrt(vx**2 + vy**2)
            dynamic_pressure = 0.5 * air_density * speed**2
            alpha_effective = abs(gamma - theta - math.pi)
            reward = 0
            #reward += 1 - math.log(1 + alpha_effective)/(math.log(1+max_alpha_effective)) # [0, 1]
            if actions.ndim == 2:
                u0 = actions[0][0]
            else:
                u0 = actions[0]
            tau = (u0 + 1)/2
            reward += (1.5 - math.log(1 + alpha_effective)/(math.log(1+max_alpha_effective)) - tau*0.5)*(1-y/y_0) * 2/3
            # Throttle reward, u0 is normalised throttle (-1 to 1) so have to move to (0 to 1)
            
            #reward += (u0 + 1)/2*0.25
            #reward /= 1.25 # Scale reward to 1.0
            if y < 100: # Want to minimise the vy, vy = 30 -> r = 0.25
                reward += 1 - math.tanh((speed-15)/15)
            if truncated and y < 5:
                reward += 1 - math.tanh((speed-5)/5)
            if done: # Not looked at this yet
                reward += 5
            reward *= (1 - discount_factor)/(1 - discount_factor**trajectory_length) # n-step rewards scaling
            # CHANGED THIS REWARD FUNCTION
            return reward
    elif not sparse_bool and pure_throttle:
        # ---------- constants ----------
        Q_MAX_THRES    = 60_000.0
        Q_MAX          = 65_000.0
        G_MAX_THRES    = 5.5
        G_MAX          = 6.0
        W_Q_PENALTY    = 1.0
        W_G_PENALTY    = 1.0
        W_PROGRESS     = 0.5
        W_TERMINAL     = 400.0
        W_CRASH        = 50.0
        W_SLOWDOWN     = 5.5
        CLIP_LIMIT     = 10.0
        Y0_FIXED       = y_0
        # ---------------------------------

        def reward_func_lambda(state, done, truncated, actions, previous_state, info):
            # unpack state
            x,  y,  vx,     vy,     theta,  theta_dot,  gamma,      alpha,  mass,   mass_propellant,    time =  state

            # atmosphere model
            air_density, _, _ = endo_atmospheric_model(y)
            speed             = math.hypot(vx, vy)
            q                 = 0.5 * air_density * speed**2
            reward = 0.0

            # 1.
            if q > Q_MAX_THRES:
                q_excess = (q - Q_MAX_THRES) / (Q_MAX - Q_MAX_THRES)
                reward  -= W_Q_PENALTY * min(q_excess**2, 1.0)

            # 2.
            g_load = info['g_load_1_sec_window']
            if g_load > G_MAX_THRES:
                g_excess = (g_load - G_MAX_THRES) / (G_MAX - G_MAX_THRES)
                reward  -= W_G_PENALTY * min(g_excess**2, 1.0)

            # 3.
            altitude_progress = (Y0_FIXED - y) / Y0_FIXED
            w_prog = W_PROGRESS if (q <= Q_MAX_THRES and g_load <= G_MAX_THRES) else W_PROGRESS * 0.1
            reward += w_prog * altitude_progress


            # 4.
            if y < 100.0:
                slowdown_reward = 1.0 - abs(vy) / 50.0
                reward         += W_SLOWDOWN * slowdown_reward

            # 5.
            if done and not truncated:
                reward  += W_TERMINAL * mass_propellant/mass_0
            elif truncated and y > 0:
                altitude_factor  = abs(y) / Y0_FIXED
                reward          -= W_CRASH * altitude_factor
            elif truncated and y < 0:
                speed_factor = abs(vy)/10
                reward -= W_CRASH * speed_factor

            # 6.
            if not done or not (truncated and y < 0):
                reward = np.clip(reward, -CLIP_LIMIT, CLIP_LIMIT)
            # so don't clip if y < 0

            # N-step rewards scaling
            #reward *= (1 - discount_factor)/(1 - discount_factor**trajectory_length)

            return reward
    elif sparse_bool:
        def reward_func_lambda(state, done, truncated, actions, previous_state, info):
            x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
            speed = math.sqrt(vx**2 + vy**2)
            reward = 0
            if truncated and y > 0.1:
                reward = -max(0, abs(y)/y_0)*6
            elif truncated and y < 0.1:
                reward = 6 - speed/200
            if y < 100:
                reward += 2 * (1 - abs(speed)/200)
            if done:
                reward = mass_propellant/844888*10
            return reward
    return reward_func_lambda, truncated_func_lambda, done_func_lambda
        
def compile_rtd_rl_landing_burn_PDcontrol(trajectory_length, discount_factor, pure_throttle = False, dt = 0.1):
    max_alpha_effective = math.radians(20)
    x_0, y_0, vx_0, vy_0, theta_0, theta_dot_0, gamma_0, alpha_0, mass_0, mass_propellant_0, time_0 = load_landing_burn_initial_state()
    def done_func_lambda(state):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * density * speed**2
        if y > 0 and y < 5:
            if speed < 1:
                return True
            else:
                return False
        else:
            return False
    
    def truncated_func_lambda(state, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        xp, yp, vxp, vyp, thetap, theta_dotp, gammamp, alphap, massp, mass_propellantp, timep = previous_state
        speed = math.sqrt(vx**2 + vy**2)
        dynamic_pressure = 0.5 * air_density * speed**2
        speed_p = math.sqrt(vxp**2 + vyp**2)
        speed_diff = abs(speed - speed_p)
        acceleration = speed_diff/dt * 1/9.81
        if vy < 0:
            alpha_effective = abs(gamma - theta - math.pi)
        else:
            alpha_effective = abs(theta - gamma)
        if y < -10:
            print(f'Truncated due to y < -10, y = {y}')
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
            print(f'Truncated due to g_load_1_sec_window > 6.0, g_load_1_sec_window = {info["g_load_1_sec_window"]}, y = {y}')
            return True, 5
        elif vy > 0.0:
            print(f'Truncated due to vy > 0.0, vy = {vy}, y = {y}')
            return True, 6
        else:
            return False, 0

    def reward_func_lambda(state, done, truncated, actions, previous_state, info):
        x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time = state
        xp, yp, vxp, vyp, thetap, theta_dotp, gammamp, alphap, massp, mass_propellantp, timep = previous_state
        air_density, atmospheric_pressure, speed_of_sound = endo_atmospheric_model(y)
        speed = math.sqrt(vx**2 + vy**2)
        q = 0.5 * air_density * speed**2

        air_density_p, atmospheric_pressure_p, speed_of_sound_p = endo_atmospheric_model(yp)
        speed_p = math.sqrt(vxp**2 + vyp**2)
        dynamic_pressure_p = 0.5 * air_density_p * speed_p**2

        if actions.ndim == 2:
            v_ref = actions[0][0]
        else:
            v_ref = actions[0]

        reward = 0.0
        w_q_penalty = 200
        q_max_thres = 60000
        q_max = 65000
        if q > q_max_thres:
            reward -= w_q_penalty * ((q - q_max_thres) / (q_max -  q_max_thres))
        
        # 2.
        w_g_penalty = 100
        g_max = 6.0
        g_max_thres = 5.5
        g_load = info['g_load_1_sec_window']
        if g_load > g_max_thres:
            reward -= w_g_penalty * ((g_load - g_max_thres) / (g_max -  g_max_thres))

        # 3.
        w_progress = 5.0
        if q <= q_max and g_load <= g_max_thres:
            altitude_progress = (y_0 - y) / y_0
            vel_tracking = max(0.0, 1.0 - abs(speed - v_ref)/10)
            reward += w_progress * altitude_progress * vel_tracking

        # 4.
        w_mass_used=1.0
        w_crash = 800
        w_terminal=1000
        if done:
            reward += w_terminal
            mass_used = y_0 - mass
            reward -= w_mass_used * mass_used
        elif truncated:
            altitude_penalty = y / y_0
            reward -= w_crash * altitude_penalty

        if y < 100:
            slowdown_reward = 1 - math.tanh((vy - 50) / 50)
            reward += slowdown_reward

        reward *= (1 - discount_factor)/(1 - discount_factor**trajectory_length)
        return reward

    GAMMA          = discount_factor
    Q_MAX_THRES    = 60_000.0
    Q_MAX          = 65_000.0
    G_MAX_THRES    = 5.5
    G_MAX          = 6.0
    W_Q_PENALTY    = 1.0
    W_G_PENALTY    = 1.0
    W_PROGRESS     = 0.5
    W_TERMINAL     = 5.0
    W_CRASH        = 4.0
    W_MASS_USED    = 0.1
    W_SLOWDOWN     = 0.5
    ALIVE_BONUS    = 0.01 * (1 - GAMMA)
    CLIP_LIMIT     = 10.0
    Y0_FIXED       = y_0
    VY_TARGET_NEAR = 0.0
    VEL_TRACK_DEN  = 10.0
    # ---------------------------------

    def reward_func_lambda(state, done, truncated, actions, previous_state, info):
        # unpack state
        x,  y,  vx,     vy,     theta,  theta_dot,  gamma,      alpha,  mass,   mass_propellant,    time =  state
        xp, yp, vxp,    vyp,    thetap, theta_dotp, gammamp,    alphap, massp,  mass_propellantp,   timep = previous_state

        # atmosphere model
        air_density, _, _ = endo_atmospheric_model(y)
        speed             = math.hypot(vx, vy)
        q                 = 0.5 * air_density * speed**2

        v_ref = actions[0][0] if actions.ndim == 2 else actions[0]

        reward = 0.0

        # 1.
        if q > Q_MAX_THRES:
            q_excess = (q - Q_MAX_THRES) / (Q_MAX - Q_MAX_THRES)
            reward  -= W_Q_PENALTY * min(q_excess**2, 1.0)

        # 2.
        g_load = info['g_load_1_sec_window']
        if g_load > G_MAX_THRES:
            g_excess = (g_load - G_MAX_THRES) / (G_MAX - G_MAX_THRES)
            reward  -= W_G_PENALTY * min(g_excess**2, 1.0)

        # 3.
        altitude_progress = (Y0_FIXED - y) / Y0_FIXED
        vel_tracking      = max(0.0, 1.0 - abs(speed - v_ref) / VEL_TRACK_DEN)
        w_prog            = W_PROGRESS if (q <= Q_MAX_THRES and g_load <= G_MAX_THRES) else W_PROGRESS * 0.1
        reward           += w_prog * altitude_progress * vel_tracking

        # 4.
        if y < 100.0:
            slowdown_reward = max(0.0, 1.0 - abs(vy - VY_TARGET_NEAR) / 50.0)
            reward         += W_SLOWDOWN * slowdown_reward

        # 5.
        reward += ALIVE_BONUS

        # 6.
        if done and not truncated:
            reward += W_TERMINAL
            mass_used = Y0_FIXED * 0.0 + (mass_0 - mass)
            reward   -= min(W_MASS_USED * mass_used, 1.0)
        elif truncated:
            impact_speed     = abs(vy)
            altitude_factor  = y / Y0_FIXED
            reward          -= min(W_CRASH * altitude_factor * (impact_speed / 100.0), 5.0)

        # 7.
        reward = np.clip(reward, -CLIP_LIMIT, CLIP_LIMIT)

        return reward


    return reward_func_lambda, truncated_func_lambda, done_func_lambda
           

def compile_rtd_rl(flight_phase, trajectory_length, discount_factor, dt):
    assert flight_phase in ['subsonic','supersonic','flip_over_boostbackburn','ballistic_arc_descent', 'landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']
    if flight_phase not in ['landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
        reference_trajectory_func_y, terminal_state = reference_trajectory_lambda_func_y(flight_phase)

        # [[mach, max_x_error, max_vy_error, max_vx_error, max_alpha_deg, alpha_reward_weight, x_reward_weight, vy_reward_weight, vx_reward_weight], ...]
        subsonic_learning_hyperparameters = [
            # [mach,    max_x_error,    max_vy_error,   max_vx_error,   max_alpha_deg,  alpha_reward_weight,    x_reward_weight,    vy_reward_weight,   vx_reward_weight]
            [0.0,     50,              10,               10,            0.5,              100,                    100,                100,                 100],
            [0.1,     50,              15,               10,             10,              100,                    100,                100,                 100],
            [0.2,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.3,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.4,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.5,     50,              20,                5,              2,              100,                    100,                100,                 100],
            [0.6,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [0.7,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [0.8,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [0.9,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [1.0,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
            [1.1,     50,              20,                5,           1.75,              100,                    100,                100,                 100],
        ]

        # For mach in range 1 to max mach append a mock config for now
        supersonic_learning_hyperparameters = [
            # [mach,    max_x_error,    max_vy_error,   max_vx_error,   max_alpha_deg,  alpha_reward_weight,    x_reward_weight,    vy_reward_weight,   vx_reward_weight]
            [1.0,     100,            50,              9,              8,              100,                    100,                100,                  100],
            [1.1,     100,            60,             20,              8,              100,                    100,                100,                  100],
            [1.5,     100,            60,             20,              8,              100,                    100,                100,                  100],
            [1.75,    100,            60,             30,              8,              100,                    100,                100,                  100],
            [2.0,     100,            60,             40,              8,              100,                    100,                100,                  100],
            [2.25,    100,            60,             50,              8,              100,                    100,                100,                  100],
            [2.5,     100,            60,             60,              8,              100,                    100,                100,                  100],
            [2.75,    100,            60,             70,              8,              100,                    100,                100,                  100],
            [3.0,     100,            60,             80,              8,              100,                    100,                100,                  100],
            [3.25,    100,            60,             90,              8,              100,                    100,                100,                  100],
            [3.5,     100,            60,            100,              8,              100,                    100,                100,                  100],
            [3.75,    100,            60,            100,              8,              100,                    100,                100,                  100],
        ]
    
    if flight_phase == 'subsonic':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_ascent(reference_trajectory_func_y,
                                                                                                  learning_hyperparameters = subsonic_learning_hyperparameters,
                                                                                                  terminal_mach = 1.0)
    elif flight_phase == 'supersonic':
        # Extract maximum Mach Number
        xt, yt, vxt, vyt, mt = terminal_state
        print(f'Terminal state: {terminal_state}')
        density_t, atmospheric_pressure_t, speed_of_sound_t = endo_atmospheric_model(yt)
        speed_t = math.sqrt(vxt**2 + vyt**2)
        mach_number_t = speed_t / speed_of_sound_t
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_ascent(reference_trajectory_func_y,
                                                                                                  learning_hyperparameters = supersonic_learning_hyperparameters,
                                                                                                  terminal_mach = mach_number_t)
    elif flight_phase == 'flip_over_boostbackburn':
        theta_abs_error_max_rad = math.radians(4)
        reward_func_lambda, truncated_func_lambda, done_func_lambda =  compile_rtd_rl_test_boostback_burn(theta_abs_error_max_rad)
    elif flight_phase == 'ballistic_arc_descent':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_ballistic_arc_descent(dynamic_pressure_threshold = 10000)
    elif flight_phase == 'landing_burn' or flight_phase == 'landing_burn_ACS':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_landing_burn(trajectory_length, discount_factor, pure_throttle = False, dt = dt)
    elif flight_phase == 'landing_burn_pure_throttle':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_landing_burn(trajectory_length, discount_factor, pure_throttle = True, dt = dt)
    elif flight_phase == 'landing_burn_pure_throttle_Pcontrol':
        reward_func_lambda, truncated_func_lambda, done_func_lambda = compile_rtd_rl_landing_burn_PDcontrol(trajectory_length, discount_factor, dt = dt)
    else:
        raise ValueError(f'Invalid flight stage: {flight_phase}')

    return reward_func_lambda, truncated_func_lambda, done_func_lambda