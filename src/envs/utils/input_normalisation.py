import math
import numpy as np
import pandas as pd

def subsonic_input_normalisation():
    file_path = 'data/reference_trajectory/ascent_controls/subsonic_state_action_ascent_control.csv'
    data = pd.read_csv(file_path)
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'mass[kg]']].values
    # Find max absolute values of each state
    x_norm_val = np.max(np.abs(states[:, 0])) + 100
    y_norm_val = np.max(np.abs(states[:, 1])) + 500
    vx_norm_val = np.max(np.abs(states[:, 2])) + 5
    vy_norm_val = np.max(np.abs(states[:, 3])) + 50
    theta_norm_val = np.max(np.abs(states[:, 4])) + math.radians(2)
    theta_dot_norm_val = np.max(np.abs(states[:, 5]))*2.5
    alpha_norm_val = np.max(np.abs(states[:, 6])) + math.radians(3)
    mass_norm_val = np.max(np.abs(states[:, 7]))

    # np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])

    input_normalisation_vals = np.array([x_norm_val, y_norm_val, vx_norm_val, vy_norm_val, theta_norm_val, theta_dot_norm_val, alpha_norm_val, mass_norm_val])

    return input_normalisation_vals

def supersonic_input_normalisation():
    file_path = 'data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv'
    data = pd.read_csv(file_path)
    states = data[['x[m]', 'y[m]', 'vx[m/s]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'mass[kg]']].values
    # Find max absolute values of each state
    x_norm_val = np.max(np.abs(states[:, 0])) + 2500
    y_norm_val = np.max(np.abs(states[:, 1])) + 5000
    vx_norm_val = np.max(np.abs(states[:, 2])) + 100
    vy_norm_val = np.max(np.abs(states[:, 3])) + 150
    theta_norm_val = np.max(np.abs(states[:, 4])) + math.radians(5)
    theta_dot_norm_val = np.max(np.abs(states[:, 5]))*2.5
    alpha_norm_val = np.max(np.abs(states[:, 6])) + math.radians(3)
    mass_norm_val = np.max(np.abs(states[:, 7]))

    # np.array([x, y, vx, vy, theta, theta_dot, alpha, mass])

    input_normalisation_vals = np.array([x_norm_val, y_norm_val, vx_norm_val, vy_norm_val, theta_norm_val, theta_dot_norm_val, alpha_norm_val, mass_norm_val])

    return input_normalisation_vals

def flip_over_boostbackburn_input_normalisation():
    file_path = 'data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv'
    data = pd.read_csv(file_path)
    states = data[['theta[rad]', 'theta_dot[rad/s]']].values
    # Find max absolute values of each state
    theta_norm_val = np.max(np.abs(states[:, 0])) + math.radians(5)
    theta_dot_norm_val = np.max(np.abs(states[:, 1]))*2.5

    # np.array([theta, theta_dot])

    input_normalisation_vals = np.array([theta_norm_val, theta_dot_norm_val])

    return input_normalisation_vals

def ballistic_arc_descent_input_normalisation():
    file_path = 'data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv'
    data = pd.read_csv(file_path)
    states = data[['theta[rad]', 'theta_dot[rad/s]', 'alpha[rad]', 'gamma[rad]']].values
    # Find max absolute values of each state
    theta_norm_val = np.max(np.abs(states[:, 0])) + math.radians(5)
    theta_dot_norm_val = np.max(np.abs(states[:, 1]))*2.5
    gamma_norm_val = np.max(np.abs(states[:, 3])) + math.radians(5)
    alpha_norm_val = np.max(np.abs(states[:, 2])) + math.radians(5)

    input_normalisation_vals = np.array([theta_norm_val, theta_dot_norm_val, gamma_norm_val, alpha_norm_val])

    return input_normalisation_vals

def landing_burn_input_normalisation():
    #action_state = np.array([y, vy,])
    file_path_ballistic_arc = 'data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv'
    data_ballistic_arc = pd.read_csv(file_path_ballistic_arc)
    states_ballistic_arc = data_ballistic_arc[['y[m]', 'vy[m/s]', 'theta[rad]', 'theta_dot[rad/s]', 'gamma[rad]', 'x[m]', 'vx[m/s]']].values
    y_norm_val = np.max(np.abs(states_ballistic_arc[:, 0])) + 100
    vy_norm_val = np.max(np.abs(states_ballistic_arc[:, 1])) + 50
    theta_norm_val = math.pi/2
    theta_dot_norm_val = 0.01
    gamma_norm_val = math.pi * 3/2
    x_norm_val = np.max(np.abs(states_ballistic_arc[:, 5])) + 1500
    vx_norm_val = np.max(np.abs(states_ballistic_arc[:, 6])) + 30


    input_normalisation_vals = np.array([y_norm_val, vy_norm_val, theta_norm_val, theta_dot_norm_val, gamma_norm_val, x_norm_val, vx_norm_val])

    return input_normalisation_vals

def find_input_normalisation_vals(flight_phase : str):
    if flight_phase == 'subsonic':
        return subsonic_input_normalisation()
    elif flight_phase == 'supersonic':
        return supersonic_input_normalisation()
    elif flight_phase == 'flip_over_boostbackburn':
        return flip_over_boostbackburn_input_normalisation()
    elif flight_phase == 'ballistic_arc_descent':
        return ballistic_arc_descent_input_normalisation()
    elif flight_phase in ['landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
        return landing_burn_input_normalisation()
    else:
        raise ValueError(f"Invalid flight phase: {flight_phase}")