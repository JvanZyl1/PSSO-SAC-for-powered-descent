import csv
import pandas as pd
import numpy as np

def load_supersonic_initial_state():
    data = pd.read_csv('data/reference_trajectory/ascent_controls/subsonic_state_action_ascent_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], last_row['mass[kg]'], last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state

def load_subsonic_initial_state():
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]
    # Initial physics state : x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
    initial_physics_state = np.array([0,                                                        # x [m]
                                      1.5,                                                        # y [m] slightly up to allow for negative vy for learning.
                                      0,                                                        # vx [m/s]
                                      0,                                                        # vy [m/s]
                                      np.pi/2,                                                  # theta [rad]
                                      0,                                                        # theta_dot [rad/s]
                                      0,                                                        # gamma [rad]
                                      0,                                                        # alpha [rad]
                                      float(sizing_results['Initial mass (subrocket 0)'])*1000,             # mass [kg]
                                      float(sizing_results['Actual propellant mass stage 1'])*1000,       # mass_propellant [kg]
                                      0])                                                       # time [s]
    return initial_physics_state

def load_flip_over_initial_state():
    data = pd.read_csv('data/reference_trajectory/ascent_controls/supersonic_state_action_ascent_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]
    mass = last_row['mass_propellant[kg]'] + float(sizing_results['Actual structural mass stage 1'])*1000
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], mass, last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state

def load_high_altitude_ballistic_arc_initial_state():
    data = pd.read_csv('data/reference_trajectory/flip_over_and_boostbackburn_controls/state_action_flip_over_and_boostbackburn_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], last_row['mass[kg]'], last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state


def load_landing_burn_initial_state():
    data = pd.read_csv('data/reference_trajectory/ballistic_arc_descent_controls/state_action_ballistic_arc_descent_control.csv')
    # time,x,y,vx,vy,theta,theta_dot,gamma,alpha,mass,mass_propellant : csv
    # state = [x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time]
    last_row = data.iloc[-1]
    state = [last_row['x[m]'], last_row['y[m]'], last_row['vx[m/s]'], last_row['vy[m/s]'], last_row['theta[rad]'], last_row['theta_dot[rad/s]'], last_row['gamma[rad]'], last_row['alpha[rad]'], last_row['mass[kg]'], last_row['mass_propellant[kg]'], last_row['time[s]']]
    return state