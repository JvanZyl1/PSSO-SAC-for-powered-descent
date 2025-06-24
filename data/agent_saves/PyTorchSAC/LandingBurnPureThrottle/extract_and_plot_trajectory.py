import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# from trajectory_data/trajectory.csv extract: time[s],x[m],y[m],vx[m/s],vy[m/s],theta[rad],theta_dot[rad/s],gamma[rad],alpha[rad],mass[kg],mass_propellant[kg]
# from trajectory_data/info_data.csv extract action_info_throttle as throttle
# and dynamic_pressure

# Then plot
# (x, y) | (t, vy) & (t, vx)
# (t, dynamic_pressure) | (t, throttle)

TICK_SIZE = 10
AXIS_LABEL_SIZE = 10
TITLE_SIZE = 10

def extract_and_plot_trajectory():
    file_path = 'data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/5_A_Uniform_2000/trajectories/trajectory.csv'
    output_path = 'data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/plots/trajectory_uniform_2000.png'
    trajectory_df = pd.read_csv(file_path)
    time = trajectory_df['time'].to_numpy()
    x = trajectory_df['x'].to_numpy()
    y = trajectory_df['y'].to_numpy()
    vx = trajectory_df['vx'].to_numpy()
    vy = trajectory_df['vy'].to_numpy()
    throttle = trajectory_df['throttle'].to_numpy()
    dynamic_pressure = trajectory_df['dynamic_pressure'].to_numpy()
    
    plt.figure(figsize=(8.27, 5.5))
    plt.subplot(2, 2, 1)
    plt.plot(x[:-90]/1000, y[:-90]/1000, color = 'blue', linewidth = 2)
    plt.xlabel(r'$x$ [$km$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$y$ [$km$]', fontsize=AXIS_LABEL_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'A', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    
    # Plot 2: (t, vy) & (t, vx)
    plt.subplot(2, 2, 2)
    plt.plot(time, vx, label=r'$v_x$', color = 'blue', linewidth = 2)
    plt.plot(time, vy, label=r'$v_y$', color = 'red', linewidth = 2)
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$V$ [$m/s$]', fontsize=AXIS_LABEL_SIZE)
    plt.legend(fontsize=TICK_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'B', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    
    # Plot 3: (t, dynamic_pressure)
    plt.subplot(2, 2, 3)
    plt.plot(time[:len(dynamic_pressure)], dynamic_pressure/1000, color = 'green', linewidth = 2)
    plt.xlabel(r't [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'q [$kPa$]', fontsize=AXIS_LABEL_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'C', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    
    # Plot 4: (t, throttle)
    plt.subplot(2, 2, 4)
    plt.plot(time[:len(throttle)], throttle, color = 'orange', linewidth = 2)
    plt.xlabel(r't [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$\tau$', fontsize=AXIS_LABEL_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'D', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"Plots saved to {output_path}")
    return output_path

if __name__ == "__main__":
    extract_and_plot_trajectory()