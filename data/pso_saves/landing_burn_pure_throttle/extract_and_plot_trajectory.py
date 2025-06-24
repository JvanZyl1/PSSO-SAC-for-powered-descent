import pandas as pd
import matplotlib.pyplot as plt
import os

# (x, y) | (t, vy) & (t, vx)
# (t, dynamic_pressure) | (t, throttle)

TICK_SIZE = 10
AXIS_LABEL_SIZE = 10
TITLE_SIZE = 11

def extract_and_plot_trajectory(run_name=None, output_filename="trajectory_analysis.png"):
    script_dir = os.path.dirname(os.path.abspath(__file__))    
    run_path = os.path.join(script_dir, run_name)
    
    traj_data_path = os.path.join(run_path, "trajectory_data")
    trajectory_csv = os.path.join(traj_data_path, "trajectory.csv")
    info_data_csv = os.path.join(traj_data_path, "info_data.csv")
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    trajectory_df = pd.read_csv(trajectory_csv)
    info_df = pd.read_csv(info_data_csv)
    time = trajectory_df['time[s]'].to_numpy()
    x = trajectory_df['x[m]'].to_numpy()
    y = trajectory_df['y[m]'].to_numpy()
    vx = trajectory_df['vx[m/s]'].to_numpy()
    vy = trajectory_df['vy[m/s]'].to_numpy()
    throttle = info_df['action_info_throttle'].to_numpy()
    dynamic_pressure = info_df['dynamic_pressure'].to_numpy()
    
    # plot
    plt.figure(figsize=(8.27, 5.5))
    plt.subplot(2, 2, 1)
    plt.plot(x[:-90]/1000, y[:-90]/1000, color = 'blue', linewidth = 2)
    plt.xlabel(r'$x$ [$km$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$y$ [$km$]', fontsize=AXIS_LABEL_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'A', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    plt.subplot(2, 2, 2)
    plt.plot(time, vx, label=r'$v_x$', color = 'blue', linewidth = 2)
    plt.plot(time, vy, label=r'$v_y$', color = 'red', linewidth = 2)
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$v$ [$m/s$]', fontsize=AXIS_LABEL_SIZE)
    plt.legend(fontsize=TICK_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'B', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    plt.subplot(2, 2, 3)
    plt.plot(time[:len(dynamic_pressure)], dynamic_pressure/1000, color = 'green', linewidth = 2)
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$q$ [$kPa$]', fontsize=AXIS_LABEL_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'C', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    plt.subplot(2, 2, 4)
    plt.plot(time[:len(throttle)], throttle, color = 'orange', linewidth = 2)
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$\tau$', fontsize=AXIS_LABEL_SIZE)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    plt.text(-0.1, 1.15, 'D', transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold', va='top')
    plt.tight_layout()
    output_path = os.path.join(plots_dir, output_filename)
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

if __name__ == "__main__":
    extract_and_plot_trajectory('Landed', 'PSO_trajectory_analysis.png')