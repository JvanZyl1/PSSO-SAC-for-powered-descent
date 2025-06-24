import matplotlib.pyplot as plt
import numpy as np
from src.classical_controls.flip_over_and_boostbackburn_control import FlipOverandBoostbackBurnControl

def compare_pitch_gains():
    
    # Define the two sets of gains
    pso_gains = (-22.0, -4.8)  # PSO optimized gains
    normal_gains = (-18.0, -5.0)  # Normal gains
    
    
    pso_controller = FlipOverandBoostbackBurnControl(
        pitch_tuning_bool=True, 
        Kp_theta_flip=pso_gains[0], 
        Kd_theta_flip=pso_gains[1]
    )
    pso_controller.run_closed_loop()
    
    
    normal_controller = FlipOverandBoostbackBurnControl(
        pitch_tuning_bool=True, 
        Kp_theta_flip=normal_gains[0], 
        Kd_theta_flip=normal_gains[1]
    )
    normal_controller.run_closed_loop()
    
    plt.figure(figsize=(8.27, 5.5))
    AXIS_LABEL_SIZE = 10
    LEGEND_SIZE = 10
    TICK_SIZE = 10
    plt.subplot(2, 2, 1)
    plt.plot(pso_controller.time_vals, pso_controller.pitch_angle_deg_vals, 
             linewidth=3, color='blue', label=f'PSO Gains')
    plt.plot(normal_controller.time_vals, normal_controller.pitch_angle_deg_vals, 
             linewidth=3, color='red', label=f'Manual Gains')
    plt.plot(pso_controller.time_vals, pso_controller.pitch_angle_reference_deg_vals, 
             linewidth=2, color='black', linestyle='--', label='Reference')
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$\theta$ [$^\circ$]', fontsize=AXIS_LABEL_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(True)
    plt.subplot(2, 2, 2)
    pso_errors = [abs(184 - pitch) for pitch in pso_controller.pitch_angle_deg_vals]
    normal_errors = [abs(184 - pitch) for pitch in normal_controller.pitch_angle_deg_vals]
    plt.plot(pso_controller.time_vals, pso_errors, 
             linewidth=3, color='blue', label=f'PSO Gains')
    plt.plot(normal_controller.time_vals, normal_errors, 
             linewidth=3, color='red', label=f'Manual Gains')
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$\theta_{\text{ref}} - \theta$ [$^\circ$]', fontsize=AXIS_LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(True)
    plt.subplot(2, 2, 3)
    plt.plot(pso_controller.time_vals, pso_controller.gimbal_angle_commanded_deg_vals, 
             linewidth=2, color='lightblue', linestyle='--', label=f'PSO Gains - Commanded')
    plt.plot(pso_controller.time_vals, pso_controller.gimbal_angle_deg_vals, 
             linewidth=3, color='blue', label=f'PSO Gains - Actual')
    plt.plot(normal_controller.time_vals, normal_controller.gimbal_angle_commanded_deg_vals, 
             linewidth=2, color='lightcoral', linestyle='--', label=f'Normal Gains - Commanded')
    plt.plot(normal_controller.time_vals, normal_controller.gimbal_angle_deg_vals, 
             linewidth=3, color='red', label=f'Normal Gains - Actual')
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$\theta^g$ [$^\circ$]', fontsize=AXIS_LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(True)
    plt.subplot(2, 2, 4)
    plt.plot(pso_controller.time_vals, pso_controller.pitch_rate_deg_vals, 
             linewidth=3, color='blue', label=f'PSO Gains')
    plt.plot(normal_controller.time_vals, normal_controller.pitch_rate_deg_vals, 
             linewidth=3, color='red', label=f'Manual Gains')
    plt.xlabel(r'$t$ [$s$]', fontsize=AXIS_LABEL_SIZE)
    plt.ylabel(r'$\dot{\theta}$ [$^\circ$/s]', fontsize=AXIS_LABEL_SIZE)
    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/classical_controllers/pitch_gains_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    def calculate_settling_time(time_vals, pitch_vals, reference=184, threshold=2):
        settled_time = 0
        for i, (t, pitch) in enumerate(zip(time_vals, pitch_vals)):
            error = abs(reference - pitch)
            if error < threshold:
                settled_time += 0.1  # dt = 0.1
                if settled_time >= 1.0:
                    return t - settled_time + 1.0
            else:
                settled_time = 0
        return None
    
    pso_settling = calculate_settling_time(pso_controller.time_vals, pso_controller.pitch_angle_deg_vals)
    normal_settling = calculate_settling_time(normal_controller.time_vals, normal_controller.pitch_angle_deg_vals)
    
    print(f"PSO Gains ({pso_gains[0]}, {pso_gains[1]}):")
    print(f"  - Settling time: {pso_settling:.2f} s")
    print(f"  - Final pitch error: {abs(184 - pso_controller.pitch_angle_deg_vals[-1]):.2f} deg")
    print(f"  - Performance metric: {pso_controller.performance_metrics():.2f}")
    
    print(f"\nNormal Gains ({normal_gains[0]}, {normal_gains[1]}):")
    print(f"  - Settling time: {normal_settling:.2f} s"
    print(f"  - Final pitch error: {abs(184 - normal_controller.pitch_angle_deg_vals[-1]):.2f} deg")
    print(f"  - Performance metric: {normal_controller.performance_metrics():.2f}")

if __name__ == "__main__":
    compare_pitch_gains() 