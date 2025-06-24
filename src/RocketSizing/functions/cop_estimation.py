import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import log as ln
from math import pi as pi

def cop_func_to_move_to(L, alpha, M,
             d_0 = 0.25,
             d_alpha = 0.45 * pi,
             d_mach_tilde = 0.1):
    if M < 0.8:
        d = d_0 + d_alpha * alpha
    elif M < 1.2:
        lambda_val = (M - 0.8) / 0.4
        d = d_0 + (1 - lambda_val) * d_alpha * alpha - lambda_val * d_mach_tilde * ln(M)
    else:
        d = d_0 - d_mach_tilde * ln(M)
    return L * d

def cop_func(L, alpha, M, d_0):
    return d_0 * L

def plot_cop_func(L=1.0):
    # Define Mach and angle-of-attack ranges
    mach_vals = np.linspace(0.0, 4.5, 100)
    alpha_vals = np.linspace(-10, 10, 100)  # in degrees
    alpha_vals_rad = np.radians(alpha_vals)
    
    mach_grid, alpha_grid = np.meshgrid(mach_vals, alpha_vals_rad)
    
    cop_values = np.zeros_like(mach_grid)
    for i in range(len(alpha_vals_rad)):
        for j in range(len(mach_vals)):
            cop_values[i, j] = cop_func_to_move_to(L, alpha_vals_rad[i], mach_vals[j])
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(np.degrees(alpha_grid), mach_grid, cop_values, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Center of Pressure (x/L)')
    contour_lines = plt.contour(np.degrees(alpha_grid), mach_grid, cop_values, levels=10, colors='white', linewidths=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=16, fmt='%.2f')
    
    plt.title('Center of Pressure Location')
    plt.xlabel(r'Angle of Attack ($^{\circ}$)', fontsize=18)
    plt.ylabel('Mach Number', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()
    plt.savefig('results/Sizing/cop_contour.png')
    plt.close()

if __name__ == "__main__":
    plot_cop_func()
    