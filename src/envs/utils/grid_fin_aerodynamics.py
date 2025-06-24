import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

def compile_grid_fin_Ca():
    cd_data = pd.read_csv('data/rocket_parameters/GridFin/C_D_grid_fin.csv', header=None)
    mach_cd = cd_data[0].values
    ca = cd_data[1].values  # C_D is approximately C_a
    min_mach_cd = np.min(mach_cd)
    f = interp1d(mach_cd, ca, kind='linear', fill_value="extrapolate")
    min_ca = ca[np.argmin(mach_cd)]
    def ca_func(mach_val):
        if mach_val < min_mach_cd:
            return min_ca
        else:
            return f(mach_val)
    return ca_func

def compile_grid_fin_Cn():
    cn_alpha_data = pd.read_csv('data/rocket_parameters/GridFin/C_N_alpha_grid_fin.csv', skiprows=2, header=None)
    mach_cn_alpha = cn_alpha_data[0].values
    cn_alpha = cn_alpha_data[1].values
    min_mach_cn_alpha = np.min(mach_cn_alpha)
    max_mach_cn_alpha = np.max(mach_cn_alpha)
    f = interp1d(mach_cn_alpha, cn_alpha, kind='linear')
    min_cn_alpha = cn_alpha[np.argmin(mach_cn_alpha)]
    # Extrapolation slope
    sorted_indices = np.argsort(mach_cn_alpha)
    max_idx = sorted_indices[-1]
    second_max_idx = sorted_indices[-2]
    max_cn_alpha = cn_alpha[max_idx]
    slope = (cn_alpha[max_idx] - cn_alpha[second_max_idx]) / (mach_cn_alpha[max_idx] - mach_cn_alpha[second_max_idx])
    def cn_func(mach_val, alpha_val_rad):
        alpha_val_degrees = math.degrees(alpha_val_rad)
        if mach_val < min_mach_cn_alpha:
            cn_alpha = min_cn_alpha
            return cn_alpha * alpha_val_degrees
        elif mach_val <= max_mach_cn_alpha:
            cn_alpha = f(mach_val)
            return cn_alpha * alpha_val_degrees
        else:
            cn_alpha = max_cn_alpha + slope * (mach_val - max_mach_cn_alpha)
            return cn_alpha * alpha_val_degrees
    return cn_func

if __name__ == "__main__":
    ca_func = compile_grid_fin_Ca()
    cn_func = compile_grid_fin_Cn()
    print("Grid fin aerodynamics compiled")

    mach_vals = np.linspace(0, 5, 100)
    ca_vals = [ca_func(m) for m in mach_vals]
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(mach_vals, ca_vals)
    plt.title('Grid Fin Axial Force Coefficient (Ca) vs Mach Number')
    plt.xlabel('Mach Number')
    plt.ylabel('Ca')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    alpha_vals_deg = [2, 5, 10, 15]
    for alpha_deg in alpha_vals_deg:
        alpha_rad = math.radians(alpha_deg)
        cn_vals = [cn_func(m, alpha_rad) for m in mach_vals]
        plt.plot(mach_vals, cn_vals, label=f'Alpha = {alpha_deg}°')
    
    plt.title('Grid Fin Normal Force Coefficient (Cn) vs Mach Number')
    plt.xlabel('Mach Number')
    plt.ylabel('Cn')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/Sizing/grid_fin_aerodynamics.png')
    plt.close()