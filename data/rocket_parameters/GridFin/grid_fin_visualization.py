import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load data
cd_data = pd.read_csv('data/rocket_parameters/GridFin/C_D_grid_fin.csv', header=None)
cn_alpha_data = pd.read_csv('data/rocket_parameters/GridFin/C_N_alpha_grid_fin.csv', skiprows=2, header=None)
mach_cd = cd_data[0].values
ca = cd_data[1].values  # C_D is approximately C_a

mach_cn = cn_alpha_data[0].values
cn_alpha = cn_alpha_data[1].values

min_mach_cd = np.min(mach_cd)
max_mach_cd = np.max(mach_cd)
min_mach_cn = np.min(mach_cn)
max_mach_cn = np.max(mach_cn)


def ca_extrapolate(mach_vals):
    result = np.zeros_like(mach_vals, dtype=float)
    for i, m in enumerate(mach_vals):
        if m < min_mach_cd:
            # For values below minimum, use the value at minimum Mach
            min_idx = np.argmin(mach_cd)
            result[i] = ca[min_idx]
        else:
            f = interp1d(mach_cd, ca, kind='linear', fill_value="extrapolate")
            result[i] = f(m)
    return result

def cn_alpha_extrapolate(mach_vals):
    result = np.zeros_like(mach_vals, dtype=float)
    sorted_indices = np.argsort(mach_cn)
    max_idx = sorted_indices[-1]
    second_max_idx = sorted_indices[-2]
    slope = (cn_alpha[max_idx] - cn_alpha[second_max_idx]) / (mach_cn[max_idx] - mach_cn[second_max_idx])
    
    for i, m in enumerate(mach_vals):
        if m < min_mach_cn:
            min_idx = np.argmin(mach_cn)
            result[i] = cn_alpha[min_idx]
        elif m <= max_mach_cn:
            f = interp1d(mach_cn, cn_alpha, kind='linear')
            result[i] = f(m)
        else:
            result[i] = cn_alpha[max_idx] + slope * (m - mach_cn[max_idx])
    
    return result

mach_range = np.linspace(0, 4.5, 500)
ca_extrapolated = ca_extrapolate(mach_range)
cn_alpha_extrapolated = cn_alpha_extrapolate(mach_range)

plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(mach_cd, ca, 'o', color='blue', label='Data')
plt.plot(mach_range, ca_extrapolated, '-', color='blue', label='Extrapolation')
plt.axvline(x=min_mach_cd, linestyle='--', color='gray', label=f'Min Mach={min_mach_cd:.2f}')
plt.axvline(x=max_mach_cd, linestyle='--', color='black', label=f'Max Mach={max_mach_cd:.2f}')
plt.grid(True)
plt.xlabel('Mach Number')
plt.ylabel(r'C_a')
plt.title('Grid Fin Axial Force Coefficient')
plt.xlim(0, 4.5)
plt.legend()
plt.subplot(2, 2, 2)
plt.plot(mach_cn, cn_alpha, 'o', color='red', label='Data')
plt.plot(mach_range, cn_alpha_extrapolated, '-', color='red', label='Extrapolation')
plt.axvline(x=min_mach_cn, linestyle='--', color='gray', label=f'Min Mach={min_mach_cn:.2f}')
plt.axvline(x=max_mach_cn, linestyle='--', color='black', label=f'Max Mach={max_mach_cn:.2f}')
plt.grid(True)
plt.xlabel('Mach Number')
plt.ylabel(r'C_n_alpha')
plt.title('Grid Fin Normal Force Coefficient Derivative')
plt.xlim(0, 4.5)
plt.legend()

plt.subplot(2, 1, 2)

mach_targets = [0.5, 1.0, 2.0, 3.0, 4.0]
alpha_range = np.linspace(-15, 15, 100)  # Alpha range in degrees

colors = ['blue', 'red', 'green', 'purple', 'orange']
for i, mach in enumerate(mach_targets):
    cn_alpha_value = cn_alpha_extrapolate(np.array([mach]))[0]
    cn = cn_alpha_value * np.deg2rad(alpha_range)
    plt.plot(alpha_range, cn, label=f'Mach {mach}', color=colors[i])

plt.grid(True)
plt.xlabel(r'Angle of Attack ($\alpha$) [$^{\circ}$]', fontsize=14)
plt.ylabel(r'Normal Force Coefficient ($C_n$)', fontsize=14)
plt.title('Grid Fin Normal Force Coefficient vs Angle of Attack', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig('data/rocket_parameters/GridFin/grid_fin_aerodynamics.png', dpi=300)
plt.show() 