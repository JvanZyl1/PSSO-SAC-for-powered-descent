import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import math

def load_velocity_data(file_path='data/TiltAngle/RETALT/RETALT_Vx_Vy.csv'):
    """Load velocity data from CSV file and return cleaned arrays."""
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]
    vx = df['X'].values
    vy = df['Y'].values
    
    return vx, vy

def create_flight_path_angle_function():
    vx_raw, vy_raw = load_velocity_data()
    vx = np.maximum(vx_raw, 0)
    vy = np.maximum(vy_raw, 0)
    flight_path_angles = np.arctan2(vy, vx)
    sorted_indices = np.argsort(vy)
    vy_sorted = vy[sorted_indices]
    angles_sorted = flight_path_angles[sorted_indices]
    unique_indices = np.concatenate(([True], np.diff(vy_sorted) > 0))
    vy_unique = vy_sorted[unique_indices]
    angles_unique = angles_sorted[unique_indices]
    flight_path_angle_func = interp1d(
        vy_unique, 
        angles_unique, 
        kind='linear', 
        bounds_error=False, 
        fill_value=(angles_unique[0], angles_unique[-1])
    )
    
    return flight_path_angle_func

def get_flight_path_angle(vy):
    angle_func = create_flight_path_angle_function()
    vy = max(0, vy)
    return angle_func(vy)

def get_flight_path_angle_degrees(vy):
    if vy <= 100:
        return 90.0
    else:
        return math.degrees(get_flight_path_angle(vy))

if __name__ == "__main__":
    test_vy_values = [0, 1,2,3,100, 110, 120, 130, 140, 150, 200, 300, 400, 500, 600, 700, 800, 900]
    
    print("Vy (m/s) | Flight Path Angle (degrees)")
    print("-" * 40)
    
    for vy in test_vy_values:
        angle_deg = get_flight_path_angle_degrees(vy)
        print(f"{vy:7.1f} | {angle_deg:7.2f}")
