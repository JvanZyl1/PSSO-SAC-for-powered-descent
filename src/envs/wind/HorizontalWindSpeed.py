import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def extract_horizontal_wind_data():
    file_path = 'data/Wind/horizontal_wind.csv'
    with open(file_path, 'r') as f:
        lines = f.readlines()
    header_line = lines[0].strip().split(',')
    percentiles = []
    for item in header_line:
        if item and not item.isspace():
            percentiles.append(item)
    data_lines = lines[2:]
    wind_data = {}
    for percentile in percentiles:
        wind_data[percentile] = {'wind_speed': [], 'altitude_km': []}
    for line in data_lines:
        if not line.strip():  # Skip empty lines
            continue
            
        values = line.strip().split(',')
        if len(values) < len(percentiles) * 2:
            continue  # Skip incomplete lines
            
        for i, percentile in enumerate(percentiles):
            try:
                x_index = i * 2
                y_index = i * 2 + 1
                
                if x_index < len(values) and y_index < len(values):
                    wind_speed = float(values[x_index])
                    altitude_km = float(values[y_index])
                    
                    wind_data[percentile]['wind_speed'].append(wind_speed)
                    wind_data[percentile]['altitude_km'].append(altitude_km)
            except (ValueError, IndexError) as e:
                pass  # Skip invalid values
    for percentile in percentiles:
        wind_data[percentile]['wind_speed'] = np.array(wind_data[percentile]['wind_speed'])
        wind_data[percentile]['altitude_km'] = np.array(wind_data[percentile]['altitude_km'])
    return wind_data, percentiles

def compile_horizontal_fixed_wind(percentile):
    wind_data, _ = extract_horizontal_wind_data()
    
    if percentile in wind_data:
        wind_speed = wind_data[percentile]['wind_speed']
        altitude_km = wind_data[percentile]['altitude_km']
    else:
        wind_speed, altitude_km = interpolate_percentile(wind_data, percentile)
    
    sort_idx = np.argsort(altitude_km)
    altitude_km = altitude_km[sort_idx]
    wind_speed = wind_speed[sort_idx]
    
    # Create the interpolation function
    def wind_at_altitude(altitude_m):
        altitude_km_input = altitude_m / 1000.0
        interpolator = interp1d(
            altitude_km, 
            wind_speed, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(wind_speed[0], wind_speed[-1])
        )
        
        return interpolator(altitude_km_input)
    
    return wind_at_altitude

def interpolate_percentile(wind_data, requested_percentile):
    if isinstance(requested_percentile, str) and "_percentile" in requested_percentile:
        req_perc_value = float(requested_percentile.split('_')[0])
    else:
        req_perc_value = float(requested_percentile)
        requested_percentile = f"{req_perc_value}_percentile"
    
    available_percentiles = list(wind_data.keys())
    available_perc_values = [float(p.split('_')[0]) for p in available_percentiles]
    
    idx = np.searchsorted(available_perc_values, req_perc_value)
    if idx == 0:
        lower_perc = available_percentiles[0]
        upper_perc = available_percentiles[0]
        weight = 1.0
    elif idx == len(available_perc_values):
        lower_perc = available_percentiles[-1]
        upper_perc = available_percentiles[-1]
        weight = 0.0
    else:
        lower_perc = available_percentiles[idx-1]
        upper_perc = available_percentiles[idx]
        
        lower_val = available_perc_values[idx-1]
        upper_val = available_perc_values[idx]
        weight = (req_perc_value - lower_val) / (upper_val - lower_val)
    
    lower_alt = wind_data[lower_perc]['altitude_km']
    upper_alt = wind_data[upper_perc]['altitude_km']
    
    all_altitudes = np.unique(np.concatenate([lower_alt, upper_alt]))
    
    lower_interp = interp1d(lower_alt, wind_data[lower_perc]['wind_speed'], 
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    upper_interp = interp1d(upper_alt, wind_data[upper_perc]['wind_speed'],
                          kind='linear', bounds_error=False, fill_value='extrapolate')
    
    lower_speeds = lower_interp(all_altitudes)
    upper_speeds = upper_interp(all_altitudes)
    
    interpolated_speeds = lower_speeds * (1 - weight) + upper_speeds * weight
    
    return interpolated_speeds, all_altitudes

def plot_horizontal_fixed_wind():
    wind_data, percentiles = extract_horizontal_wind_data()

    wind_interpolators = {}
    for percentile in percentiles:
        wind_interpolators[percentile] = compile_horizontal_fixed_wind(percentile)

    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, percentile in enumerate(percentiles):
        wind_speed = wind_data[percentile]['wind_speed']
        altitude_km = wind_data[percentile]['altitude_km']
        altitude_m = altitude_km * 1000
        min_alt_m = np.min(altitude_m)
        max_alt_m = np.max(altitude_m)
        altitude_grid_m = np.linspace(min_alt_m, max_alt_m, 1000)
        interpolator = wind_interpolators[percentile]
        interpolated_wind_speed = interpolator(altitude_grid_m)
        
        label = f"{percentile}"
        color = colors[i % len(colors)]
        plt.plot(wind_speed, altitude_m, 'o', markersize=5, alpha=0.5, color=color)
        plt.plot(interpolated_wind_speed, altitude_grid_m, '-', linewidth=2, label=label, color=color)
    plt.xlabel('Wind Speed (m/s)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.title('Horizontal Wind Speed vs Altitude - All Percentiles', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('results/disturbance/horizontal_wind/horizontal_wind_all_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 10))
    for i, percentile in enumerate(percentiles):
        plt.subplot(2, 3, i+1)
        wind_speed = wind_data[percentile]['wind_speed']
        altitude_km = wind_data[percentile]['altitude_km']
        altitude_m = altitude_km * 1000
        min_alt_m = np.min(altitude_m)
        max_alt_m = np.max(altitude_m)
        altitude_grid_m = np.linspace(min_alt_m, max_alt_m, 1000)
        
        interpolator = wind_interpolators[percentile]
        interpolated_wind_speed = interpolator(altitude_grid_m)
        
        color = colors[i % len(colors)]
        plt.plot(wind_speed, altitude_m, 'o', markersize=5, alpha=0.7, label='Data points', color=color)
        plt.plot(interpolated_wind_speed, altitude_grid_m, '-', linewidth=2, label='Interpolated', color=color)
        
        plt.xlabel('Wind Speed (m/s)', fontsize=10)
        plt.ylabel('Altitude (m)', fontsize=10)
        plt.title(f"{percentile}", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig('results/disturbance/horizontal_wind/horizontal_wind_individual_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close() 
    
    print("Example wind speeds at different altitudes (50_percentile):")
    test_altitudes = [1000, 5000, 10000, 20000, 50000]  # meters
    for alt in test_altitudes:
        wind_speed = wind_interpolators['50_percentile'](alt)
        print(f"Altitude: {alt} m, Wind Speed: {wind_speed:.2f} m/s")

def test_percentile_interpolation():
    test_percentiles = ['50_percentile', '53_percentile', '75_percentile', '85_percentile', '90_percentile']
    
    interpolators = {}
    for percentile in test_percentiles:
        interpolators[percentile] = compile_horizontal_fixed_wind(percentile)
    
    test_altitudes = [1000, 10000, 50000]  # meters
    
    for percentile, interpolator in interpolators.items():
        print(f"\nWind speeds for {percentile}:")
        for altitude in test_altitudes:
            wind_speed = interpolator(altitude)
            print(f"  At {altitude} m: {wind_speed:.2f} m/s")

def plot_horizontal_wind():
    wind_data, percentiles = extract_horizontal_wind_data()
    original_percentiles = percentiles.copy()
    
    all_percentiles = original_percentiles.copy()
    all_percentiles.extend(['53_percentile', '60_percentile', '80_percentile', '85_percentile'])
    
    plt.figure(figsize=(12, 8))
    
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown', 'black', 'cyan', 'magenta']
    for i, percentile in enumerate(all_percentiles):
        interpolator = compile_horizontal_fixed_wind(percentile)
        
        if percentile in original_percentiles:
            wind_speed = wind_data[percentile]['wind_speed']
            altitude_km = wind_data[percentile]['altitude_km']
            altitude_m = altitude_km * 1000
            
            color = colors[i % len(colors)]
            plt.plot(wind_speed, altitude_m, 'o', markersize=5, alpha=0.5, color=color)
        
        max_alt_m = 80000  #  maximum altitude
        altitude_grid_m = np.linspace(0, max_alt_m, 1000)
        
        interpolated_wind_speed = interpolator(altitude_grid_m)
        
        linestyle = '-' if percentile in original_percentiles else '--'
        label = f"{percentile}"
        color = colors[i % len(colors)]
        plt.plot(interpolated_wind_speed, altitude_grid_m, linestyle, linewidth=2, 
                label=label, color=color)
    
    plt.xlabel('Wind Speed (m/s)', fontsize=12)
    plt.ylabel('Altitude (m)', fontsize=12)
    plt.title('Horizontal Wind Speed vs Altitude - With Interpolated Percentiles', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('results/disturbance/horizontal_wind/horizontal_wind_all_percentiles.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_percentile_interpolation()
    plot_horizontal_wind()
    plot_horizontal_fixed_wind()