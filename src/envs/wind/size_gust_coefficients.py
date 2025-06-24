import csv

def size_gust_coefficients():
    # read sizing results
    sizing_results = {}
    with open('data/rocket_parameters/sizing_results.csv', 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            sizing_results[row[0]] = row[2]

    burnout_mass = (float(sizing_results['Stage 1 Mass']) - float(sizing_results['Actual propellant mass stage 1'])) * 1000.0 # [kg]
    frontal_area = float(sizing_results['Rocket frontal area']) # [m^2]
    a_max_gust = 0.5 * 9.81 # 0.5 g's
    rho_0 = 1.225 # [kg/m^3]

    V_g_max = 6.0 # [m/s] roughly
    V_constant_max = 10 # [m/s]
    V_wind_max = V_constant_max + V_g_max

    C_gust_x = 2 * burnout_mass * a_max_gust / (rho_0 * V_wind_max**2 * frontal_area)
    C_gust_y = 0.0 # not used
    return C_gust_x, C_gust_y


if __name__ == '__main__':
    C_gust_x, C_gust_y = size_gust_coefficients()
    print(C_gust_x, C_gust_y)

