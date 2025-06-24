import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import cont2discrete

class VonKarmanFilter:
    def __init__(self, L: float, sigma: float, V: float, dt: float):
        self.L = L
        self.sigma = sigma
        self.V = V
        self.dt = dt
        self._init_filter()

    def _init_filter(self):
        omega0 = self.V / self.L
        zeta = 1.0 / math.sqrt(2.0)
        scale = math.sqrt(math.pi / (2.0 * omega0**3))
        A_c = np.array([[0.0, 1.0],
                        [-omega0**2, -2.0 * zeta * omega0]])
        B_c = np.array([[0.0],
                        [self.sigma * scale]])
        C_c = np.array([[0.0, 1.0]])
        D_c = np.zeros((1, 1))
        A_d, B_d, C_d, D_d, _ = cont2discrete((A_c, B_c, C_c, D_c), self.dt)
        self.Ad = A_d
        self.Bd = B_d.flatten()
        self.Cd = C_d
        self.state = np.zeros(2)

    def step(self) -> float:
        w = np.random.randn()
        self.state = self.Ad @ self.state + self.Bd * w
        return float(self.Cd @ self.state)

    def reset(self):
        self._init_filter()

class VKDisturbanceGenerator:
    def __init__(self, dt: float, V: float):
        self.L_u = 100.0
        self.L_v = 30.0
        self.sigma_u_min, self.sigma_u_max = 0.5, 4.0
        self.sigma_v_min, self.sigma_v_max = 0.5, 2.0
        self.V = V
        self.dt = dt
        self.u_filter, self.v_filter = self._new_filters()

        self.log_data = {
            'L_u': self.L_u,
            'L_v': self.L_v,
            'sigma_u': self.sigma_u,
            'sigma_v': self.sigma_v,
            'gust_u': [],
            'gust_v': []}


    def _new_filters(self):
        # sample noise intensities
        sigma_u = random.uniform(self.sigma_u_min, (self.sigma_u_min + self.sigma_u_max) / 2)
        sigma_v = random.uniform((self.sigma_v_min + self.sigma_v_max) / 2, self.sigma_v_max)
        # create filters
        u = VonKarmanFilter(self.L_u, sigma_u, self.V, self.dt)
        v = VonKarmanFilter(self.L_v, sigma_v, self.V, self.dt)
        self.sigma_u, self.sigma_v = sigma_u, sigma_v
        return u, v
    
    def return_dict_characteristics(self):
        return {
            'L_u': self.L_u,
            'L_v': self.L_v,
            'sigma_u': self.sigma_u,
            'sigma_v': self.sigma_v
        }

    def __call__(self):
        gust_u = self.u_filter.step()
        gust_v = self.v_filter.step()
        
        self.log_data['gust_u'].append(gust_u)
        self.log_data['gust_v'].append(gust_v)
        
        return gust_u, gust_v
    
    def reset(self):
        np.random.seed(None)
        self.u_filter, self.v_filter = self._new_filters()
        self.log_data = {
            'L_u': self.L_u,
            'L_v': self.L_v,
            'sigma_u': self.sigma_u,
            'sigma_v': self.sigma_v,
            'gust_u': [],
            'gust_v': []}
        
    def plot_disturbance_generator(self, save_path):
        plot_horizontal_only = True
        time = np.arange(0, len(self.log_data['gust_u'])) * self.dt
        if not plot_horizontal_only:
            plt.figure(figsize=(20,15))
            plt.suptitle('Von Karman Disturbance Generator', fontsize=24)
            gs = gridspec.GridSpec(1,2, width_ratios=[1,1])

            ax1 = plt.subplot(gs[0])
            ax1.plot(time, self.log_data['gust_v'], linewidth=4, color='blue', label='Vertical')
            ax1.set_xlabel('Time [s]', fontsize=20)
            ax1.set_ylabel('Gust [m/s]', fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=18)
            ax1.set_title('Vertical', fontsize=20)
            ax1.grid(True)

            ax2 = plt.subplot(gs[1])
            ax2.plot(time, self.log_data['gust_u'], linewidth=4, color='blue', label='Horizontal')
            ax2.set_xlabel('Time [s]', fontsize=20)
            ax2.set_ylabel('', fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=18)
            ax2.set_title('Horizontal', fontsize=20)
            ax2.grid(True)
            plt.savefig(save_path + 'VonKarmenDisturbanceGenerator.png')
            plt.close()
        else:
            plt.figure(figsize=(10,5))
            plt.plot(time, self.log_data['gust_u'], linewidth=4, color='blue', label='Horizontal')
            plt.xlabel('Time [s]', fontsize=20)
            plt.ylabel('Gust [m/s]', fontsize=20)
            plt.tick_params(axis='both', which='major', labelsize=18)
            plt.title('Horizontal', fontsize=20)
            plt.grid(True)
            plt.savefig(save_path + 'VonKarmenDisturbanceGenerator.png')
            plt.close()
def compile_disturbance_generator(dt : float):
    V = 42 # mock value
    return VKDisturbanceGenerator(dt, V)

def test_disturbance_generator_subsonic():
    dt = 0.01
    flight_phase = 'landing_burn_pure_throttle'
    disturbance_generator = compile_disturbance_generator(dt)
    time = np.arange(0, 100, dt)
    for t in time:
        gust_u, gust_v = disturbance_generator()
    disturbance_generator.plot_disturbance_generator('results/disturbance/')

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    print(sys.path)
    test_disturbance_generator_subsonic()



