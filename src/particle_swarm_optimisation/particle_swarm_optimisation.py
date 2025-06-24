import os
import pickle
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
import json
from datetime import datetime

from src.envs.pso.env_wrapped_ea import pso_wrapped_env

from configs.evolutionary_algorithms_config import subsonic_pso_params, supersonic_pso_params, flip_over_boostbackburn_pso_params, ballistic_arc_descent_pso_params, landing_burn_pure_throttle_pso_params, landing_burn_pso_params

# Multithreading
def evaluate_worker_function(args):
    position, flight_phase, enable_wind, stochastic_wind, horiontal_wind_percentile = args
    model = pso_wrapped_env(flight_phase=flight_phase,
                          enable_wind=enable_wind,
                          stochastic_wind=stochastic_wind,
                          horiontal_wind_percentile=horiontal_wind_percentile)
    fitness = model.objective_function(position)
    model.reset()
    return fitness

class ParticleSwarmOptimisation:
    def __init__(self, flight_phase, enable_wind = False, stochastic_wind = False, horiontal_wind_percentile = 95):
        if flight_phase == 'subsonic':
            self.pso_params = subsonic_pso_params
        elif flight_phase == 'supersonic':
            self.pso_params = supersonic_pso_params
        elif flight_phase == 'flip_over_boostbackburn':
            self.pso_params = flip_over_boostbackburn_pso_params
        elif flight_phase == 'ballistic_arc_descent':
            self.pso_params = ballistic_arc_descent_pso_params
        elif flight_phase == 'landing_burn_pure_throttle':
            self.pso_params = landing_burn_pure_throttle_pso_params
        elif flight_phase == 'landing_burn':
            self.pso_params = landing_burn_pso_params

        self.model = pso_wrapped_env(flight_phase, enable_wind = enable_wind, stochastic_wind = stochastic_wind, horiontal_wind_percentile = horiontal_wind_percentile)

        self.pop_size = self.pso_params['pop_size']
        self.generations = self.pso_params['generations']
        self.w_start = self.pso_params['w_start']
        self.w_end = self.pso_params['w_end']
        self.c1 = self.pso_params['c1']
        self.c2 = self.pso_params['c2']

        self.bounds = self.model.bounds
        self.flight_phase = flight_phase

        self.best_fitness_array = []
        self.best_individual_array = []

        self.initialize_swarm()   
        self.w = self.w_start     

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.global_best_fitness_array = []
        self.global_best_position_array = []
        self.average_particle_fitness_array = []

    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.initialize_swarm()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.global_best_fitness_array = []
        self.global_best_position_array = []
        self.average_particle_fitness_array = []
        
    def initialize_swarm(self):
        swarm = []
        for _ in range(self.pop_size):
            position_array = []
            for bound in self.bounds:
                position = random.uniform(bound[0], bound[1])
                position_array.append(position)
            particle = {
                'position': np.array(position_array),
                'velocity': np.zeros(len(self.bounds)),
                'best_position': None,
                'best_fitness': float('inf')
            }
            swarm.append(particle)
        self.swarm = swarm

    def evaluate_particle(self, particle):
        individual = particle['position']
        fitness = self.model.objective_function(individual)
        self.model.reset()
        if fitness < particle['best_fitness']:
            particle['best_fitness'] = fitness
            particle['best_position'] = particle['position'].copy()
        return fitness

    def update_velocity(self, particle, global_best_position):
        inertia = self.w * particle['velocity']
        cognitive = self.c1 * np.random.rand() * (particle['best_position'] - particle['position'])
        social = self.c2 * np.random.rand() * (global_best_position - particle['position'])
        particle['velocity'] = inertia + cognitive + social

    def update_position(self, particle):
        particle['position'] += particle['velocity']
        for i in range(len(self.bounds)):
            if particle['position'][i] < self.bounds[i][0]:
                particle['position'][i] = self.bounds[i][0]
            elif particle['position'][i] > self.bounds[i][1]:
                particle['position'][i] = self.bounds[i][1]

    def weight_linear_decrease(self, generation):
        return self.w_start - (self.w_start - self.w_end) * generation / self.generations

    def run(self):
        # Create tqdm progress bar with dynamic description
        pbar = tqdm(range(self.generations), desc='Running Particle Swarm Optimisation')
        
        for generation in pbar:
            particle_fitnesses = []
            for i, particle in enumerate(self.swarm):
                fitness = self.evaluate_particle(particle)
                particle_fitnesses.append(fitness)
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle['position'].copy()

            average_particle_fitness = np.mean(particle_fitnesses)
            self.average_particle_fitness_array.append(average_particle_fitness)            

            self.w = self.weight_linear_decrease(generation)
            for i, particle in enumerate(self.swarm):
                self.update_velocity(particle, self.global_best_position)
                self.update_position(particle)

            
 
            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)

            if generation % 5 == 0 and generation != 0:
                self.plot_convergence()
            
            pbar.set_description(f"Particle Swarm Optimisation - Best Fitness: {self.global_best_fitness:.4e}")

            if generation % 50 == 0:
                self.save_swarm(f"swarm_state_gen_{generation}.pkl")

        return self.global_best_position, self.global_best_fitness

    def plot_convergence(self):
        # Is inverse fitness so we gotta flip it...

        if len(self.global_best_fitness_array) == 0:
            return
        generations = range(len(self.global_best_fitness_array))

        plot_dir = f'{self.base_save_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)
        
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        
        
        file_path = f'{plot_dir}/convergence.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(generations, -global_best_fitness_array, linewidth=2.5, label='Global Best Fitness', color='black')
        plt.plot(generations, -average_particle_fitness_array, linewidth=2.5, label='Overall Average Fitness', color='blue', alpha=0.7)
        for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
            subswarm_fitness_array = np.array(subswarm_fitness)
            plt.plot(generations, -subswarm_fitness_array, linewidth=2, 
                        label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        for i, subswarm_avg in enumerate(self.subswarm_avg_array):
            if len(subswarm_avg) > 0:
                subswarm_avg_array = np.array(subswarm_avg)
                plt.plot(generations, -subswarm_avg_array, linewidth=2, 
                            label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle=':')
        
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Particle SubSwarm Optimisation Convergence', fontsize=18)
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Plot last 10 generations
        last_10_generations_idx = generations[-10:]
        file_path = f'{plot_dir}/last_10_fitnesses.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        
        # Only plot if we have at least 10 generations
        if len(self.global_best_fitness_array) >= 10:
            last_10_best_fitness = np.array(self.global_best_fitness_array[-10:])
            plt.plot(last_10_generations_idx, -last_10_best_fitness, linewidth=2, label='Global Best Fitness', color='black')
            
            if hasattr(self, 'subswarm_best_fitness_array') and len(self.subswarm_best_fitness_array) > 0:
                for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                    if len(subswarm_fitness) >= 10:  # Only plot if we have enough data
                        last_10_subswarm_fitness = np.array(subswarm_fitness[-10:])
                        plt.plot(last_10_generations_idx, -last_10_subswarm_fitness, linewidth=2, 
                                 label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
            
            plt.xlabel('Generations', fontsize=16)
            plt.ylabel('Fitness', fontsize=16)
            plt.title('Particle SubSwarm Optimisation (Last 10 Generations)', fontsize=18)
            plt.legend(fontsize=12, loc='lower right')
            plt.grid(True)
            plt.savefig(file_path)        
        plt.close()
        
        self.plot_convergence_subplots()

    def plot_convergence_subplots(self):
        # Skip plotting if we don't have any data yet
        if len(self.global_best_fitness_array) == 0:
            return
            
        generations = range(len(self.global_best_fitness_array))
        plot_dir = f'{self.base_save_dir}/plots'
        
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
        plt.rcParams.update({'font.size': 14})
        ax1.plot(generations, -global_best_fitness_array, linewidth=2.5, 
                 label='Global Best Fitness', color='black')
        
        for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
            if len(subswarm_fitness) > 0:
                subswarm_fitness_array = np.array(subswarm_fitness)
                ax1.plot(generations, -subswarm_fitness_array, linewidth=2, 
                            label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        
        ax1.set_ylabel('Best Fitness', fontsize=16)
        ax1.set_title('Particle SubSwarm Optimisation - Best Fitness per Subswarm', fontsize=18)
        ax1.grid(True)
        ax1.legend(fontsize=12, loc='lower right')
        
        ax2.plot(generations, -average_particle_fitness_array, linewidth=2.5, 
                 label='Overall Average Fitness', color='blue')
        
        for i, subswarm_avg in enumerate(self.subswarm_avg_array):
            if len(subswarm_avg) > 0:
                subswarm_avg_array = np.array(subswarm_avg)
                ax2.plot(generations, -subswarm_avg_array, linewidth=2, 
                            label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle='--')
        
        ax2.set_xlabel('Generations', fontsize=16)
        ax2.set_ylabel('Average Fitness', fontsize=16)
        ax2.set_title('Particle SubSwarm Optimisation - Average Fitness per Subswarm', fontsize=18)
        ax2.grid(True)
        ax2.legend(fontsize=12, loc='lower right')
        
        plt.tight_layout()
        file_path = f'{plot_dir}/convergence_subplots.png'
        fig.savefig(file_path)
        plt.close(fig)

    def save_swarm(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self.swarm, f)
        print(f"Swarm state saved to {file_path}")

    def load_swarm(self, file_path):
        with open(file_path, 'rb') as f:
            self.swarm = pickle.load(f)
        print(f"Swarm state loaded from {file_path}")

        for particle in self.swarm:
            if particle['best_fitness'] < self.global_best_fitness:
                self.global_best_fitness = particle['best_fitness']
                self.global_best_position = particle['best_position'].copy()

class ParticleSubswarmOptimisation(ParticleSwarmOptimisation):
    def __init__(self,
                 flight_phase,
                 save_interval,
                 enable_wind = False,
                 stochastic_wind = False,
                 horiontal_wind_percentile = 50,
                 load_swarms = False,
                 use_multiprocessing = True,
                 num_processes = None):
        super().__init__(flight_phase, enable_wind = enable_wind, stochastic_wind = stochastic_wind, horiontal_wind_percentile = horiontal_wind_percentile)
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_pure_throttle']
        self.num_sub_swarms = self.pso_params["num_sub_swarms"]
        self.communication_freq = self.pso_params.get("communication_freq", 10)
        self.migration_freq = self.pso_params.get("migration_freq", 20)
        self.number_of_migrants = self.pso_params.get("number_of_migrants", 1)
        self.re_initialise_number_of_particles = self.pso_params.get("re_initialise_number_of_particles", 500)
        self.re_initialise_generation = self.pso_params.get("re_initialise_generation", 60)
        self.initialize_swarms()

        # for the multithreading option
        self.enable_wind = enable_wind
        self.stochastic_wind = stochastic_wind
        self.horiontal_wind_percentile = horiontal_wind_percentile
        self.flight_phase = flight_phase
        self.use_multiprocessing = use_multiprocessing
        self.num_processes = num_processes if num_processes else cpu_count()
        
        
        self.save_interval = save_interval
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.base_save_dir = f'data/pso_saves/{self.flight_phase}/run_{self.timestamp}'
        os.makedirs(self.base_save_dir, exist_ok=True)
        self.saves_dir = f'{self.base_save_dir}/saves'
        self.metrics_dir = f'{self.base_save_dir}/metrics'
        self.trajectory_dir = f'{self.base_save_dir}/trajectory_data'
        self.plots_dir = f'{self.base_save_dir}/plots'
        os.makedirs(self.saves_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.trajectory_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.save_swarm_dir = f'{self.saves_dir}/swarm.pkl'
        self.individual_dictionary_initial = self.model.mock_dictionary_of_opt_params
        self.save_config_to_json()

        if load_swarms:
            self.load_swarms()
        
    def parallel_evaluate(self, positions):
        results = []
        
        if self.use_multiprocessing:
            args_list = [(position, self.flight_phase, self.enable_wind, 
                         self.stochastic_wind, self.horiontal_wind_percentile) 
                         for position in positions]
            
            with Pool(processes=self.num_processes) as pool:
                results = pool.map(evaluate_worker_function, args_list)
        else:
            for position in positions:
                fitness = self.model.objective_function(position)
                self.model.reset()
                results.append(fitness)
            
        return results
            
    def reset(self):
        self.best_fitness_array = []
        self.best_individual_array = []
        self.best_fitness_array_array = []
        self.best_individual_array_array = []
        self.initialize_swarms()
        self.w = self.w_start

        self.global_best_position = None
        self.global_best_fitness = float('inf')

        self.subswarm_best_positions = []
        self.subswarm_best_fitnesses = []
        self.subswarm_best_fitness_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_best_position_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_avg_array = [[] for _ in range(self.num_sub_swarms)]

    def __call__(self):
        self.run()

    def __del__(self):
        pass

    def re_initialise_swarms(self):
        '''
        The purpose of this function is to re-initialise swarms at a set generation,
        such that the problem can be ran in a feasible time.
        The swarms will select the best performing particles from the previous generation,
        and then re-initialise the swarm with these particles.
        '''
        number_of_particle_per_swarm_new = self.re_initialise_number_of_particles // self.num_sub_swarms
        for swarm_idx, swarm in enumerate(self.swarms):
            best_particles = sorted(swarm, key=lambda x: x['best_fitness'])[:number_of_particle_per_swarm_new]
            self.swarms[swarm_idx] = best_particles
        print(f'Swarms re-initialised to {number_of_particle_per_swarm_new} particles each')


    def initialize_swarms(self):
        self.swarms = []
        sub_swarm_size = self.pop_size // self.num_sub_swarms
        
        self.subswarm_best_positions = []
        self.subswarm_best_fitnesses = []
        self.subswarm_best_fitness_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_best_position_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_avg_array = [[] for _ in range(self.num_sub_swarms)]
        for _ in range(self.num_sub_swarms):
            swarm = []
            for _ in range(sub_swarm_size):
                position_array = [random.uniform(bound[0], bound[1]) for bound in self.bounds]
                particle = {
                    'position': np.array(position_array),
                    'velocity': np.zeros(len(self.bounds)),
                    'best_position': None,
                    'best_fitness': float('inf')
                }
                swarm.append(particle)
            self.swarms.append(swarm)
            self.subswarm_best_positions.append(None)
            self.subswarm_best_fitnesses.append(float('inf'))

    def run(self):
        pbar = tqdm(range(self.generations), desc='Particle Swarm Optimisation with Subswarms')
        
        os.makedirs(f'data/pso_saves/{self.flight_phase}/metrics/', exist_ok=True)
        
        for generation in pbar:
            generation_metrics = {
                'generation': generation,
                'swarm_metrics': []
            }

            all_particle_fitnesses = []
            for swarm_idx, swarm in enumerate(self.swarms):
                all_particles_swarm_idx_fitnesses = []
                
                if self.use_multiprocessing:
                    positions = [p['position'] for p in swarm]
                    fitnesses = self.parallel_evaluate(positions)
                    for i, (particle, fitness) in enumerate(zip(swarm, fitnesses)):
                        all_particle_fitnesses.append(fitness)
                        all_particles_swarm_idx_fitnesses.append(fitness)
                        if fitness < particle['best_fitness']:
                            particle['best_fitness'] = fitness
                            particle['best_position'] = particle['position'].copy()
                        if fitness < self.subswarm_best_fitnesses[swarm_idx]:
                            self.subswarm_best_fitnesses[swarm_idx] = fitness
                            self.subswarm_best_positions[swarm_idx] = particle['position'].copy()
                else:
                    for particle in swarm:
                        fitness = self.evaluate_particle(particle)
                        all_particle_fitnesses.append(fitness)
                        all_particles_swarm_idx_fitnesses.append(fitness)
                        if fitness < particle['best_fitness']:
                            particle['best_fitness'] = fitness
                            particle['best_position'] = particle['position'].copy()
                        if fitness < self.subswarm_best_fitnesses[swarm_idx]:
                            self.subswarm_best_fitnesses[swarm_idx] = fitness
                            self.subswarm_best_positions[swarm_idx] = particle['position'].copy()

                # Metrics
                swarm_best_fitness = self.subswarm_best_fitnesses[swarm_idx]
                swarm_avg_fitness = np.mean(all_particles_swarm_idx_fitnesses)
                swarm_min_fitness = np.min(all_particles_swarm_idx_fitnesses)
                swarm_max_fitness = np.max(all_particles_swarm_idx_fitnesses)
                swarm_std_fitness = np.std(all_particles_swarm_idx_fitnesses)
                swarm_metrics = {
                    'swarm_idx': swarm_idx,
                    'best_fitness': swarm_best_fitness,
                    'avg_fitness': swarm_avg_fitness,
                    'min_fitness': swarm_min_fitness,
                    'max_fitness': swarm_max_fitness,
                    'std_fitness': swarm_std_fitness,
                    'num_particles': len(swarm)
                }
                generation_metrics['swarm_metrics'].append(swarm_metrics)

                # Update subswarm best fitness and position arrays
                self.subswarm_best_fitness_array[swarm_idx].append(self.subswarm_best_fitnesses[swarm_idx])
                self.subswarm_best_position_array[swarm_idx].append(self.subswarm_best_positions[swarm_idx])
                self.subswarm_avg_array[swarm_idx].append(np.mean(all_particles_swarm_idx_fitnesses))
                
            for i, fitness in enumerate(self.subswarm_best_fitnesses):
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = self.subswarm_best_positions[i].copy()
            average_particle_fitness = np.mean(all_particle_fitnesses)
            self.average_particle_fitness_array.append(average_particle_fitness)
            generation_metrics['global_best_fitness'] = self.global_best_fitness
            generation_metrics['global_avg_fitness'] = average_particle_fitness
            self.save_generation_metrics(generation_metrics, generation)
            self.w = self.weight_linear_decrease(generation)
            for swarm_idx, swarm in enumerate(self.swarms):
                for particle in swarm:
                    self.update_velocity_with_local_best(
                        particle, 
                        self.subswarm_best_positions[swarm_idx]
                    )
                    self.update_position(particle)

            if generation % self.communication_freq == 0 and generation > 0:
                self.share_information()

            if generation % self.migration_freq == 0 and generation > 0:
                self.migrate_particles()

            self.global_best_fitness_array.append(self.global_best_fitness)
            self.global_best_position_array.append(self.global_best_position)
            
            if generation % self.save_interval == 0 and generation != 0:
                self.plot_convergence()
                self.model.plot_results(self.global_best_position, self.plots_dir + '/')
                
                trajectory_data = self.collect_trajectory_data(self.global_best_position)
                self.save_trajectory_data(trajectory_data)

                self.save()
                self.save_results()
            if generation == self.re_initialise_generation:
                self.re_initialise_swarms()
            
            pbar.set_description(f"Particle Subswarm Optimisation - Best Fitness: {self.global_best_fitness:.6e}")
        
        return self.global_best_position, self.global_best_fitness

    def update_velocity_with_local_best(self, particle, local_best_position):
        inertia = self.w * particle['velocity']
        cognitive = self.c1 * np.random.rand() * (particle['best_position'] - particle['position'])
        social = self.c2 * np.random.rand() * (local_best_position - particle['position'])
        particle['velocity'] = inertia + cognitive + social

    def share_information(self):
        influence_factor = 0.3
        sharing_probability = 0.5
        best_swarm_idx = np.argmin(self.subswarm_best_fitnesses)
        best_swarm_position = self.subswarm_best_positions[best_swarm_idx]
        for i in range(self.num_sub_swarms):
            if i != best_swarm_idx and random.random() < sharing_probability:  # 50% chance to share
                self.subswarm_best_positions[i] = (
                    (1 - influence_factor) * self.subswarm_best_positions[i] + 
                    influence_factor * best_swarm_position
                )
                temp_particle = {
                    'position': self.subswarm_best_positions[i],
                    'velocity': np.zeros(len(self.bounds)),
                    'best_position': None,
                    'best_fitness': float('inf')
                }
                new_fitness = self.evaluate_particle(temp_particle)
                if new_fitness < self.subswarm_best_fitnesses[i]:
                    self.subswarm_best_fitnesses[i] = new_fitness

    def migrate_particles(self):
        for i in range(self.num_sub_swarms):
            if len(self.swarms[i]) > 1:
                for _ in range(self.number_of_migrants):
                    particle_index = random.randrange(len(self.swarms[i]))
                    particle_to_migrate = self.swarms[i][particle_index]
                    target_swarm_index = random.choice([j for j in range(self.num_sub_swarms) if j != i])
                    self.swarms[target_swarm_index].append(particle_to_migrate)
                    self.swarms[i].pop(particle_index)

    def plot_convergence(self):
        # Is inverse fitness so we gotta flip it...
        if len(self.global_best_fitness_array) == 0:
            return
        generations = range(len(self.global_best_fitness_array))
        plot_dir = f'{self.base_save_dir}/plots'
        os.makedirs(plot_dir, exist_ok=True)
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        
        file_path = f'{plot_dir}/convergence.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        plt.plot(generations, -global_best_fitness_array, linewidth=2.5, label='Global Best Fitness', color='black')
        plt.plot(generations, -average_particle_fitness_array, linewidth=2.5, label='Overall Average Fitness', color='blue', alpha=0.7)
        for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
            subswarm_fitness_array = np.array(subswarm_fitness)
            plt.plot(generations, -subswarm_fitness_array, linewidth=2, 
                        label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        for i, subswarm_avg in enumerate(self.subswarm_avg_array):
            if len(subswarm_avg) > 0:
                subswarm_avg_array = np.array(subswarm_avg)
                plt.plot(generations, -subswarm_avg_array, linewidth=2, 
                            label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle=':')
        plt.xlabel('Generations', fontsize=16)
        plt.ylabel('Fitness', fontsize=16)
        plt.title('Particle SubSwarm Optimisation Convergence', fontsize=18)
        plt.legend(fontsize=12, loc='lower right')
        plt.grid(True)
        plt.savefig(file_path)
        plt.close()

        # Plot last 10 generations
        last_10_generations_idx = generations[-10:]
        file_path = f'{plot_dir}/last_10_fitnesses.png'
        plt.figure(figsize=(12, 10))
        plt.rcParams.update({'font.size': 14})
        if len(self.global_best_fitness_array) >= 10:
            last_10_best_fitness = np.array(self.global_best_fitness_array[-10:])
            plt.plot(last_10_generations_idx, -last_10_best_fitness, linewidth=2, label='Global Best Fitness', color='black')
            for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
                if len(subswarm_fitness) >= 10:
                    last_10_subswarm_fitness = np.array(subswarm_fitness[-10:])
                    plt.plot(last_10_generations_idx, -last_10_subswarm_fitness, linewidth=2, 
                                label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
            plt.xlabel('Generations', fontsize=16)
            plt.ylabel('Fitness', fontsize=16)
            plt.title('Particle SubSwarm Optimisation (Last 10 Generations)', fontsize=18)
            plt.legend(fontsize=12, loc='lower right')
            plt.grid(True)
            plt.savefig(file_path)        
        plt.close()
        self.plot_convergence_subplots()

    def plot_convergence_subplots(self):
        if len(self.global_best_fitness_array) == 0:
            return
            
        generations = range(len(self.global_best_fitness_array))
        plot_dir = f'{self.base_save_dir}/plots'
        global_best_fitness_array = np.array(self.global_best_fitness_array)
        average_particle_fitness_array = np.array(self.average_particle_fitness_array)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)
        plt.rcParams.update({'font.size': 14})
        ax1.plot(generations, -global_best_fitness_array, linewidth=2.5, 
                 label='Global Best Fitness', color='black')
        for i, subswarm_fitness in enumerate(self.subswarm_best_fitness_array):
            if len(subswarm_fitness) > 0:  # Only plot if we have data
                subswarm_fitness_array = np.array(subswarm_fitness)
                ax1.plot(generations, -subswarm_fitness_array, linewidth=2, 
                            label=f'Subswarm {i+1} Best', alpha=0.8, linestyle='--')
        ax1.set_ylabel('Best Fitness', fontsize=16)
        ax1.set_title('Particle SubSwarm Optimisation - Best Fitness per Subswarm', fontsize=18)
        ax1.grid(True)
        ax1.legend(fontsize=12, loc='lower right')
        
        ax2.plot(generations, -average_particle_fitness_array, linewidth=2.5, 
                 label='Overall Average Fitness', color='blue')
        for i, subswarm_avg in enumerate(self.subswarm_avg_array):
            if len(subswarm_avg) > 0:  # Only plot if we have data
                subswarm_avg_array = np.array(subswarm_avg)
                ax2.plot(generations, -subswarm_avg_array, linewidth=2, 
                            label=f'Subswarm {i+1} Avg', alpha=0.8, linestyle='--')
        ax2.set_xlabel('Generations', fontsize=16)
        ax2.set_ylabel('Average Fitness', fontsize=16)
        ax2.set_title('Particle SubSwarm Optimisation - Average Fitness per Subswarm', fontsize=18)
        ax2.grid(True)
        ax2.legend(fontsize=12, loc='lower right')
        plt.tight_layout()
        file_path = f'{plot_dir}/convergence_subplots.png'
        fig.savefig(file_path)
        plt.close(fig)

    def save(self):
        with open(self.save_swarm_dir, 'wb') as f:
            pickle.dump(self.swarms, f)

    def load_swarms(self):
        #data/pso_saves/landing_burn/saves/swarm.pkl
        file_path = f'data/pso_saves/{self.flight_phase}/saves/swarm.pkl'
        with open(file_path, 'rb') as f:
            self.swarms = pickle.load(f)
        print(f"Subswarm states loaded from {file_path}")
        self.global_best_fitness_array = []
        self.global_best_position_array = []
        self.average_particle_fitness_array = []
        self.subswarm_best_fitness_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_best_position_array = [[] for _ in range(self.num_sub_swarms)]
        self.subswarm_avg_array = [[] for _ in range(self.num_sub_swarms)]
        
        self.global_best_fitness = float('inf')
        self.global_best_position = None
        self.subswarm_best_positions = [None for _ in range(self.num_sub_swarms)]
        self.subswarm_best_fitnesses = [float('inf') for _ in range(self.num_sub_swarms)]
        
        for swarm_idx, swarm in enumerate(self.swarms):
            for particle in swarm:
                if particle['best_fitness'] < self.subswarm_best_fitnesses[swarm_idx]:
                    self.subswarm_best_fitnesses[swarm_idx] = particle['best_fitness']
                    self.subswarm_best_positions[swarm_idx] = particle['best_position'].copy()
                if particle['best_fitness'] < self.global_best_fitness:
                    self.global_best_fitness = particle['best_fitness']
                    self.global_best_position = particle['best_position'].copy()
        
        self.global_best_fitness_array.append(self.global_best_fitness)
        self.global_best_position_array.append(self.global_best_position)
        
        for i in range(self.num_sub_swarms):
            self.subswarm_best_fitness_array[i].append(self.subswarm_best_fitnesses[i])
            self.subswarm_best_position_array[i].append(self.subswarm_best_positions[i])
            avg_fitness = np.mean([p['best_fitness'] for p in self.swarms[i]])
            self.subswarm_avg_array[i].append(avg_fitness)
        
        all_fitnesses = [p['best_fitness'] for swarm in self.swarms for p in swarm]
        self.average_particle_fitness_array.append(np.mean(all_fitnesses))

    def save_results(self):
        # Save results to CSV
        file_path = f'{self.base_save_dir}/particle_subswarm_optimisation_results.csv'
        try:
            existing_df = pd.read_csv(file_path, index_col=0)
            file_exists = True
        except (FileNotFoundError, pd.errors.EmptyDataError):
            file_exists = False
        
        column_titles = list(self.individual_dictionary_initial.keys()) + ['Best Fitness']

        if not file_exists:
            mock_params = [10e10] * (len(column_titles) - 1) 
            data = [['Particle Subswarm Optimisation'] + mock_params + [10e10]]
            df_columns = ['Algorithm'] + column_titles
            existing_df = pd.DataFrame(data, columns=df_columns)
            existing_df.set_index('Algorithm', inplace=True)
            existing_df.to_csv(file_path)

        best_solution = self.global_best_position
        best_value = self.global_best_fitness
        row_data = dict(zip(column_titles[:-1], best_solution))
        row_data[column_titles[-1]] = best_value
        existing_df.loc['Particle Subswarm Optimisation'] = row_data
        existing_df.to_csv(file_path)      
        self.save_fitness_history()
        
    def save_fitness_history(self):
        generations = list(range(len(self.global_best_fitness_array)))
        data = {
            'Generation': generations,
            'Global_Best_Fitness': self.global_best_fitness_array,
            'Average_Fitness': self.average_particle_fitness_array
        }
        for i in range(self.num_sub_swarms):
            if len(self.subswarm_best_fitness_array[i]) > 0:
                if len(self.subswarm_best_fitness_array[i]) < len(generations):
                    pad_length = len(generations) - len(self.subswarm_best_fitness_array[i])
                    padded_array = self.subswarm_best_fitness_array[i] + [None] * pad_length
                else:
                    padded_array = self.subswarm_best_fitness_array[i][:len(generations)]
                data[f'Subswarm_{i+1}_Best'] = padded_array
            
            if len(self.subswarm_avg_array[i]) > 0:
                if len(self.subswarm_avg_array[i]) < len(generations):
                    pad_length = len(generations) - len(self.subswarm_avg_array[i])
                    padded_array = self.subswarm_avg_array[i] + [None] * pad_length
                else:
                    padded_array = self.subswarm_avg_array[i][:len(generations)]
                data[f'Subswarm_{i+1}_Average'] = padded_array
        
        df = pd.DataFrame(data)
        file_path = f'{self.metrics_dir}/fitness_history.csv'
        df.to_csv(file_path, index=False)
        print(f"Fitness history saved to {file_path}")        

    def plot_results(self, individual):
        plots_dir = f'{self.base_save_dir}/plots'
        os.makedirs(plots_dir, exist_ok=True)
        trajectory_data = self.collect_trajectory_data(individual)
        self.save_trajectory_data(trajectory_data)
        self.model.individual_update_model(individual)
        # Commented out to save time for now.
        #universal_physics_plotter(self.model.env,
        #                         self.model.actor,
        #                         plots_dir + '/',
        #                         flight_phase=self.flight_phase,
        #                         type='pso')

    def collect_trajectory_data(self, individual):
        self.model.individual_update_model(individual)
        state = self.model.env.reset()

        trajectory_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'info': []
        }

        done_or_truncated = False
        while not done_or_truncated:
            action = self.model.actor.forward(state)
            next_state, reward, done, truncated, info = self.model.env.step(action)
            
            # Store data
            trajectory_data['states'].append(state.tolist() if hasattr(state, 'tolist') else state)
            trajectory_data['actions'].append(action.detach().numpy().tolist() if hasattr(action, 'detach') else action)
            trajectory_data['rewards'].append(reward)
            trajectory_data['info'].append(info)
            
            # Update for next iteration
            done_or_truncated = done or truncated
            state = next_state
            
        return trajectory_data

    def save_trajectory_data(self, trajectory_data):
        os.makedirs(self.trajectory_dir, exist_ok=True)
        states_df = pd.DataFrame(trajectory_data['states'])
        actions_df = pd.DataFrame(trajectory_data['actions'])
        rewards_df = pd.DataFrame(trajectory_data['rewards'], columns=['reward'])
        
        states_df.to_csv(f'{self.trajectory_dir}/states.csv', index=False)
        actions_df.to_csv(f'{self.trajectory_dir}/actions.csv', index=False)
        rewards_df.to_csv(f'{self.trajectory_dir}/rewards.csv', index=False)
        if trajectory_data['info']:
            flat_data = []
            for info in trajectory_data['info']:
                flat_info = {}
                def flatten_dict(d, prefix=''):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            flatten_dict(value, f"{prefix}{key}_")
                        else:
                            flat_info[f"{prefix}{key}"] = value
                flatten_dict(info)
                flat_data.append(flat_info)
            info_df = pd.DataFrame(flat_data)
            info_df.to_csv(f'{self.trajectory_dir}/info_data.csv', index=False)

    def save_generation_metrics(self, metrics, generation):
        for swarm_metric in metrics['swarm_metrics']:
            swarm_idx = swarm_metric['swarm_idx']
            swarm_df = pd.DataFrame([swarm_metric])
            swarm_df['generation'] = generation
            swarm_df['global_best_fitness'] = metrics['global_best_fitness']
            swarm_df['global_avg_fitness'] = metrics['global_avg_fitness']
            swarm_file = f'{self.metrics_dir}/subswarm_{swarm_idx}_metrics.csv'
            if generation == 0:
                swarm_df.to_csv(swarm_file, index=False, mode='w')
            else:
                swarm_df.to_csv(swarm_file, index=False, mode='a', header=False)
        
        global_df = pd.DataFrame({
            'generation': [generation],
            'global_best_fitness': [metrics['global_best_fitness']],
            'global_avg_fitness': [metrics['global_avg_fitness']]
        })
        global_file = f'{self.metrics_dir}/global_metrics.csv'
        if generation == 0:
            global_df.to_csv(global_file, index=False, mode='w')
        else:
            global_df.to_csv(global_file, index=False, mode='a', header=False)

    def save_config_to_json(self):
        config = self.pso_params.copy()
        config.update({
            'flight_phase': self.flight_phase,
            'enable_wind': self.enable_wind,
            'stochastic_wind': self.stochastic_wind,
            'horiontal_wind_percentile': self.horiontal_wind_percentile,
            'use_multiprocessing': self.use_multiprocessing,
            'num_processes': self.num_processes,
            'save_interval': self.save_interval,
            'timestamp': self.timestamp
        })
        with open(f'{self.base_save_dir}/pso_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {self.base_save_dir}/pso_config.json")        