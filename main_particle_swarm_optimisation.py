from src.particle_swarm_optimisation.particle_swarm_optimisation import ParticleSubswarmOptimisation
import multiprocessing


cpu_count = multiprocessing.cpu_count()
print(f"Number of CPUs available: {cpu_count}")

flight_phase = 'landing_burn' # 'subsonic' or 'supersonic' or 'flip_over_boostbackburn' or 'ballistic_arc_descent'
particle_swarm_optimisation = ParticleSubswarmOptimisation(flight_phase= flight_phase,
                                                           save_interval = 5,
                                                           enable_wind = False,
                                                           stochastic_wind = False,
                                                           horiontal_wind_percentile = 50,
                                                           load_swarms = False,
                                                           use_multiprocessing = True,
                                                           num_processes = cpu_count - 1)
particle_swarm_optimisation()

# To run tensorboard:
# cd data/pso_saves/{flight_phase} 
# tensorboard --logdir=runs

# To view on another device:
# ngrok http 6006

'''
tmux new -s rocket
python main_particle_swarm_optimisation.py

# leave ctrl+b, then press d

# re-attach:
tmux attach -t rocket


'''