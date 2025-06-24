from src.trainers.trainer_rocket_SAC import RocketTrainer_ReinforcementLearning

trainer = RocketTrainer_ReinforcementLearning(flight_phase = 'landing_burn_pure_throttle',
                             load_from = None,
                             load_buffer_bool= False,
                             save_interval = 50,
                             pre_train_critic_bool = False,
                             buffer_type = 'uniform',
                             rl_type = 'sac', # sac or td3
                             enable_wind = False,
                             stochastic_wind = False,
                             horiontal_wind_percentile = 50)
trainer()