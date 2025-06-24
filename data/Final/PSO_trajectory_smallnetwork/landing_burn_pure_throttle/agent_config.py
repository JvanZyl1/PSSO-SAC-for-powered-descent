config_landing_burn = {
    'sac' : {
        'hidden_dim_actor': 116, # was 254
        'number_of_hidden_layers_actor': 3,
        'hidden_dim_critic': 512,
        'number_of_hidden_layers_critic': 4,
        'temperature_initial': 0.9,
        'gamma': 0.95,
        'tau': 0.1,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 5000,
        'trajectory_length': 2,
        'batch_size': 512,
        'critic_learning_rate': 1e-3,
        'actor_learning_rate': 1e-3,
        'temperature_learning_rate': 6e-3,
        'critic_grad_max_norm': 10.0,
        'actor_grad_max_norm': 10.0,
        'temperature_grad_max_norm': 1.0,
        'max_std': 1.0,
        'l2_reg_coef': 0.008,
        'expected_updates_to_convergence': 50000
    },
    'td3' : {
        'hidden_dim_actor': 253,
        'number_of_hidden_layers_actor': 3,
        'hidden_dim_critic': 253,
        'number_of_hidden_layers_critic': 3,
        'gamma': 0.95,
        'tau': 0.01,
        'alpha_buffer': 0.6,
        'beta_buffer': 0.4,
        'beta_decay_buffer': 0.99,
        'buffer_size': 50000, # 25000 -> 50000
        'trajectory_length': 2,
        'batch_size': 512,
        'critic_learning_rate': 1e-4, # Also for critic warm-up
        'actor_learning_rate': 1e-3, # from 1e-7 -> 1e-5
        'critic_grad_max_norm': 10.0,
        'actor_grad_max_norm': 10.0,
        'policy_noise': 0.1/3,  # Divide maxstd by 3 to still get the Gaussian feel as most vals within 3 std.
        'noise_clip': 0.1,      # Essentially the max std * normal distribution.
        'policy_delay': 2,
        'l2_reg_coef': 0.003,    # L2 regularization coefficient
        'expected_updates_to_convergence': 50000
    },
    'num_episodes': 5650,
    'critic_warm_up_steps': 250000,
    'pre_train_critic_learning_rate' : 1e-5, # from loading from pso, not used atm.
    'pre_train_critic_batch_size' : 128,
    'update_agent_every_n_steps' : 1, # was 10
    'critic_warm_up_early_stopping_loss' : 1e-9,
    'priority_update_interval': 50,
    'max_added_deviation_filling' : 0.75
}