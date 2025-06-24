from src.trainers.trainers import TrainerRL
from src.envs.universal_physics_plotter import universal_physics_plotter
from configs.agent_config import config_subsonic, config_supersonic, config_flip_over_boostbackburn, config_ballistic_arc_descent, config_landing_burn
from src.agents.soft_actor_critic import SoftActorCritic
from src.agents.td3 import TD3
from src.agents.functions.load_agent import load_sac, load_td3
from src.envs.rl.env_wrapped_rl import rl_wrapped_env as env
from src.particle_swarm_optimisation.network_loader import load_pso_actor
from src.critic_pre_train.pre_train_critic import pre_train_critic_from_pso_experiences
from src.envs.supervisory.agent_load_supervisory import load_supervisory_actor
from src.agents.functions.buffers import PERBuffer
import jax.numpy as jnp

class TrainerEndo(TrainerRL):
    def __init__(self,
                 env,
                 agent,
                 flight_phase : str,
                 num_episodes: int,
                 save_interval: int = 10,
                 critic_warm_up_steps: int = 0,
                 critic_warm_up_early_stopping_loss: float = 0.0,
                 load_buffer_from_experiences_bool : bool = False,
                 update_agent_every_n_steps: int = 10,
                 priority_update_interval: int = 25,
                 buffer_save_interval: int = 100):
        super(TrainerEndo, self).__init__(env, agent, flight_phase, num_episodes, save_interval, critic_warm_up_steps, critic_warm_up_early_stopping_loss, load_buffer_from_experiences_bool, update_agent_every_n_steps, priority_update_interval)
        self.buffer_save_interval = buffer_save_interval

    def test_env(self):
        if self.flight_phase in ['landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
            reward_total, y_array = universal_physics_plotter(self.env,
                                                              self.agent,
                                                              self.agent.save_path,
                                                              flight_phase = self.env.flight_phase,
                                                              type = 'rl')
            return reward_total, y_array
        else:
            universal_physics_plotter(self.env,
                                  self.agent,
                                  self.agent.save_path,
                                  flight_phase = self.env.flight_phase,
                                  type = 'rl')
        
class RocketTrainer_ReinforcementLearning:
    def __init__(self,
                 flight_phase : str,
                 save_interval : int = 10,
                 load_from : str = 'None',
                 load_buffer_bool : bool = False,
                 pre_train_critic_bool : bool = False,
                 buffer_type : str = 'uniform',
                 rl_type : str = 'sac',
                 enable_wind : bool = False,
                 stochastic_wind : bool = True,
                 horiontal_wind_percentile : int = 50,
                 shared_buffer = None,
                 buffer_save_interval : int = 100):
        assert not enable_wind or stochastic_wind, "If enable_wind is True, stochastic_wind must also be True"
        assert rl_type in ['sac', 'td3']
        assert flight_phase in ['subsonic', 'supersonic', 'flip_over_boostbackburn', 'ballistic_arc_descent', 'landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']
        self.rl_type = rl_type
        self.flight_phase = flight_phase
        self.shared_buffer = shared_buffer
        self.buffer_save_interval = buffer_save_interval

        if flight_phase == 'subsonic':
            self.agent_config = config_subsonic
        elif flight_phase == 'supersonic':
            self.agent_config = config_supersonic
        elif flight_phase == 'flip_over_boostbackburn':
            self.agent_config = config_flip_over_boostbackburn
        elif flight_phase == 'ballistic_arc_descent':
            self.agent_config = config_ballistic_arc_descent
        elif flight_phase in ['landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
            self.agent_config = config_landing_burn
        if rl_type == 'sac':
            self.env = env(flight_phase = flight_phase,
                           enable_wind = enable_wind,
                           stochastic_wind = stochastic_wind,
                           horiontal_wind_percentile = horiontal_wind_percentile,
                           trajectory_length = self.agent_config['sac']['trajectory_length'],
                           discount_factor = self.agent_config['sac']['gamma'])
        elif rl_type == 'td3':
            self.env = env(flight_phase = flight_phase,
                           enable_wind = enable_wind,
                           stochastic_wind = stochastic_wind,
                           horiontal_wind_percentile = horiontal_wind_percentile,
                           trajectory_length = self.agent_config['td3']['trajectory_length'],
                           discount_factor = self.agent_config['td3']['gamma'])

        if load_from == 'pso':
            self.load_agent_from_pso()
        elif load_from == 'rl':
            self.load_agent_from_rl()
        elif load_from == 'supervisory':
            self.load_agent_from_supervisory()
        else:
            if rl_type == 'sac':
                self.agent = SoftActorCritic(
                    state_dim=self.env.state_dim,
                    action_dim=self.env.action_dim,
                    flight_phase = self.flight_phase,
                    **self.agent_config['sac'])
            elif rl_type == 'td3':
                self.agent = TD3(
                    state_dim=self.env.state_dim,
                    action_dim=self.env.action_dim,
                    flight_phase = self.flight_phase,
                    **self.agent_config['td3'])
        
        if self.shared_buffer is not None:
            self.agent.buffer = self.shared_buffer
                        
        if pre_train_critic_bool:
            self.pre_train_critic()

        self.trainer = TrainerEndo(env   = self.env,
                                   agent = self.agent,
                                   flight_phase = self.flight_phase,
                                   num_episodes = self.agent_config['num_episodes'],
                                   save_interval = save_interval,
                                   critic_warm_up_steps = self.agent_config['critic_warm_up_steps'],
                                   critic_warm_up_early_stopping_loss = self.agent_config['critic_warm_up_early_stopping_loss'],
                                   load_buffer_from_experiences_bool = load_buffer_bool,
                                   update_agent_every_n_steps = self.agent_config['update_agent_every_n_steps'],
                                   priority_update_interval = self.agent_config['priority_update_interval'],
                                   buffer_save_interval = self.buffer_save_interval)
        
        self.save_interval = save_interval
        if buffer_type == 'uniform':
            self.agent.use_uniform_sampling()
        elif buffer_type == 'prioritised':
            self.agent.use_prioritized_sampling()
        else:
            raise ValueError(f"Invalid buffer type: {buffer_type}")

    def __call__(self):
        self.trainer.train()
        return self.agent.buffer

    def load_agent_from_pso(self):
        actor_params, hidden_dim, number_of_hidden_layers = load_pso_actor(self.flight_phase, self.rl_type)
        self.agent_config['sac']['hidden_dim_actor'] = hidden_dim
        self.agent_config['sac']['number_of_hidden_layers_actor'] = number_of_hidden_layers
        if self.rl_type == 'sac':
            self.agent = SoftActorCritic(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                flight_phase = self.flight_phase,
                **self.agent_config['sac'])
        elif self.rl_type == 'td3':
            self.agent = TD3(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                flight_phase = self.flight_phase,
                **self.agent_config['td3'])
        else:
            raise ValueError(f"Invalid RL type: {self.rl_type}")
        self.agent.actor_params = actor_params  

    def load_agent_from_rl(self):
        if self.rl_type == 'sac':
            self.agent = load_sac(f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/soft-actor-critic.pkl')
            config_buffer_size = self.agent_config['sac']['buffer_size']
            current_buffer_size = self.agent.buffer_size
            # Only resize if the config buffer size is larger
            if current_buffer_size < config_buffer_size:
                print(f"Increasing buffer size from {current_buffer_size} to {config_buffer_size} based on agent_config")
                old_buffer = self.agent.buffer
                new_buffer = PERBuffer(
                    gamma=self.agent.gamma,
                    alpha=self.agent.alpha_buffer,
                    beta=self.agent.beta_buffer,
                    beta_decay=self.agent.beta_decay_buffer,
                    buffer_size=config_buffer_size,
                    state_dim=self.agent.state_dim,
                    action_dim=self.agent.action_dim,
                    trajectory_length=self.agent.trajectory_length,
                    batch_size=self.agent.batch_size,
                    expected_updates_to_convergence=self.agent.expected_updates_to_convergence
                )
                valid_entries = min(old_buffer.current_size, current_buffer_size)
                print(f"Copying {valid_entries} entries to the new buffer")
                for i in range(valid_entries):
                    entry = old_buffer.buffer[i]
                    if jnp.any(entry != 0):
                        new_buffer.buffer = new_buffer.buffer.at[i].set(entry)
                        new_buffer.priorities = new_buffer.priorities.at[i].set(old_buffer.priorities[i])
                
                new_buffer.position = valid_entries % config_buffer_size
                new_buffer.current_size = valid_entries
                self.agent.buffer = new_buffer
                self.agent.buffer_size = config_buffer_size
                print(f"Buffer resized successfully. New buffer has {new_buffer.current_size} entries and capacity {new_buffer.buffer_size}")
                
        elif self.rl_type == 'td3':
            self.agent = load_td3(f'data/agent_saves/TD3/{self.flight_phase}/saves/td3.pkl')
        else:
            raise ValueError(f"Invalid RL type: {self.rl_type}")

    def load_agent_from_supervisory(self):
        actor, actor_params, hidden_dim, hidden_layers = load_supervisory_actor(self.flight_phase, self.rl_type)
        self.agent_config['sac']['hidden_dim_actor'] = hidden_dim
        self.agent_config['sac']['number_of_hidden_layers_actor'] = hidden_layers
        if self.rl_type == 'sac':
            self.agent = SoftActorCritic(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                flight_phase = self.flight_phase,
                **self.agent_config['sac'])
        elif self.rl_type == 'td3':
            self.agent = TD3(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                flight_phase = self.flight_phase,
                **self.agent_config['td3'])
        self.agent.re_init_actor(actor, actor_params)
    
    def pre_train_critic(self):
        if self.rl_type == 'sac':
            critic_params_learner = pre_train_critic_from_pso_experiences(flight_phase = self.flight_phase,
                                                                        state_dim = self.env.state_dim,
                                                                        action_dim = self.env.action_dim,
                                                                        hidden_dim_critic = self.agent_config['sac']['hidden_dim_critic'],
                                                                        number_of_hidden_layers_critic = self.agent_config['sac']['number_of_hidden_layers_critic'],
                                                                        gamma = self.agent_config['sac']['gamma'],
                                                                        tau = self.agent_config['sac']['tau'],
                                                                        critic_learning_rate = self.agent_config['pre_train_critic_learning_rate'],
                                                                        batch_size = self.agent_config['pre_train_critic_batch_size'],
                                                                        reinforcement_type = self.rl_type)
        elif self.rl_type == 'td3':
            critic_params_learner = pre_train_critic_from_pso_experiences(flight_phase = self.flight_phase,
                                                                        state_dim = self.env.state_dim,
                                                                        action_dim = self.env.action_dim,
                                                                        hidden_dim_critic = self.agent_config['td3']['hidden_dim_critic'],
                                                                        number_of_hidden_layers_critic = self.agent_config['td3']['number_of_hidden_layers_critic'],
                                                                        gamma = self.agent_config['td3']['gamma'],
                                                                        tau = self.agent_config['td3']['tau'],
                                                                        critic_learning_rate = self.agent_config['pre_train_critic_learning_rate'],
                                                                        batch_size = self.agent_config['pre_train_critic_batch_size'],
                                                                        reinforcement_type = self.rl_type)
        else:
            raise ValueError(f"Invalid RL type: {self.rl_type}")
        self.agent.critic_params, self.agent.critic_target_params, self.agent.critic_opt_state = critic_params_learner()