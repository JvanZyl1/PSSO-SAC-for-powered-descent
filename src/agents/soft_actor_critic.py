import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.agents.functions.soft_actor_critic_functions import lambda_compile_sac, gaussian_likelihood
from src.agents.functions.plotter import agent_plotter_sac
from src.agents.functions.buffers import PERBuffer
from src.agents.functions.networks import DoubleCritic
from src.agents.functions.networks import GaussianActor as Actor
    
class SoftActorCritic:
    def __init__(self,
                 state_dim : int,
                 action_dim : int,
                 flight_phase : str,
                 # Dimensions
                 hidden_dim_actor : int,
                 number_of_hidden_layers_actor : int,
                 hidden_dim_critic : int,
                 number_of_hidden_layers_critic : int,
                 # Hyperparams
                 temperature_initial : float,
                 gamma : float,
                 tau : float,
                 alpha_buffer : float,
                 beta_buffer : float,
                 beta_decay_buffer : float,
                 buffer_size : int,
                 trajectory_length : int,
                 batch_size : int,
                 # Learning rates
                 critic_learning_rate : float,
                 actor_learning_rate : float,
                 temperature_learning_rate : float,
                 # Grad max norms
                 critic_grad_max_norm : float,
                 actor_grad_max_norm : float,
                 temperature_grad_max_norm : float,
                 # Max std
                 max_std : float,
                 # L2 reg
                 l2_reg_coef : float,
                 # Expected updates to convergence
                 expected_updates_to_convergence : int):
        
        self.rng_key = jax.random.PRNGKey(0)
        
        self.save_path = f'results/VanillaSAC/{flight_phase}/'
        self.flight_phase = flight_phase
        self.buffer = PERBuffer(
            gamma=gamma,
            alpha=alpha_buffer,
            beta=beta_buffer,
            beta_decay=beta_decay_buffer,
            buffer_size=buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            trajectory_length=trajectory_length,
            batch_size=batch_size,
            expected_updates_to_convergence=expected_updates_to_convergence
        )
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=f'data/agent_saves/VanillaSAC/{flight_phase}/runs/{self.run_id}')

        self.max_std = max_std

        self.actor = Actor(action_dim=action_dim,
                           hidden_dim=hidden_dim_actor,
                           number_of_hidden_layers=number_of_hidden_layers_actor)
        self.actor_params = self.actor.init(self.get_subkey(), jnp.zeros((1, state_dim)))
        self.actor_opt_state = optax.adam(learning_rate=actor_learning_rate).init(self.actor_params)

        self.critic = DoubleCritic(state_dim=state_dim,
                                   action_dim=action_dim,
                                   hidden_dim=hidden_dim_critic,
                                   number_of_hidden_layers=number_of_hidden_layers_critic)
        self.critic_params = self.critic.init(self.get_subkey(), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.critic_opt_state = optax.adam(learning_rate=critic_learning_rate).init(self.critic_params)
        self.critic_target_params = jax.tree_util.tree_map(lambda x: x.copy(), self.critic_params)

        self.temperature = temperature_initial
        self.temperature_opt_state = optax.adam(learning_rate=temperature_learning_rate).init(self.temperature)

        self.critic_learning_rate = critic_learning_rate
        self.critic_grad_max_norm = critic_grad_max_norm

        self.actor_learning_rate = actor_learning_rate
        self.actor_grad_max_norm = actor_grad_max_norm

        self.temperature_learning_rate = temperature_learning_rate
        self.temperature_grad_max_norm = temperature_grad_max_norm

        self.gamma = gamma
        self.tau = tau
        self.l2_reg_coef = l2_reg_coef

        self.expected_updates_to_convergence = expected_updates_to_convergence

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.hidden_dim_actor = hidden_dim_actor
        self.number_of_hidden_layers_actor = number_of_hidden_layers_actor
        self.hidden_dim_critic = hidden_dim_critic
        self.number_of_hidden_layers_critic = number_of_hidden_layers_critic
        self.temperature_initial = temperature_initial
        self.alpha_buffer = alpha_buffer
        self.beta_buffer = beta_buffer
        self.beta_decay_buffer = beta_decay_buffer
        self.buffer_size = buffer_size
        self.trajectory_length = trajectory_length
        self.critic_learning_rate = critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.temperature_learning_rate = temperature_learning_rate
        self.critic_grad_max_norm = critic_grad_max_norm
        self.actor_grad_max_norm = actor_grad_max_norm
        self.temperature_grad_max_norm = temperature_grad_max_norm
        self.batch_size = batch_size
        self.target_entropy = -self.action_dim * 4
        self.update_function, self.calculate_td_error_lambda, self.critic_warm_up_update_lambda \
            = lambda_compile_sac(critic_optimiser = optax.adam(learning_rate = self.critic_learning_rate),
                                 critic = self.critic,
                                 critic_grad_max_norm = self.critic_grad_max_norm,
                                 actor_optimiser = optax.adam(learning_rate = self.actor_learning_rate),
                                 actor = self.actor,
                                 actor_grad_max_norm = self.actor_grad_max_norm,
                                 temperature_optimiser = optax.adam(learning_rate=self.temperature_learning_rate),
                                 temperature_grad_max_norm = self.temperature_grad_max_norm,
                                 gamma = self.gamma,
                                 tau = self.tau,
                                 target_entropy = self.target_entropy,
                                 initial_temperature = self.temperature_initial,
                                 max_std = self.max_std,
                                 l2_reg_coef = self.l2_reg_coef)

        # LOGGING
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.actor_entropy_loss_episode = 0.0
        self.actor_q_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_losses = []
        self.actor_losses = []
        self.actor_entropy_losses = []
        self.actor_q_losses = []
        self.temperature_losses = []
        self.td_errors = []
        self.temperature_values = []
        self.number_of_steps = []
        self.critic_warm_up_step_idx = 0
        self.critic_weighted_mse_losses = []
        self.critic_l2_regs = []
        self.critic_weighted_mse_loss_episode = 0.0
        self.critic_l2_reg_episode = 0.0
        # Log initial temperature
        self.writer.add_scalar('Initial/Temperature', np.array(self.temperature), 0)
        self.first_step_bool = True

        self.use_prioritized_sampling()

        self.name = 'VanillaSAC'

        self.action_std = []
        self.action_std_mean = []
        self.action_std_std = []
        self.action_std_max = []
        self.action_std_min = []
        self.action_mean = []
        self.action_mean_mean = []
        self.action_mean_std = []
        self.action_mean_max = []
        self.action_mean_min = []

        self.td_errors_mean = []
        self.td_errors_std = []
        self.td_errors_max = []
        self.td_errors_min = []

        self.log_probabilities_mean = []
        self.log_probabilities_std = []
        self.log_probabilities_max = []
        self.log_probabilities_min = []

        self.sampled_rewards_min = []
        self.sampled_rewards_max = []
        self.sampled_rewards_mean = []
        self.sampled_rewards_std = []

        self.sampled_states_min = []
        self.sampled_states_max = []
        self.sampled_states_mean = []
        self.sampled_states_std = []

        self.sampled_actions_min = []
        self.sampled_actions_max = []
        self.sampled_actions_mean = []
        self.sampled_actions_std = []

        self.critic_losses_episode_list = []
        self.actor_losses_episode_list = []
        self.temperature_losses_episode_list = []
        self.actor_q_losses_episode_list = []
        self.actor_entropy_losses_episode_list = []
        self.critic_weighted_mse_losses_episode_list = []
        self.critic_l2_regs_episode_list = []
        self.temperature_values_episode_list = []

        self.critic_losses_mean = []
        self.critic_losses_std = []
        self.critic_losses_max = []
        self.critic_losses_min = []

        self.critic_weighted_mse_losses_mean = []
        self.critic_weighted_mse_losses_std = []
        self.critic_weighted_mse_losses_max = []
        self.critic_weighted_mse_losses_min = []
        
        self.critic_l2_regs_mean = []
        self.critic_l2_regs_std = []
        self.critic_l2_regs_max = []
        self.critic_l2_regs_min = []

        self.actor_losses_mean = []
        self.actor_losses_std = []
        self.actor_losses_max = []
        self.actor_losses_min = []

        self.actor_entropy_losses_mean = []
        self.actor_entropy_losses_std = []
        self.actor_entropy_losses_max = []
        self.actor_entropy_losses_min = []

        self.actor_q_losses_mean = []
        self.actor_q_losses_std = []
        self.actor_q_losses_max = []
        self.actor_q_losses_min = []

        self.temperature_losses_mean = []
        self.temperature_losses_std = []
        self.temperature_losses_max = []
        self.temperature_losses_min = []

        self.temperature_values_mean = []
        self.temperature_values_std = []
        self.temperature_values_max = []
        self.temperature_values_min = []

        self.q1_values_episode_list = []
        self.q2_values_episode_list = []
        self.q1_values_mean = []
        self.q1_values_std = []
        self.q1_values_max = []
        self.q1_values_min = []

        self.q2_values_mean = []
        self.q2_values_std = []
        self.q2_values_max = []
        self.q2_values_min = []
        
        # Add fields for next_q1 and next_q2
        self.next_q1_values_episode_list = []
        self.next_q2_values_episode_list = []
        self.next_q1_values_mean = []
        self.next_q1_values_std = []
        self.next_q1_values_max = []
        self.next_q1_values_min = []
        self.next_q2_values_mean = []
        self.next_q2_values_std = []
        self.next_q2_values_max = []
        self.next_q2_values_min = []
        
        # Add fields for abs(Q1-Q2) and abs(next_Q1-next_Q2)
        self.q_diff_values_episode_list = []
        self.q_diff_values_mean = []
        self.q_diff_values_std = []
        self.q_diff_values_max = []
        self.q_diff_values_min = []
        
        self.next_q_diff_values_episode_list = []
        self.next_q_diff_values_mean = []
        self.next_q_diff_values_std = []
        self.next_q_diff_values_max = []
        self.next_q_diff_values_min = []
        
    def re_init_actor(self, new_actor, new_actor_params):
        self.actor = new_actor
        self.actor_params = new_actor_params
        self.actor_opt_state = optax.adam(learning_rate=self.actor_learning_rate).init(self.actor_params)
        self.update_function, self.calculate_td_error_lambda, self.critic_warm_up_update_lambda \
            = lambda_compile_sac(critic_optimiser = optax.adam(learning_rate = self.critic_learning_rate),
                                 critic = self.critic,
                                 critic_grad_max_norm = self.critic_grad_max_norm,
                                 actor_optimiser = optax.adam(learning_rate = self.actor_learning_rate),
                                 actor = self.actor,
                                 actor_grad_max_norm = self.actor_grad_max_norm,
                                 temperature_optimiser = optax.adam(learning_rate=self.temperature_learning_rate),
                                 temperature_grad_max_norm = self.temperature_grad_max_norm,
                                 gamma = self.gamma,
                                 tau = self.tau,
                                 target_entropy = self.target_entropy,
                                 initial_temperature = self.temperature_initial,
                                 max_std = self.max_std,
                                 l2_reg_coef = self.l2_reg_coef)

    def reset(self):
        # LOGGING
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.actor_entropy_loss_episode = 0.0
        self.actor_q_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_losses = []
        self.actor_losses = []
        self.actor_entropy_losses = []
        self.actor_q_losses = []
        self.temperature_losses = []
        self.td_errors = []
        self.temperature_values = []
        self.number_of_steps = []
        self.rng_key = jax.random.PRNGKey(0)
        self.buffer.reset()
        self.critic_warm_up_step_idx = 0
        self.first_step_bool = True
        self.critic_weighted_mse_loss_episode = 0.0
        self.critic_l2_reg_episode = 0.0
        self.critic_weighted_mse_losses = []
        self.critic_l2_regs = []


        self.action_std = []
        self.action_std_mean = []
        self.action_std_std = []
        self.action_std_max = []
        self.action_std_min = []
        self.action_mean = []
        self.action_mean_mean = []
        self.action_mean_std = []
        self.action_mean_max = []
        self.action_mean_min = []

        self.td_errors_mean = []
        self.td_errors_std = []
        self.td_errors_max = []
        self.td_errors_min = []

        self.log_probabilities_mean = []
        self.log_probabilities_std = []
        self.log_probabilities_max = []
        self.log_probabilities_min = []

        self.sampled_rewards_min = []
        self.sampled_rewards_max = []
        self.sampled_rewards_mean = []
        self.sampled_rewards_std = []

        self.sampled_states_min = []
        self.sampled_states_max = []
        self.sampled_states_mean = []
        self.sampled_states_std = []

        self.sampled_actions_min = []
        self.sampled_actions_max = []
        self.sampled_actions_mean = []
        self.sampled_actions_std = []

        self.critic_losses_episode_list = []
        self.actor_losses_episode_list = []
        self.temperature_losses_episode_list = []
        self.actor_q_losses_episode_list = []
        self.actor_entropy_losses_episode_list = []
        self.critic_weighted_mse_losses_episode_list = []
        self.critic_l2_regs_episode_list = []
        self.temperature_values_episode_list = []
        self.q1_values_episode_list = []
        self.q2_values_episode_list = []
        self.next_q1_values_episode_list = []
        self.next_q2_values_episode_list = []
        self.q_diff_values_episode_list = []
        self.next_q_diff_values_episode_list = []

        self.critic_losses_mean = []
        self.critic_losses_std = []
        self.critic_losses_max = []
        self.critic_losses_min = []

        self.critic_weighted_mse_losses_mean = []
        self.critic_weighted_mse_losses_std = []
        self.critic_weighted_mse_losses_max = []
        self.critic_weighted_mse_losses_min = []
        
        self.critic_l2_regs_mean = []
        self.critic_l2_regs_std = []
        self.critic_l2_regs_max = []
        self.critic_l2_regs_min = []

        self.actor_losses_mean = []
        self.actor_losses_std = []
        self.actor_losses_max = []
        self.actor_losses_min = []

        self.actor_entropy_losses_mean = []
        self.actor_entropy_losses_std = []
        self.actor_entropy_losses_max = []
        self.actor_entropy_losses_min = []

        self.actor_q_losses_mean = []
        self.actor_q_losses_std = []
        self.actor_q_losses_max = []
        self.actor_q_losses_min = []

        self.temperature_losses_mean = []
        self.temperature_losses_std = []
        self.temperature_losses_max = []
        self.temperature_losses_min = []

        self.temperature_values_mean = []
        self.temperature_values_std = []
        self.temperature_values_max = []
        self.temperature_values_min = []

        self.q1_values_mean = []
        self.q1_values_std = []
        self.q1_values_max = []
        self.q1_values_min = []

        self.q2_values_mean = []
        self.q2_values_std = []
        self.q2_values_max = []
        self.q2_values_min = []
        
        self.next_q1_values_mean = []
        self.next_q1_values_std = []
        self.next_q1_values_max = []
        self.next_q1_values_min = []

        self.next_q2_values_mean = []
        self.next_q2_values_std = []
        self.next_q2_values_max = []
        self.next_q2_values_min = []
        
        self.q_diff_values_mean = []
        self.q_diff_values_std = []
        self.q_diff_values_max = []
        self.q_diff_values_min = []
        
        self.next_q_diff_values_mean = []
        self.next_q_diff_values_std = []
        self.next_q_diff_values_max = []
        self.next_q_diff_values_min = []
        
    def get_subkey(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey
    
    def get_normal_distributions_batched(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.batch_size, self.action_dim)))
        return normal_distribution
    
    def get_normal_distribution(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.action_dim, )))
        return normal_distribution
    
    def critic_warm_up_step(self):
        states, actions, rewards, next_states, dones, _, _ = self.buffer(self.get_subkey())
        self.critic_params, self.critic_opt_state, self.critic_target_params, critic_loss_warm_up, weighted_td_error_loss, l2_reg = self.critic_warm_up_update_lambda(actor_params = self.actor_params,
                                                                                                                                      normal_distribution_for_next_actions = self.get_normal_distributions_batched(),
                                                                                                                                      states = states,
                                                                                                                                      actions = actions,
                                                                                                                                      rewards = rewards,
                                                                                                                                      next_states = next_states,
                                                                                                                                      dones = dones,
                                                                                                                                      critic_params = self.critic_params,
                                                                                                                                      critic_target_params = self.critic_target_params,
                                                                                                                                      critic_opt_state = self.critic_opt_state)
        self.writer.add_scalar('CriticWarmUp/Loss', np.array(critic_loss_warm_up), self.critic_warm_up_step_idx)
        self.critic_warm_up_step_idx += 1
        return critic_loss_warm_up, weighted_td_error_loss, l2_reg

    def calculate_td_error(self,
                           state: jnp.ndarray,
                           action: jnp.ndarray,
                           reward: jnp.ndarray,
                           next_state: jnp.ndarray,
                           done: jnp.ndarray) -> jnp.ndarray:
        # This is non-batched.
        next_action_mean, next_action_std = self.actor.apply(self.actor_params, next_state)
        noise   = jnp.clip(self.get_normal_distribution(), -self.max_std, self.max_std)
        next_actions = jnp.clip(noise * next_action_std + next_action_mean, -1, 1)
        next_log_policy = gaussian_likelihood(next_actions, next_action_mean, next_action_std)
        td_errors =  self.calculate_td_error_lambda(states = state,
                                                    actions = action,
                                                    rewards = reward,
                                                    next_states = next_state,
                                                    dones = done,
                                                    temperature = self.temperature,
                                                    critic_params = self.critic_params,
                                                    critic_target_params = self.critic_target_params,
                                                    next_actions = next_actions,
                                                    next_log_policy = next_log_policy)
        return td_errors

    def calculate_td_error_vmap(self,
                               states: jnp.ndarray,
                               actions: jnp.ndarray,
                               rewards: jnp.ndarray,
                               next_states: jnp.ndarray,
                               dones: jnp.ndarray) -> jnp.ndarray:
        # Ensure proper shapes
        states = jnp.reshape(states, (len(states), -1))  # (batch_size, state_dim)
        actions = jnp.reshape(actions, (len(actions), -1))  # (batch_size, action_dim)
        rewards = jnp.reshape(rewards, (len(rewards), 1))  # (batch_size, 1)
        next_states = jnp.reshape(next_states, (len(next_states), -1))  # (batch_size, state_dim)
        dones = jnp.reshape(dones, (len(dones), 1))  # (batch_size, 1)
        
        next_action_means, next_action_stds = jax.vmap(self.actor.apply, in_axes=(None, 0))(self.actor_params, next_states)
        self.rng_key, *subkeys = jax.random.split(self.rng_key, len(next_states) + 1)
        normal_distributions = jax.vmap(lambda key: jax.random.normal(key, (self.action_dim,)) * self.max_std)(jnp.array(subkeys))
        next_actions = jax.vmap(lambda mean, std, normal: normal * std + mean)(
            next_action_means, next_action_stds, normal_distributions
        )
        next_log_policies = jax.vmap(gaussian_likelihood)(next_actions, next_action_means, next_action_stds)
        next_log_policies = jnp.reshape(next_log_policies, (len(next_log_policies), 1))  # (batch_size, 1) shape

        def td_error_calc(state, action, reward, next_state, done, next_action, next_log_policy):
            return self.calculate_td_error_lambda(
                states=state,
                actions=action,
                rewards=reward,
                next_states=next_state,
                dones=done,
                temperature=self.temperature,
                critic_params=self.critic_params,
                critic_target_params=self.critic_target_params,
                next_actions=next_action,
                next_log_policy=next_log_policy
            )
        td_errors = jax.vmap(td_error_calc)(states, actions, rewards, next_states, dones, next_actions, next_log_policies)
        
        return td_errors

    def select_actions(self,
                       state : jnp.ndarray) -> jnp.ndarray:
        action_mean, action_std = self.actor.apply(self.actor_params, jax.lax.stop_gradient(state))
        noise   = jnp.clip(self.get_normal_distribution(), -self.max_std, self.max_std)
        actions = jnp.clip(noise * action_std + action_mean, -1, 1)
        return actions

    def select_actions_no_stochastic(self,
                                     state : jnp.ndarray) -> jnp.ndarray:
        action_mean, _ = self.actor.apply(self.actor_params, state)
        action_mean = jnp.clip(action_mean, -1, 1)
        return action_mean

    def update_episode(self):
        self.critic_losses.append(self.critic_loss_episode)
        self.actor_losses.append(self.actor_loss_episode)
        self.temperature_losses.append(self.temperature_loss_episode)
        self.td_errors.append(self.td_errors_episode)
        for temperature in self.temperature_values_all_episode:
            self.temperature_values.append(temperature)
        self.number_of_steps.append(self.number_of_steps_episode)
        self.critic_weighted_mse_losses.append(self.critic_weighted_mse_loss_episode)
        self.critic_l2_regs.append(self.critic_l2_reg_episode)
        self.actor_entropy_losses.append(self.actor_entropy_loss_episode)
        self.actor_q_losses.append(self.actor_q_loss_episode)


        self.critic_losses_mean.append(np.mean(np.array(self.critic_losses_episode_list)))
        self.critic_losses_std.append(np.std(np.array(self.critic_losses_episode_list)))
        self.critic_losses_max.append(np.max(np.array(self.critic_losses_episode_list)))
        self.critic_losses_min.append(np.min(np.array(self.critic_losses_episode_list)))
        
        self.critic_weighted_mse_losses_mean.append(np.mean(np.array(self.critic_weighted_mse_losses_episode_list)))
        self.critic_weighted_mse_losses_std.append(np.std(np.array(self.critic_weighted_mse_losses_episode_list)))
        self.critic_weighted_mse_losses_max.append(np.max(np.array(self.critic_weighted_mse_losses_episode_list)))
        self.critic_weighted_mse_losses_min.append(np.min(np.array(self.critic_weighted_mse_losses_episode_list)))
        self.critic_l2_regs_mean.append(np.mean(np.array(self.critic_l2_regs_episode_list)))
        self.critic_l2_regs_std.append(np.std(np.array(self.critic_l2_regs_episode_list)))
        self.critic_l2_regs_max.append(np.max(np.array(self.critic_l2_regs_episode_list)))
        self.critic_l2_regs_min.append(np.min(np.array(self.critic_l2_regs_episode_list)))
        self.actor_losses_mean.append(np.mean(np.array(self.actor_losses_episode_list)))
        self.actor_losses_std.append(np.std(np.array(self.actor_losses_episode_list)))
        self.actor_losses_max.append(np.max(np.array(self.actor_losses_episode_list)))
        self.actor_losses_min.append(np.min(np.array(self.actor_losses_episode_list)))
        self.actor_entropy_losses_mean.append(np.mean(np.array(self.actor_entropy_losses_episode_list)))
        self.actor_entropy_losses_std.append(np.std(np.array(self.actor_entropy_losses_episode_list)))
        self.actor_entropy_losses_max.append(np.max(np.array(self.actor_entropy_losses_episode_list)))
        self.actor_entropy_losses_min.append(np.min(np.array(self.actor_entropy_losses_episode_list)))
        self.actor_q_losses_mean.append(np.mean(np.array(self.actor_q_losses_episode_list)))
        self.actor_q_losses_std.append(np.std(np.array(self.actor_q_losses_episode_list)))
        self.actor_q_losses_max.append(np.max(np.array(self.actor_q_losses_episode_list)))
        self.actor_q_losses_min.append(np.min(np.array(self.actor_q_losses_episode_list)))
        self.temperature_losses_mean.append(np.mean(np.array(self.temperature_losses_episode_list)))
        self.temperature_losses_std.append(np.std(np.array(self.temperature_losses_episode_list)))
        self.temperature_losses_max.append(np.max(np.array(self.temperature_losses_episode_list)))
        self.temperature_losses_min.append(np.min(np.array(self.temperature_losses_episode_list)))
        self.temperature_values_mean.append(np.mean(np.array(self.temperature_values_episode_list)))
        self.temperature_values_std.append(np.std(np.array(self.temperature_values_episode_list)))
        self.temperature_values_max.append(np.max(np.array(self.temperature_values_episode_list)))
        self.temperature_values_min.append(np.min(np.array(self.temperature_values_episode_list)))

        self.q1_values_mean.append(np.mean(np.array(self.q1_values_episode_list)))
        self.q1_values_std.append(np.std(np.array(self.q1_values_episode_list)))
        self.q1_values_max.append(np.max(np.array(self.q1_values_episode_list)))
        self.q1_values_min.append(np.min(np.array(self.q1_values_episode_list)))

        self.q2_values_mean.append(np.mean(np.array(self.q2_values_episode_list)))
        self.q2_values_std.append(np.std(np.array(self.q2_values_episode_list)))
        self.q2_values_max.append(np.max(np.array(self.q2_values_episode_list)))
        self.q2_values_min.append(np.min(np.array(self.q2_values_episode_list)))

        self.next_q1_values_mean.append(np.mean(np.array(self.next_q1_values_episode_list)))
        self.next_q1_values_std.append(np.std(np.array(self.next_q1_values_episode_list)))
        self.next_q1_values_max.append(np.max(np.array(self.next_q1_values_episode_list)))
        self.next_q1_values_min.append(np.min(np.array(self.next_q1_values_episode_list)))

        self.next_q2_values_mean.append(np.mean(np.array(self.next_q2_values_episode_list)))
        self.next_q2_values_std.append(np.std(np.array(self.next_q2_values_episode_list)))
        self.next_q2_values_max.append(np.max(np.array(self.next_q2_values_episode_list)))
        self.next_q2_values_min.append(np.min(np.array(self.next_q2_values_episode_list)))

        self.q_diff_values_mean.append(np.mean(np.array(self.q_diff_values_episode_list)))
        self.q_diff_values_std.append(np.std(np.array(self.q_diff_values_episode_list)))
        self.q_diff_values_max.append(np.max(np.array(self.q_diff_values_episode_list)))
        self.q_diff_values_min.append(np.min(np.array(self.q_diff_values_episode_list)))
        
        self.next_q_diff_values_mean.append(np.mean(np.array(self.next_q_diff_values_episode_list)))
        self.next_q_diff_values_std.append(np.std(np.array(self.next_q_diff_values_episode_list)))
        self.next_q_diff_values_max.append(np.max(np.array(self.next_q_diff_values_episode_list)))
        self.next_q_diff_values_min.append(np.min(np.array(self.next_q_diff_values_episode_list)))

        self.critic_losses_episode_list = []
        self.actor_losses_episode_list = []
        self.temperature_losses_episode_list = []
        self.actor_q_losses_episode_list = []
        self.actor_entropy_losses_episode_list = []
        self.critic_weighted_mse_losses_episode_list = []
        self.critic_l2_regs_episode_list = []
        self.temperature_values_episode_list = []
        self.q1_values_episode_list = []
        self.q2_values_episode_list = []
        self.next_q1_values_episode_list = []
        self.next_q2_values_episode_list = []
        self.q_diff_values_episode_list = []
        self.next_q_diff_values_episode_list = []

        self.writer.add_scalar('Episode/CriticLoss', np.array(self.critic_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/ActorLoss', np.array(self.actor_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/ActorEntropyLoss', np.array(self.actor_entropy_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/ActorQLoss', np.array(self.actor_q_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/TemperatureLoss', np.array(self.temperature_loss_episode), self.episode_idx)
        self.writer.add_histogram('Episode/TDError', np.array(self.td_errors_episode), self.episode_idx)
        self.writer.add_scalar('Episode/Temperature', np.array(self.temperature), self.episode_idx)
        self.writer.add_scalar('Episode/NumberOfSteps', np.array(self.number_of_steps_episode), self.episode_idx)
        self.writer.add_scalar('Episode/LogTemperature', np.log(self.temperature), self.episode_idx)
        self.writer.add_scalar('Episode/WeightedTDErrorLoss', np.array(self.critic_weighted_mse_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/L2Reg', np.array(self.critic_l2_reg_episode), self.episode_idx)
        for layer_name, layer_params in self.actor_params['params'].items():
            for param_name, param in layer_params.items():
                self.writer.add_histogram(f'Episode/Actor/{layer_name}/{param_name}', np.array(param).flatten(), self.episode_idx)

        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.temperature_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.temperature_values_all_episode = []
        self.number_of_steps_episode = 0.0
        self.episode_idx += 1
        self.first_step_bool = False
        self.critic_weighted_mse_loss_episode = 0.0
        self.critic_l2_reg_episode = 0.0

    def update(self):
        states, actions, rewards, next_states, dones, index, weights_buffer = self.buffer(self.get_subkey())
        normal_distribution_for_actions = self.get_normal_distributions_batched()
        self.critic_params, self.critic_opt_state, critic_loss, td_errors, \
            self.actor_params, self.actor_opt_state, actor_loss, actor_entropy_loss, actor_q_loss,q1,q2, next_q1, next_q2, \
            self.temperature, self.temperature_opt_state, temperature_loss, \
            self.critic_target_params, \
            current_log_probabilities, action_std, action_mean, \
            weighted_td_error_loss, l2_reg = self.update_function(actor_params = self.actor_params,
                                                             actor_opt_state = self.actor_opt_state,
                                                             normal_distribution_for_next_actions = self.get_normal_distributions_batched(),
                                                             normal_distribution_for_actions = normal_distribution_for_actions,
                                                             states = states,
                                                             actions = actions,
                                                             rewards = rewards,
                                                             next_states = next_states,
                                                             dones = dones,
                                                             buffer_weights = weights_buffer,
                                                             temperature = self.temperature,
                                                             temperature_opt_state = self.temperature_opt_state,
                                                             critic_params = self.critic_params,
                                                             critic_target_params = self.critic_target_params,
                                                             critic_opt_state = self.critic_opt_state,
                                                             first_step_bool = self.first_step_bool)
        self.buffer.update_priorities(index, td_errors)

        self.critic_loss_episode += critic_loss
        self.critic_weighted_mse_loss_episode += weighted_td_error_loss
        self.critic_l2_reg_episode += l2_reg
        self.actor_loss_episode += actor_loss
        self.actor_entropy_loss_episode += actor_entropy_loss
        self.actor_q_loss_episode += actor_q_loss
        self.temperature_loss_episode += temperature_loss
        self.td_errors_episode += td_errors
        self.temperature_values_all_episode.append(float(self.temperature))
        self.number_of_steps_episode += 1
        self.step_idx += 1
        
        # JAX -> Numpy for log
        critic_loss_np = np.array(critic_loss)
        actor_loss_np = np.array(actor_loss)
        temperature_loss_np = np.array(temperature_loss)
        td_errors_np = np.array(td_errors)
        temperature_np = np.array(self.temperature)
        actor_entropy_loss_np = np.array(actor_entropy_loss)
        actor_q_loss_np = np.array(actor_q_loss)
        
        self.writer.add_scalar('Steps/CriticLoss', critic_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/ActorLoss', actor_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/ActorEntropyLoss', actor_entropy_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/ActorQLoss', actor_q_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/NormalDistribution_Mean', np.mean(np.array(normal_distribution_for_actions)), self.step_idx)
        self.writer.add_scalar('Steps/NormalDistribution_Std', np.std(np.array(normal_distribution_for_actions)), self.step_idx)
        self.writer.add_scalar('Steps/NormalDistribution_Max', np.max(np.array(normal_distribution_for_actions)), self.step_idx)
        self.writer.add_scalar('Steps/NormalDistribution_Min', np.min(np.array(normal_distribution_for_actions)), self.step_idx)
        self.writer.add_scalar('Steps/TemperatureLoss', temperature_loss_np, self.step_idx)
        self.writer.add_scalar('Steps/ActionStd_Mean', np.mean(np.array(action_std)), self.step_idx)
        self.writer.add_scalar('Steps/ActionStd_Std', np.std(np.array(action_std)), self.step_idx)
        self.writer.add_scalar('Steps/ActionStd_Max', np.max(np.array(action_std)), self.step_idx)
        self.writer.add_scalar('Steps/ActionStd_Min', np.min(np.array(action_std)), self.step_idx)
        self.writer.add_scalar('Steps/ActionMean_Mean', np.mean(np.array(action_mean)), self.step_idx)
        self.writer.add_scalar('Steps/ActionMean_Std', np.std(np.array(action_mean)), self.step_idx)
        self.writer.add_scalar('Steps/ActionMean_Max', np.max(np.array(action_mean)), self.step_idx)
        self.writer.add_scalar('Steps/ActionMean_Min', np.min(np.array(action_mean)), self.step_idx)
        self.writer.add_scalar('Steps/CurrentLogProbabilities_Mean', np.mean(np.array(current_log_probabilities)), self.step_idx)
        self.writer.add_scalar('Steps/CurrentLogProbabilities_Std', np.std(np.array(current_log_probabilities)), self.step_idx)
        self.writer.add_scalar('Steps/CurrentLogProbabilities_Max', np.max(np.array(current_log_probabilities)), self.step_idx)
        self.writer.add_scalar('Steps/CurrentLogProbabilities_Min', np.min(np.array(current_log_probabilities)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights_Mean', np.mean(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights_Std', np.std(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights_Max', np.max(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights_Min', np.min(np.array(weights_buffer)), self.step_idx)
        self.writer.add_histogram('Steps/ActionStd', np.array(action_std), self.step_idx)
        self.writer.add_histogram('Steps/CurrentLogProbabilities', np.array(current_log_probabilities), self.step_idx)
        self.writer.add_histogram('Steps/BufferWeights', np.array(weights_buffer), self.step_idx)
        self.writer.add_scalar('Steps/TDError/mean', np.mean(td_errors_np), self.step_idx)
        self.writer.add_scalar('Steps/TDError/std', np.std(td_errors_np), self.step_idx)
        self.writer.add_scalar('Steps/TDError/max', np.max(td_errors_np), self.step_idx)
        self.writer.add_scalar('Steps/TDError/min', np.min(td_errors_np), self.step_idx)
        self.writer.add_scalar('Steps/Temperature', temperature_np, self.step_idx)
        self.writer.add_scalar('Steps/LogTemperature', np.log(temperature_np), self.step_idx)
        self.writer.add_scalar('Update/StartTemperature', np.array(self.temperature), self.step_idx)
        self.writer.add_scalar('Steps/L2Reg', np.array(l2_reg), self.step_idx)
        self.writer.add_scalar('Steps/WeightedTDErrorLoss', np.array(weighted_td_error_loss), self.step_idx)
        self.writer.add_scalar('Steps/Temperature', np.array(self.temperature), self.step_idx)

        self.action_std.append(np.array(action_std))
        self.action_std_mean.append(np.mean(np.array(action_std)))
        self.action_std_std.append(np.std(np.array(action_std)))
        self.action_std_max.append(np.max(np.array(action_std)))
        self.action_std_min.append(np.min(np.array(action_std)))
        self.action_mean.append(np.array(action_mean))
        self.action_mean_mean.append(np.mean(np.array(action_mean)))
        self.action_mean_std.append(np.std(np.array(action_mean)))
        self.action_mean_max.append(np.max(np.array(action_mean)))
        self.action_mean_min.append(np.min(np.array(action_mean)))

        self.td_errors_mean.append(np.mean(np.array(td_errors)))
        self.td_errors_std.append(np.std(np.array(td_errors)))
        self.td_errors_max.append(np.max(np.array(td_errors)))
        self.td_errors_min.append(np.min(np.array(td_errors)))

        self.log_probabilities_mean.append(np.mean(np.array(current_log_probabilities)))
        self.log_probabilities_std.append(np.std(np.array(current_log_probabilities)))
        self.log_probabilities_max.append(np.max(np.array(current_log_probabilities)))
        self.log_probabilities_min.append(np.min(np.array(current_log_probabilities)))

        self.sampled_rewards_min.append(np.min(np.array(rewards)))
        self.sampled_rewards_max.append(np.max(np.array(rewards)))
        self.sampled_rewards_mean.append(np.mean(np.array(rewards)))
        self.sampled_rewards_std.append(np.std(np.array(rewards)))

        self.sampled_states_min.append(np.min(np.array(states)))
        self.sampled_states_max.append(np.max(np.array(states)))
        self.sampled_states_mean.append(np.mean(np.array(states)))
        self.sampled_states_std.append(np.std(np.array(states)))

        self.sampled_actions_min.append(np.min(np.array(actions)))
        self.sampled_actions_max.append(np.max(np.array(actions)))
        self.sampled_actions_mean.append(np.mean(np.array(actions)))
        self.sampled_actions_std.append(np.std(np.array(actions)))

        self.critic_losses_episode_list.append(critic_loss)
        self.actor_losses_episode_list.append(actor_loss)
        self.temperature_losses_episode_list.append(temperature_loss)
        self.actor_q_losses_episode_list.append(actor_q_loss)
        self.actor_entropy_losses_episode_list.append(actor_entropy_loss)
        self.critic_weighted_mse_losses_episode_list.append(weighted_td_error_loss)
        self.critic_l2_regs_episode_list.append(l2_reg)
        self.temperature_values_episode_list.append(float(self.temperature))
        self.q1_values_episode_list.append(q1)
        self.q2_values_episode_list.append(q2)
        self.next_q1_values_episode_list.append(next_q1)
        self.next_q2_values_episode_list.append(next_q2)
        self.q_diff_values_episode_list.append(np.abs(q1 - q2))
        self.next_q_diff_values_episode_list.append(np.abs(next_q1 - next_q2))
        
    def plotter(self):
        agent_plotter_sac(self)

    def save(self):
        file_path = f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/soft-actor-critic.pkl'
        agent_state = {
            'inputs' : {
                'flight_phase' : self.flight_phase,
                'hidden_dim_actor' : self.hidden_dim_actor,
                'number_of_hidden_layers_actor' : self.number_of_hidden_layers_actor,
                'hidden_dim_critic' : self.hidden_dim_critic,
                'number_of_hidden_layers_critic' : self.number_of_hidden_layers_critic,
                'temperature_initial' : self.temperature_initial,
                'gamma' : self.gamma,
                'tau' : self.tau,
                'alpha_buffer' : self.alpha_buffer,
                'beta_buffer' : self.beta_buffer,
                'beta_decay_buffer' : self.beta_decay_buffer,
                'buffer_size' : self.buffer_size,
                'state_dim' : self.state_dim,
                'action_dim' : self.action_dim,
                'trajectory_length' : self.trajectory_length,
                'batch_size' : self.batch_size,
                'critic_learning_rate' : self.critic_learning_rate,
                'actor_learning_rate' : self.actor_learning_rate,
                'temperature_learning_rate' : self.temperature_learning_rate,
                'critic_grad_max_norm' : self.critic_grad_max_norm,
                'actor_grad_max_norm' : self.actor_grad_max_norm,
                'temperature_grad_max_norm' : self.temperature_grad_max_norm,
                'max_std' : self.max_std,
                'l2_reg_coef' : self.l2_reg_coef,
                'expected_updates_to_convergence' : self.expected_updates_to_convergence
            },
            'misc' : {
                'rng_key' : self.rng_key,
                'run_id' : self.run_id
            },
            'logging' : {
                'critic_loss_episode' : self.critic_loss_episode,
                'actor_loss_episode' : self.actor_loss_episode,
                'temperature_loss_episode' : self.temperature_loss_episode,
                'td_errors_episode' : self.td_errors_episode,
                'temperature_values_all_episode' : self.temperature_values_all_episode,
                'number_of_steps_episode' : self.number_of_steps_episode,
                'episode_idx' : self.episode_idx,
                'step_idx' : self.step_idx,
                'critic_losses' : self.critic_losses,
                'actor_losses' : self.actor_losses,
                'temperature_losses' : self.temperature_losses,
                'td_errors' : self.td_errors,
                'temperature_values' : self.temperature_values,
                'number_of_steps' : self.number_of_steps
            },
            'update' : {
                'critic_params' : self.critic_params,
                'critic_opt_state' : self.critic_opt_state,
                'critic_target_params' : self.critic_target_params,
                'actor_params' : self.actor_params,
                'actor_opt_state' : self.actor_opt_state,
                'temperature' : self.temperature,
                'temperature_opt_state' : self.temperature_opt_state,
                'buffer' : self.buffer
            }
        }

        with open(file_path, 'wb') as f:
                pickle.dump(agent_state, f)
                
    # Buffer stuff
    def use_prioritized_sampling(self):
        self.buffer.set_uniform_sampling(False)
        
    def use_uniform_sampling(self):
        self.buffer.set_uniform_sampling(True)
        
    def toggle_sampling_mode(self): # check
        current = self.buffer.is_using_uniform_sampling()
        self.buffer.set_uniform_sampling(not current)
        return not current
        
    def get_sampling_mode(self): # check
        return self.buffer.is_using_uniform_sampling()