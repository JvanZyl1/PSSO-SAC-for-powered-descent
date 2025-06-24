import jax
import jax.numpy as jnp
import optax
import pickle
import numpy as np
import datetime
from torch.utils.tensorboard import SummaryWriter

from src.agents.functions.td3_functions import lambda_compile_td3
from src.agents.functions.plotter import agent_plotter_td3
from src.agents.functions.buffers import PERBuffer
from src.agents.functions.networks import DoubleCritic
from src.agents.functions.networks import ClassicalActor as Actor

class TD3:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 flight_phase: str,
                 # Dimensions
                 hidden_dim_actor: int,
                 number_of_hidden_layers_actor: int,
                 hidden_dim_critic: int,
                 number_of_hidden_layers_critic: int,
                 # Hyper-parameters
                 gamma: float,
                 tau: float,
                 alpha_buffer: float,
                 beta_buffer: float,
                 beta_decay_buffer: float,
                 buffer_size: int,
                 trajectory_length: int,
                 batch_size: int,
                 # Learning rates
                 critic_learning_rate: float,
                 actor_learning_rate: float,
                 # Grad max norms
                 critic_grad_max_norm: float,
                 actor_grad_max_norm: float,
                 # TD3 specific
                 policy_noise: float, # STD of Gaussian noise added to the actions
                 noise_clip: float, # Clipping value for the noise i.e. max value
                 policy_delay: int,
                 # Expected updates to convergence
                 expected_updates_to_convergence : int,
                 l2_reg_coef: float):
        
        self.rng_key = jax.random.PRNGKey(0)
        
        self.save_path = f'results/TD3/{flight_phase}/'
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
        self.writer = SummaryWriter(log_dir=f'data/agent_saves/TD3/{flight_phase}/runs/{self.run_id}')

        # Initialize networks
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
        self.critic_target_params = self.critic_params

        # Hyperparam store
        self.critic_learning_rate = critic_learning_rate
        self.critic_grad_max_norm = critic_grad_max_norm
        self.actor_learning_rate = actor_learning_rate
        self.actor_grad_max_norm = actor_grad_max_norm
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.batch_size = batch_size

        self.hidden_dim_actor = hidden_dim_actor
        self.number_of_hidden_layers_actor = number_of_hidden_layers_actor
        self.hidden_dim_critic = hidden_dim_critic
        self.number_of_hidden_layers_critic = number_of_hidden_layers_critic

        self.alpha_buffer = alpha_buffer
        self.beta_buffer = beta_buffer
        self.beta_decay_buffer = beta_decay_buffer
        self.buffer_size = buffer_size
        self.trajectory_length = trajectory_length
        self.batch_size = batch_size
        
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.l2_reg_coef = l2_reg_coef
        # func compiles
        self.critic_optimiser = optax.adam(learning_rate=self.critic_learning_rate)
        self.actor_optimiser = optax.adam(learning_rate=self.actor_learning_rate)
        self.update_function, self.calculate_td_error_lambda, self.critic_warm_up_update_lambda = lambda_compile_td3(
            critic_optimiser=self.critic_optimiser,
            critic=self.critic,
            critic_grad_max_norm=self.critic_grad_max_norm,
            actor_optimiser=self.actor_optimiser,
            actor=self.actor,
            actor_grad_max_norm=self.actor_grad_max_norm,
            gamma=self.gamma,
            tau=self.tau,
            policy_delay=self.policy_delay,
            l2_reg_coef=self.l2_reg_coef
        )

        # Logging
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.number_of_steps_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_mse_loss_episode = 0.0
        self.critic_l2_reg_episode = 0.0
        self.critic_losses = []
        self.critic_mse_losses = []
        self.critic_l2_regs = []
        self.actor_losses = []
        self.td_errors = []
        self.number_of_steps = []

        self.use_prioritized_sampling()

        self.name = 'TD3'
        self.critic_warm_up_step_idx = 0
        self.expected_updates_to_convergence = expected_updates_to_convergence

    def re_init_actor(self, new_actor, new_actor_params):
        self.actor = new_actor
        self.actor_params = new_actor_params
        self.actor_opt_state = optax.adam(learning_rate=self.actor_learning_rate).init(self.actor_params)
        self.update_function, self.calculate_td_error_lambda, self.critic_warm_up_update_lambda = lambda_compile_td3(
            critic_optimiser=self.critic_optimiser,
            critic=self.critic,
            critic_grad_max_norm=self.critic_grad_max_norm,
            actor_optimiser=self.actor_optimiser,
            actor=self.actor,
            actor_grad_max_norm=self.actor_grad_max_norm,
            gamma=self.gamma,
            tau=self.tau,
            policy_delay=self.policy_delay,
            l2_reg_coef=self.l2_reg_coef
        )

    def reset(self):
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.number_of_steps_episode = 0.0
        self.critic_mse_loss_episode = 0.0
        self.critic_l2_reg_episode = 0.0
        self.episode_idx = 0
        self.step_idx = 0
        self.critic_losses = []
        self.critic_mse_losses = []
        self.critic_l2_regs = []
        self.actor_losses = []
        self.td_errors = []
        self.number_of_steps = []
        self.rng_key = jax.random.PRNGKey(0)
        self.buffer.reset()
        self.critic_warm_up_step_idx = 0

    def get_subkey(self):
        self.rng_key, subkey = jax.random.split(self.rng_key)
        return subkey
    
    def get_normal_distributions_batched(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.batch_size, self.action_dim)))
        return normal_distribution
    
    def get_normal_distribution(self):
        normal_distribution = jnp.asarray(jax.random.normal(self.get_subkey(), (self.action_dim, )))
        return normal_distribution

    def select_actions(self, state: jnp.ndarray) -> jnp.ndarray:
        actions = self.actor.apply(self.actor_params, state)
        noise = jnp.clip(
            self.get_normal_distribution()*self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )
        return jnp.squeeze(jnp.clip(actions + noise, -1, 1))

    def select_actions_no_stochastic(self, state: jnp.ndarray) -> jnp.ndarray:
        action = self.actor.apply(self.actor_params, state)
        return jnp.clip(action, -1, 1)

    def calculate_td_error(self,
                          state: jnp.ndarray,
                          action: jnp.ndarray,
                          reward: jnp.ndarray,
                          next_state: jnp.ndarray,
                          done: jnp.ndarray) -> jnp.ndarray:
        next_action = self.actor.apply(self.actor_params, next_state)
        noise = jnp.clip(
            self.get_normal_distribution()*self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )
        next_action = next_action + noise
        return self.calculate_td_error_lambda(
            states=state,
            actions=action,
            rewards=reward,
            next_states=next_state,
            dones=done,
            critic_params=self.critic_params,
            critic_target_params=self.critic_target_params,
            next_actions=next_action
        )
    
    def calculate_td_error_vmap(self,
                               states: jnp.ndarray,
                               actions: jnp.ndarray,
                               rewards: jnp.ndarray,
                               next_states: jnp.ndarray,
                               dones: jnp.ndarray) -> jnp.ndarray:
        states = jnp.reshape(states, (len(states), -1))  # (batch_size, state_dim)
        actions = jnp.reshape(actions, (len(actions), -1))  # (batch_size, action_dim)
        rewards = jnp.reshape(rewards, (len(rewards), 1))  # (batch_size, 1)
        next_states = jnp.reshape(next_states, (len(next_states), -1))  # (batch_size, state_dim)
        dones = jnp.reshape(dones, (len(dones), 1))  # (batch_size, 1)
        next_action_means = jax.vmap(self.actor.apply, in_axes=(None, 0))(self.actor_params, next_states)
        self.rng_key, *subkeys = jax.random.split(self.rng_key, len(next_states) + 1)
        normal_distributions = jax.vmap(lambda key: jax.random.normal(key, (self.action_dim,)))(jnp.array(subkeys))
        next_actions = jax.vmap(lambda mean, normal: normal * self.policy_noise + mean)(
            next_action_means, normal_distributions
        )

        def td_error_calc(state, action, reward, next_state, done, next_action):
            return self.calculate_td_error_lambda(
                states=state,
                actions=action,
                rewards=reward,
                next_states=next_state,
                dones=done,
                critic_params=self.critic_params,
                critic_target_params=self.critic_target_params,
                next_actions=next_action
            )
        td_errors = jax.vmap(td_error_calc)(states, actions, rewards, next_states, dones, next_actions)
        
        return td_errors
    
    def critic_warm_up_step(self):
        states, actions, rewards, next_states, dones, indices, weights_buffer = self.buffer(self.get_subkey())
        clipped_noise = jnp.clip(
            self.get_normal_distributions_batched() * self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )

        step = self.critic_warm_up_step_idx

        # help func
        def log_percentiles(tag, array):
            for q in (1, 5, 50, 95, 99):
                val = jnp.percentile(array, q)
                self.writer.add_scalar(f'CriticWarmUp/{tag}-P{q}', float(val), step)

        # (mean, std, min, max)
        for tag, array in (
            ('States', states),
            ('Actions', actions),
            ('Rewards', rewards),
            ('NextStates', next_states),
            ('Dones', dones),
            ('Weights', weights_buffer),
            ('Noise', clipped_noise),
        ):
            arr = jnp.ravel(array)
            mean = jnp.mean(arr)
            std  = jnp.std(arr)
            mn   = jnp.min(arr)
            mx   = jnp.max(arr)
            self.writer.add_scalar(f'CriticWarmUp/{tag}-Mean', float(mean), step)
            self.writer.add_scalar(f'CriticWarmUp/{tag}-Std',  float(std),  step)
            self.writer.add_scalar(f'CriticWarmUp/{tag}-Min',  float(mn),   step)
            self.writer.add_scalar(f'CriticWarmUp/{tag}-Max',  float(mx),   step)
            
            log_percentiles(tag, arr)

        # (skew, kurtosis)
        def moments(tag, array):
            arr = jnp.ravel(array)
            mu  = jnp.mean(arr)
            std = jnp.std(arr)
            skew = jnp.mean((arr - mu)**3) / (std**3 + 1e-12)
            kurt = jnp.mean((arr - mu)**4) / (std**4 + 1e-12) - 3
            self.writer.add_scalar(f'CriticWarmUp/{tag}-Skew',  float(skew),  step)
            self.writer.add_scalar(f'CriticWarmUp/{tag}-Kurtosis', float(kurt), step)

        for tag, array in (('States', states), ('Actions', actions), ('Rewards', rewards), ('NextStates', next_states)):
            moments(tag, array)

        # Histograms for distributions
        for tag, array in (
            ('States', states),
            ('Actions', actions),
            ('Rewards', rewards),
            ('NextStates', next_states),
            ('Dones', dones),
            ('Weights', weights_buffer),
            ('Noise', clipped_noise),
        ):
            self.writer.add_histogram(f'CriticWarmUp/{tag}-Hist', np.array(array), step)

        # Critic warmup
        self.critic_params, self.critic_opt_state, self.critic_target_params, critic_loss, td_errors, critic_mse_loss, critic_l2_reg = (
            self.critic_warm_up_update_lambda(
                actor_params         = self.actor_params,
                states               = states,
                actions              = actions,
                rewards              = rewards,
                next_states          = next_states,
                dones                = dones,
                buffer_weights       = weights_buffer,
                critic_params        = self.critic_params,
                critic_target_params = self.critic_target_params,
                critic_opt_state     = self.critic_opt_state,
                clipped_noise        = clipped_noise
            )
        )

        # Log
        self.writer.add_scalar('CriticWarmUp/Loss', float(critic_loss), step)
        self.writer.add_scalar('CriticWarmUp/MSELoss', float(jnp.mean(critic_mse_loss)), step)
        self.writer.add_scalar('CriticWarmUp/L2Reg', float(critic_l2_reg), step)
        self.writer.add_scalar('CriticWarmUp/TDError', float(np.mean(td_errors)), step)

        self.critic_warm_up_step_idx += 1
        return float(critic_loss), float(jnp.mean(critic_mse_loss)), float(critic_l2_reg)

    def update_episode(self):
        self.critic_losses.append(self.critic_loss_episode)
        self.actor_losses.append(self.actor_loss_episode)
        self.td_errors.append(self.td_errors_episode)
        self.number_of_steps.append(self.number_of_steps_episode)
        self.critic_mse_losses.append(self.critic_mse_loss_episode)
        self.critic_l2_regs.append(self.critic_l2_reg_episode)

        # Log
        self.writer.add_scalar('Episode/CriticLoss', np.array(self.critic_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/ActorLoss', np.array(self.actor_loss_episode), self.episode_idx)
        self.writer.add_histogram('Episode/TDError', np.array(self.td_errors_episode), self.episode_idx)
        self.writer.add_scalar('Episode/NumberOfSteps', np.array(self.number_of_steps_episode), self.episode_idx)
        self.writer.add_scalar('Episode/CriticMSELoss', np.array(self.critic_mse_loss_episode), self.episode_idx)
        self.writer.add_scalar('Episode/CriticL2Reg', np.array(self.critic_l2_reg_episode), self.episode_idx)
        # Reset
        self.critic_loss_episode = 0.0
        self.actor_loss_episode = 0.0
        self.td_errors_episode = 0.0
        self.number_of_steps_episode = 0.0
        self.critic_mse_loss_episode = 0.0
        self.critic_l2_reg_episode = 0.0
        self.episode_idx += 1

    def update(self):
        states, actions, rewards, next_states, dones, index, weights_buffer = self.buffer(self.get_subkey())

        clipped_noise = jnp.clip(
            self.get_normal_distributions_batched()*self.policy_noise,
            -self.noise_clip,
            self.noise_clip
        )

        self.critic_params, self.critic_opt_state, critic_loss, td_errors, \
        self.actor_params, self.actor_opt_state, actor_loss, \
        self.critic_target_params, critic_mse_loss, critic_l2_reg = self.update_function(
            actor_params=self.actor_params,
            actor_opt_state=self.actor_opt_state,
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            buffer_weights=weights_buffer,
            critic_params=self.critic_params,
            critic_target_params=self.critic_target_params,
            critic_opt_state=self.critic_opt_state,
            clipped_noise = clipped_noise,
            step=self.step_idx
        )

        # Update buffer priorities
        self.buffer.update_priorities(index, td_errors)

        # Metrics
        self.critic_loss_episode += critic_loss
        self.actor_loss_episode += actor_loss
        self.td_errors_episode += td_errors
        self.number_of_steps_episode += 1
        self.critic_mse_loss_episode += jnp.mean(critic_mse_loss)
        self.critic_l2_reg_episode += critic_l2_reg
        self.step_idx += 1
        self.writer.add_scalar('Steps/CriticLoss', np.array(critic_loss), self.step_idx)
        if np.array(actor_loss).all() != 0.0:
            self.writer.add_scalar('Steps/ActorLoss', np.array(actor_loss), self.step_idx)
        self.writer.add_scalar('Steps/TDError/mean', np.mean(np.array(td_errors)), self.step_idx)
        self.writer.add_scalar('Steps/TDError/std', np.std(np.array(td_errors)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/mean', np.mean(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/std', np.std(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/max', np.max(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/BufferWeights/min', np.min(np.array(weights_buffer)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/mean', np.mean(np.array(clipped_noise)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/std', np.std(np.array(clipped_noise)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/max', np.max(np.array(clipped_noise)), self.step_idx)
        self.writer.add_scalar('Steps/Noise/min', np.min(np.array(clipped_noise)), self.step_idx)
        self.writer.add_histogram('Steps/SampledBufferWeights', np.array(weights_buffer), self.step_idx)
        self.writer.add_histogram('Steps/SampledStates', np.array(states), self.step_idx)
        self.writer.add_histogram('Steps/SampledActions', np.array(actions), self.step_idx)
        self.writer.add_histogram('Steps/SampledRewards', np.array(rewards), self.step_idx)
        self.writer.add_histogram('Steps/SampledNextStates', np.array(next_states), self.step_idx)
        self.writer.add_histogram('Steps/SampledDones', np.array(dones), self.step_idx)
        self.writer.add_scalar('Steps/SampledActions-Mean', np.mean(np.array(actions)), self.step_idx)
        self.writer.add_scalar('Steps/SampledActions-Std', np.std(np.array(actions)), self.step_idx)
        self.writer.add_scalar('Steps/SampledActions-Max', np.max(np.array(actions)), self.step_idx)
        self.writer.add_scalar('Steps/SampledActions-Min', np.min(np.array(actions)), self.step_idx)
        self.writer.add_scalar('Steps/SampledStates-Mean', np.mean(np.array(states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledStates-Std', np.std(np.array(states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledStates-Max', np.max(np.array(states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledStates-Min', np.min(np.array(states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledRewards-Mean', np.mean(np.array(rewards)), self.step_idx)
        self.writer.add_scalar('Steps/SampledRewards-Std', np.std(np.array(rewards)), self.step_idx)
        self.writer.add_scalar('Steps/SampledRewards-Max', np.max(np.array(rewards)), self.step_idx)
        self.writer.add_scalar('Steps/SampledRewards-Min', np.min(np.array(rewards)), self.step_idx)
        self.writer.add_scalar('Steps/SampledNextStates-Mean', np.mean(np.array(next_states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledNextStates-Std', np.std(np.array(next_states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledNextStates-Max', np.max(np.array(next_states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledNextStates-Min', np.min(np.array(next_states)), self.step_idx)
        self.writer.add_scalar('Steps/SampledDones-Mean', np.mean(np.array(dones)), self.step_idx)
        self.writer.add_scalar('Steps/SampledDones-Std', np.std(np.array(dones)), self.step_idx)
        self.writer.add_scalar('Steps/SampledDones-Max', np.max(np.array(dones)), self.step_idx)
        self.writer.add_scalar('Steps/SampledDones-Min', np.min(np.array(dones)), self.step_idx)
        q1, q2 = self.critic.apply(jax.lax.stop_gradient(self.critic_params), jax.lax.stop_gradient(states), actions) 
        self.writer.add_scalar('Steps/QValue-Q1-Mean', np.mean(np.array(q1)), self.step_idx)
        self.writer.add_scalar('Steps/QValue-Q1-Std', np.std(np.array(q1)), self.step_idx)
        self.writer.add_scalar('Steps/QValue-Q1-Max', np.max(np.array(q1)), self.step_idx)
        self.writer.add_scalar('Steps/QValue-Q1-Min', np.min(np.array(q1)), self.step_idx)
        self.writer.add_scalar('Steps/QValue-Q2-Mean', np.mean(np.array(q2)), self.step_idx)
        self.writer.add_scalar('Steps/QValue-Q2-Std', np.std(np.array(q2)), self.step_idx)
        self.writer.add_scalar('Steps/QValue-Q2-Max', np.max(np.array(q2)), self.step_idx)
        self.writer.add_scalar('Steps/QValue-Q2-Min', np.min(np.array(q2)), self.step_idx)
        self.writer.add_scalar('Steps/CriticLoss-MSE', float(critic_mse_loss), self.step_idx)
        self.writer.add_scalar('Steps/CriticLoss-L2Reg', float(critic_l2_reg), self.step_idx)  

    def plotter(self):
        agent_plotter_td3(self)

    def save(self):
        file_path = f'data/agent_saves/TD3/{self.flight_phase}/saves/td3.pkl'
        agent_state = {
            'inputs': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'flight_phase': self.flight_phase,
                'hidden_dim_actor': self.hidden_dim_actor,
                'number_of_hidden_layers_actor': self.number_of_hidden_layers_actor,
                'hidden_dim_critic': self.hidden_dim_critic,
                'number_of_hidden_layers_critic': self.number_of_hidden_layers_critic,
                'gamma': self.gamma,
                'tau': self.tau,
                'alpha_buffer': self.alpha_buffer,
                'beta_buffer': self.beta_buffer,
                'beta_decay_buffer': self.beta_decay_buffer,
                'buffer_size': self.buffer_size,
                'trajectory_length': self.trajectory_length,
                'batch_size': self.batch_size,
                'critic_learning_rate': self.critic_learning_rate,
                'actor_learning_rate': self.actor_learning_rate,
                'critic_grad_max_norm': self.critic_grad_max_norm,
                'actor_grad_max_norm': self.actor_grad_max_norm,
                'policy_noise': self.policy_noise,
                'noise_clip': self.noise_clip,
                'policy_delay': self.policy_delay,
                'expected_updates_to_convergence': self.expected_updates_to_convergence,
                'l2_reg_coef': self.l2_reg_coef
            },
            'misc': {
                'rng_key': self.rng_key,
                'run_id': self.run_id
            },
            'logging': {
                'critic_loss_episode': self.critic_loss_episode,
                'actor_loss_episode': self.actor_loss_episode,
                'td_errors_episode': self.td_errors_episode,
                'number_of_steps_episode': self.number_of_steps_episode,
                'episode_idx': self.episode_idx,
                'step_idx': self.step_idx,
                'critic_losses': self.critic_losses,
                'actor_losses': self.actor_losses,
                'td_errors': self.td_errors,
                'number_of_steps': self.number_of_steps
            },
            'update': {
                'critic_params': self.critic_params,
                'critic_opt_state': self.critic_opt_state,
                'critic_target_params': self.critic_target_params,
                'actor_params': self.actor_params,
                'actor_opt_state': self.actor_opt_state,
                'buffer': self.buffer
            }
        }

        with open(file_path, 'wb') as f:
            pickle.dump(agent_state, f)

    def use_prioritized_sampling(self):
        self.buffer.set_uniform_sampling(False)
        
    def use_uniform_sampling(self):
        self.buffer.set_uniform_sampling(True)
        
    def toggle_sampling_mode(self): # Check
        current = self.buffer.is_using_uniform_sampling()
        self.buffer.set_uniform_sampling(not current)
        return not current
        
    def get_sampling_mode(self): # check
        return self.buffer.is_using_uniform_sampling() 