import os
import torch
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lmdb
import pickle

def moving_average(var, window_size=5):
    window_size = min(5, len(var))
    moving_avg = []
    for i in range(len(var)):
        if i < window_size:
            moving_avg.append(sum(var[:i+1]) / (i+1))
        else:
            moving_avg.append(sum(var[i-window_size+1:i+1]) / window_size)
    return moving_avg


# Parent Trainer class
class TrainerSkeleton:
    def __init__(self,
                 env,
                 agent,
                 load_buffer_from_experiences_bool : bool,
                 flight_phase : str,
                 num_episodes: int,
                 save_interval: int = 10,
                 critic_warm_up_steps: int = 0,
                 critic_warm_up_early_stopping_loss: float = 0.0,
                 update_agent_every_n_steps: int = 10,
                 priority_update_interval: int = 5):
        self.env = env
        self.agent = agent
        self.gamma = agent.gamma
        self.num_episodes = num_episodes
        self.buffer_size = agent.buffer.buffer_size
        self.save_interval = save_interval
        self.dt = self.env.dt
        self.critic_warm_up_steps = critic_warm_up_steps
        self.epoch_rewards = []
        self.flight_phase = flight_phase
        self.load_buffer_from_experiences_bool = load_buffer_from_experiences_bool
        self.update_agent_every_n_steps = update_agent_every_n_steps
        self.critic_warm_up_early_stopping_loss = critic_warm_up_early_stopping_loss
        self.priority_update_interval = priority_update_interval
        if flight_phase in ['landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
            self.altitudes_validation = []
            self.rewards_validation = []
            self.test_steps = 0
            self.test_steps_array = []
        self.rewards_list = []
        self.episode_rewards_mean = []
        self.episode_rewards_std = []
        self.episode_rewards_max = []
        self.episode_rewards_min = []

    def plot_rewards(self):
        save_path_rewards = self.agent.save_path + 'rewards.png'
        
        # Calculate moving average
        window_size = min(5, len(self.epoch_rewards))
        moving_avg = []
        for i in range(len(self.epoch_rewards)):
            if i < window_size:
                moving_avg.append(sum(self.epoch_rewards[:i+1]) / (i+1))
            else:
                moving_avg.append(sum(self.epoch_rewards[i-window_size+1:i+1]) / window_size)
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.epoch_rewards, label="Episode Rewards", alpha=0.5, linewidth=4, color = 'pink', linestyle = '--')
        plt.plot(moving_avg, 
                label=f"{window_size}-Episode Moving Average",
                linewidth=4,
                color = 'blue')
        plt.xlabel("Episodes", fontsize = 20)
        plt.ylabel("Total Reward", fontsize = 20)
        plt.title("Rewards Over Training", fontsize = 22)
        plt.legend(fontsize = 20)
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.grid()
        plt.savefig(save_path_rewards, format='png', dpi=300)
        plt.close()

    def test_landing_burn(self):
        reward_total, y_array = self.test_env()
        self.altitudes_validation.append(y_array)
        self.rewards_validation.append(reward_total)
        self.test_steps += 1
        self.test_steps_array.append(self.test_steps)
        plt.figure(figsize=(20, 15))
        plt.suptitle('Validation Plots', fontsize = 22)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
        ax1 = plt.subplot(gs[0])
        ax1.plot(self.test_steps_array, self.altitudes_validation, color = 'blue', linewidth = 4)
        ax1.set_xlabel('Steps', fontsize = 20)
        ax1.set_ylabel('Altitude', fontsize = 20)
        ax1.set_title('Altitude Validation', fontsize = 22)
        ax1.grid()
        ax1.tick_params(axis='both', which='major', labelsize=16)
        
        ax2 = plt.subplot(gs[1])
        ax2.plot(self.test_steps_array, self.rewards_validation, color = 'red', linewidth = 4)
        ax2.set_xlabel('Steps', fontsize = 20)
        ax2.set_ylabel('Reward', fontsize = 20)
        ax2.set_title('Reward Validation', fontsize = 22)
        ax2.grid()
        ax2.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(self.agent.save_path + 'validation_plot.png', format='png', dpi=300)
        plt.close()

    def save_all(self):
        self.plot_rewards()
        self.agent.plotter()
        self.agent.save()
        self.plot_episode_rewards()
        if hasattr(self, 'test_env'):
            if self.flight_phase in ['landing_burn', 'landing_burn_ACS', 'landing_burn_pure_throttle', 'landing_burn_pure_throttle_Pcontrol']:
                self.test_landing_burn()
            else:
                self.test_env()

    def add_experiences_to_buffer(self):
        folder_path=f"data/experience_buffer/{self.flight_phase}/experience_buffer.lmdb"
        env = lmdb.open(folder_path, readonly=True, lock=False)  # Open the folder, not the .mdb file
        
        non_zero_experiences = int(jnp.sum(jnp.any(self.agent.buffer.buffer != 0, axis=1)))
        remaining_experiences = self.buffer_size - non_zero_experiences
        
        batch_size = 1000
        experiences_batch = []
        
        pbar = tqdm(total=remaining_experiences, desc="Loading experiences from file")
        with env.begin() as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                if non_zero_experiences >= self.buffer_size:
                    break
                    
                experience = pickle.loads(value)
                experiences_batch.append(experience)
                
                if len(experiences_batch) >= batch_size or non_zero_experiences + len(experiences_batch) >= self.buffer_size:
                    
                    states = jnp.array([exp[0] for exp in experiences_batch])
                    actions = jnp.array([exp[1].detach().numpy() for exp in experiences_batch])
                    rewards = jnp.array([exp[2] for exp in experiences_batch])
                    next_states = jnp.array([exp[3] for exp in experiences_batch])
                    dones = jnp.zeros(len(experiences_batch))
                    
                    td_errors = self.agent.calculate_td_error_vmap(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        dones=dones
                    )
                    
                    for i in range(len(experiences_batch)):
                        self.agent.buffer.add(
                            state=states[i],
                            action=actions[i],
                            reward=rewards[i],
                            next_state=next_states[i],
                            done=dones[i],
                            td_error=td_errors[i]
                        )
                        non_zero_experiences += 1
                        pbar.update(1)
                    
                    indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                                      self.agent.buffer.position)
                    self.agent.buffer.update_priorities(indices, td_errors)
                    
                    experiences_batch = []
        
        if experiences_batch:
            states = jnp.array([exp[0] for exp in experiences_batch])
            actions = jnp.array([exp[1].detach().numpy() for exp in experiences_batch])
            rewards = jnp.array([exp[2] for exp in experiences_batch])
            next_states = jnp.array([exp[3] for exp in experiences_batch])
            dones = jnp.zeros(len(experiences_batch))
            
            td_errors = self.agent.calculate_td_error_vmap(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            
            for i in range(len(experiences_batch)):
                self.agent.buffer.add(
                    state=states[i],
                    action=actions[i],
                    reward=rewards[i],
                    next_state=next_states[i],
                    done=dones[i],
                    td_error=td_errors[i]
                )
                non_zero_experiences += 1
                pbar.update(1)
            
            indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                               self.agent.buffer.position)
            self.agent.buffer.update_priorities(indices, td_errors)
        
        buffer_save_path = f'data/agent_saves/VanillaSAC/{self.flight_phase}/saves/buffer_after_loading.pkl'
        os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)

        buffer_state = {
            'buffer': self.agent.buffer.buffer,
            'priorities': self.agent.buffer.priorities,
            'n_step_buffer': self.agent.buffer.n_step_buffer,
            'position': self.agent.buffer.position,
            'beta': self.agent.buffer.beta
        }

        with open(buffer_save_path, 'wb') as f:
            pickle.dump(buffer_state, f)
        print(f"Saved complete buffer state to {buffer_save_path}")

    def fill_replay_buffer(self):
        if self.load_buffer_from_experiences_bool == True:
            self.add_experiences_to_buffer()
        else:
            self.load_buffer_from_rl()
        
        non_zero_experiences = int(jnp.sum(jnp.any(self.agent.buffer.buffer != 0, axis=1)))

        remaining_experiences = self.buffer_size - non_zero_experiences
        
        batch_size = 100
        experiences_batch = []
        
        pbar = tqdm(total=remaining_experiences, desc="Filling replay buffer")
        while non_zero_experiences < self.buffer_size:
            state = self.env.reset()
            done = False
            truncated = False
            
            while not (done or truncated):
                action = self.agent.select_actions(jnp.expand_dims(state, 0)) 
                action = jnp.array(action)
                if action.ndim == 0:
                    action = jnp.expand_dims(action, 0)
                
                next_state, reward, done, truncated, _ = self.env.step(action)
                done_or_truncated = done or truncated
                
                experiences_batch.append((state, action, reward, next_state, done_or_truncated))
                non_zero_experiences += 1
                pbar.update(1)
                state = next_state
                
                if len(experiences_batch) >= batch_size or non_zero_experiences >= self.buffer_size:
                    states = jnp.array([exp[0] for exp in experiences_batch])
                    actions = jnp.array([exp[1] for exp in experiences_batch])
                    rewards = jnp.array([exp[2] for exp in experiences_batch])
                    next_states = jnp.array([exp[3] for exp in experiences_batch])
                    dones = jnp.array([exp[4] for exp in experiences_batch])
                    
                    states = jnp.reshape(states, (len(experiences_batch), -1))  # (batch_size, state_dim)
                    actions = jnp.reshape(actions, (len(experiences_batch), -1))  # (batch_size, action_dim)
                    rewards = jnp.reshape(rewards, (len(experiences_batch), 1))  # (batch_size, 1)
                    next_states = jnp.reshape(next_states, (len(experiences_batch), -1))  # (batch_size, state_dim)
                    dones = jnp.reshape(dones, (len(experiences_batch), 1))  # (batch_size, 1)
                    
                    td_errors = self.agent.calculate_td_error_vmap(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        next_states=next_states,
                        dones=dones
                    )
                    
                    for i in range(len(experiences_batch)):
                        self.agent.buffer.add(
                            state=states[i],
                            action=actions[i],
                            reward=rewards[i][0],
                            next_state=next_states[i],
                            done=dones[i][0],
                            td_error=jnp.squeeze(td_errors[i])
                        )
                    
                    indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                                      self.agent.buffer.position)
                    self.agent.buffer.update_priorities(indices, td_errors)
                    
                    experiences_batch = []
                
                if non_zero_experiences >= self.buffer_size:
                    break
        
        if experiences_batch:
            states = jnp.array([exp[0] for exp in experiences_batch])
            actions = jnp.array([exp[1] for exp in experiences_batch])
            rewards = jnp.array([exp[2] for exp in experiences_batch])
            next_states = jnp.array([exp[3] for exp in experiences_batch])
            dones = jnp.array([exp[4] for exp in experiences_batch])
            
            states = jnp.reshape(states, (len(experiences_batch), -1))  # Reshape to (batch_size, state_dim)
            actions = jnp.reshape(actions, (len(experiences_batch), -1))  # Reshape to (batch_size, action_dim)
            rewards = jnp.reshape(rewards, (len(experiences_batch), 1))  # Reshape to (batch_size, 1)
            next_states = jnp.reshape(next_states, (len(experiences_batch), -1))  # Reshape to (batch_size, state_dim)
            dones = jnp.reshape(dones, (len(experiences_batch), 1))  # Reshape to (batch_size, 1)
            
            td_errors = self.agent.calculate_td_error_vmap(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                dones=dones
            )
            
            for i in range(len(experiences_batch)):
                self.agent.buffer.add(
                            state=states[i],
                            action=actions[i],
                            reward=rewards[i][0],
                            next_state=next_states[i],
                            done=dones[i][0],
                            td_error=jnp.squeeze(td_errors[i])
                        )
            
            indices = jnp.arange(self.agent.buffer.position - len(experiences_batch), 
                               self.agent.buffer.position)
            self.agent.buffer.update_priorities(indices, td_errors)
        
        self.save_buffer()
    def save_buffer(self):
        buffer_save_path = f'data/buffer_saves/buffer_{self.flight_phase}.pkl'
        os.makedirs(os.path.dirname(buffer_save_path), exist_ok=True)

        buffer_state = {
            'buffer': self.agent.buffer.buffer,
            'priorities': self.agent.buffer.priorities,
            'n_step_buffer': self.agent.buffer.n_step_buffer,
            'position': self.agent.buffer.position,
            'beta': self.agent.buffer.beta
        }

        with open(buffer_save_path, 'wb') as f:
            pickle.dump(buffer_state, f)
        print(f"Saved complete buffer state to {buffer_save_path}")

    def load_buffer_from_rl(self):
        buffer_save_path = f'data/buffer_saves/buffer_{self.flight_phase}.pkl'
        try:
            with open(buffer_save_path, 'rb') as f:
                buffer_state = pickle.load(f)
                
                if isinstance(buffer_state, dict):
                    buffer_length = len(buffer_state['buffer'])
                    indices = jnp.arange(buffer_length)
                    self.agent.buffer.buffer = self.agent.buffer.buffer.at[indices].set(buffer_state['buffer'][:buffer_length])
                    self.agent.buffer.priorities = self.agent.buffer.priorities.at[indices].set(buffer_state['priorities'][:buffer_length])
                    n_step_length = len(buffer_state['n_step_buffer'])
                    n_step_indices = jnp.arange(n_step_length)
                    self.agent.buffer.n_step_buffer = self.agent.buffer.n_step_buffer.at[n_step_indices].set(buffer_state['n_step_buffer'][:n_step_length])
                    self.agent.buffer.position = int(buffer_length)
                    self.agent.buffer.beta = float(buffer_state['beta'])
                else: # legacy
                    self.agent.buffer = buffer_state
            
            print(f"Loaded buffer from {buffer_save_path}")
            number_of_experiences = len(buffer_state['buffer'])
            print(f"Number of experiences in buffer: {number_of_experiences}")
        except FileNotFoundError:
            print(f"Buffer file not found at {buffer_save_path}. Please ensure the file exists.")

    def calculate_td_error(self,
                           states,
                           actions,
                           rewards,
                           next_states,
                           dones):
        pass # Placeholder for child classes

    def critic_warm_up(self):
        pbar = tqdm(range(1, self.critic_warm_up_steps + 1), desc="Critic Warm Up Progress")
        for _ in pbar:
            critic_warm_up_loss = self.agent.critic_warm_up_step()
            pbar.set_description(f"Critic Warm Up Progress - Loss: {critic_warm_up_loss:.4e}")
            if critic_warm_up_loss < self.critic_warm_up_early_stopping_loss:
                break

    def update_episode_rewards(self):
        self.episode_rewards_mean.append(np.mean(np.array(self.rewards_list)))
        self.episode_rewards_std.append(np.std(np.array(self.rewards_list)))
        self.episode_rewards_max.append(np.max(np.array(self.rewards_list)))
        self.episode_rewards_min.append(np.min(np.array(self.rewards_list)))
        self.rewards_list = []

    def plot_episode_rewards(self):
        self.update_episode_rewards()

        # Create uncertainty plot
        plt.figure(figsize=(10, 5))
        episodes = np.arange(len(self.episode_rewards_mean))
        plt.plot(episodes, self.episode_rewards_mean, label="Mean Reward", linewidth=4, color='blue')
        plt.fill_between(episodes, self.episode_rewards_min, self.episode_rewards_max,
                       facecolor='C0', alpha=0.20,
                       label='min-max')
        plt.fill_between(episodes, np.array(self.episode_rewards_mean) - np.array(self.episode_rewards_std), 
                        np.array(self.episode_rewards_mean) + np.array(self.episode_rewards_std),
                       facecolor='C0', alpha=0.45,
                       label=r'$\pm 1 \sigma$')
        plt.xlabel("Episode", fontsize=20)
        plt.ylabel("Reward", fontsize=20)
        plt.title("Episode Rewards", fontsize=22)
        plt.grid(True)
        plt.legend(fontsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.savefig(self.agent.save_path + "rewards_uncertainty.png", bbox_inches='tight')
        plt.close()

    def train(self):
        self.fill_replay_buffer()

        pbar = tqdm(range(1, self.num_episodes + 1), desc="Training Progress")

        self.critic_warm_up()
        steps_since_last_update = 0
        
        total_num_steps = 0
        for episode in pbar:
            state = self.env.reset()
            done = False
            truncated = False
            total_reward = 0
            num_steps = 0
            episode_time = 0.0

            while not (done or truncated):
                steps_since_last_update += 1
                # Sample action from the agent, use sample actions function as a stochastic policy
                action = self.agent.select_actions(jnp.expand_dims(state, 0))  # Add batch dimension for input

                next_state, reward, done, truncated, _ = self.env.step(action)
                done_or_truncated = done or truncated

                state_jnp = jnp.array(state)
                action_jnp = jnp.array(action)
                reward_jnp = jnp.array(reward)
                next_state_jnp = jnp.array(next_state)
                done_jnp = jnp.array(done_or_truncated)

                if action_jnp.ndim == 0:
                    print(f'action_jnp: {action_jnp}')
                    action_jnp = jnp.expand_dims(action_jnp, axis=0)

                if self.agent.name == 'VanillaSAC':
                    td_error = self.calculate_td_error(
                        jnp.expand_dims(state_jnp, axis=0),
                        action_jnp, # td3 : jnp.expand_dims(action_jnp, axis=0)
                        jnp.expand_dims(reward_jnp, axis=0),
                        jnp.expand_dims(next_state_jnp, axis=0),
                        jnp.expand_dims(done_jnp, axis=0)
                    )
                elif self.agent.name == 'TD3':
                    td_error = self.calculate_td_error(
                        jnp.expand_dims(state_jnp, axis=0),
                        jnp.expand_dims(action_jnp, axis=0) ,
                        jnp.expand_dims(reward_jnp, axis=0),
                        jnp.expand_dims(next_state_jnp, axis=0),
                        jnp.expand_dims(done_jnp, axis=0)
                    )
                else:
                    raise ValueError(f'Invalid agent name: {self.agent.name}')

                if self.agent.name == 'VanillaSAC':
                    self.agent.buffer.add(
                        state=state_jnp,
                        action=jnp.squeeze(action_jnp),
                        reward=reward_jnp,
                        next_state=next_state_jnp,
                        done=done_jnp,
                        td_error= jnp.squeeze(td_error)
                    )
                elif self.agent.name == 'TD3':
                    self.agent.buffer.add(
                        state=state_jnp,
                        action=action_jnp,
                        reward=reward_jnp,
                        next_state=next_state_jnp,
                        done=done_jnp,
                        td_error= jnp.squeeze(td_error)
                    )
                else:
                    raise ValueError(f'Invalid agent name: {self.agent.name}')

                if steps_since_last_update % self.update_agent_every_n_steps == 0 and steps_since_last_update != 0:
                    self.agent.update()
                    steps_since_last_update = 0

                state = next_state_jnp
                total_reward += reward_jnp
                num_steps += 1
                episode_time += self.dt
                total_num_steps += 1
                self.agent.writer.add_scalar('Rewards/Reward-per-step', np.array(reward_jnp), total_num_steps)
                self.rewards_list.append(reward_jnp)

                # If done:
                if done_or_truncated:
                    self.agent.update_episode()

            self.epoch_rewards.append(total_reward)
            self.agent.writer.add_scalar('Rewards/Reward-per-episode', np.array(total_reward), episode)
            self.agent.writer.add_scalar('Rewards/Episode-time', np.array(episode_time), episode)
            pbar.set_description(f"Training Progress - Episode: {episode}, Total Reward: {total_reward:.4e}, Num Steps: {num_steps}:")
                
            if episode % self.save_interval == 0:
                self.save_all()
                
            self.agent.writer.flush()

        self.save_all()
        print("Training complete.")

    def plot_final_run(self):
        pass

    def test_env(self):
        pass

#### Reinforcement Learning Trainer ####
class TrainerRL(TrainerSkeleton):
    """
    Generic trainer class for reinforcement learning agents (SAC, TD3, etc.).
    """
    def __init__(self,
                 env,
                 agent,
                 flight_phase: str,
                 num_episodes: int,
                 save_interval: int = 10,
                 critic_warm_up_steps: int = 0,
                 critic_warm_up_early_stopping_loss: float = 0.0,
                 load_buffer_from_experiences_bool: bool = False,
                 update_agent_every_n_steps: int = 10,
                 priority_update_interval: int = 25):
        super(TrainerRL, self).__init__(
            env, agent, load_buffer_from_experiences_bool, flight_phase, num_episodes,
            save_interval, critic_warm_up_steps, critic_warm_up_early_stopping_loss,
            update_agent_every_n_steps, priority_update_interval
        )

    def calculate_td_error(self,
                          states,
                          actions,
                          rewards,
                          next_states,
                          dones):
        return self.agent.calculate_td_error(states, actions, rewards, next_states, dones)
    
    def plot_critic_warmup(self, critic_warmup_losses, critic_warmup_mse_losses, critic_warmup_l2_regs):
        plt.figure(figsize=(10, 5))
        # Plot with a y log scale
        plt.yscale('log')
        plt.plot(critic_warmup_losses, label='Critic Warmup Loss', color='blue', linewidth=4)
        plt.plot(critic_warmup_mse_losses, label='Critic Warmup MSE Loss', color='red', linewidth=4)
        plt.plot(critic_warmup_l2_regs, label='Critic Warmup L2 Reg', color='green', linewidth=4)
        plt.xlabel('Step', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.title('Critic Warmup Loss', fontsize=22)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=20)
        plt.grid(True)
        plt.savefig(f'results/{self.agent.name}/{self.flight_phase}/critic_warmup_loss.png')
        plt.close()
    
    def critic_warm_up(self):
        pbar = tqdm(range(1, self.critic_warm_up_steps + 1), desc="Critic Warm Up Progress")
        critic_warmup_losses = []
        critic_warmup_mse_losses = []
        critic_warmup_l2_regs = []
        for _ in pbar:
            critic_warmup_loss, critic_warmup_mse_loss, critic_warmup_l2_reg = self.agent.critic_warm_up_step()
            pbar.set_description(f"Critic Warm Up Progress - Loss: {critic_warmup_loss:.4e}")
            critic_warmup_losses.append(critic_warmup_loss)
            critic_warmup_mse_losses.append(critic_warmup_mse_loss)
            critic_warmup_l2_regs.append(critic_warmup_l2_reg)
            if critic_warmup_loss < self.critic_warm_up_early_stopping_loss:
                break
        
        self.save_buffer()
        self.plot_critic_warmup(critic_warmup_losses, critic_warmup_mse_losses, critic_warmup_l2_regs)