import os
import numpy as np
from typing import Tuple
from datetime import datetime
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer: # This is the uniform buffer which Mnih et al. used in their original paper on Atari games
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0
        
        # Preallocate buffers
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices])
        )
    
    def __len__(self):
        return self.size

class PrioritizedReplayBuffer: # This is the prioritized buffer which Schaul et al. used in their original paper on Atari games
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6, beta: float = 0.4, beta_annealing_steps: int = 100000, epsilon: float = 1e-6):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.position = 0
        self.size = 0
        
        # PER hyperparameters
        self.alpha = alpha  # Priority exponent 
        self.beta = beta    # Importance sampling correction factor
        self.beta_increment = (1.0 - beta) / beta_annealing_steps
        self.epsilon = epsilon  # Small constant
        
        # Initialise the buffers
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        # Initialise new experiences with max priority
        self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int):
        if self.size == 0:
            return None
        
        # Calculate probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32).reshape(-1, 1)
        
        # Anneal the beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.dones[indices]),
            torch.FloatTensor(weights),
            indices
        )
    
    def update_priorities(self, indices, td_errors):
        # convert from (batch_size, 1) to (batch_size,))
        priorities = np.abs(td_errors.squeeze()) + self.epsilon
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
    
    def __len__(self):
        return self.size

class Actor(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256, 
        n_hidden_layers: int = 2,
        max_action: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        

        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            
        self.shared_net = nn.Sequential(*layers)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        features = self.shared_net(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(state)
        if deterministic:
            action = torch.tanh(mean)
            log_prob = None
        else:
            std = log_std.exp()
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()
            action = torch.tanh(x_t)
            log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6) # log-determinant of Jacobian correction for tanh squashing function
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action * self.max_action, log_prob

class Critic(nn.Module):
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 256, 
        n_hidden_layers: int = 2
    ):
        super(Critic, self).__init__()
        
        # Q1
        self.q1_layers = []
        self.q1_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        self.q1_layers.append(nn.ReLU())
        
        for _ in range(n_hidden_layers - 1):
            self.q1_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.q1_layers.append(nn.ReLU())
            
        self.q1_layers.append(nn.Linear(hidden_dim, 1))
        self.q1 = nn.Sequential(*self.q1_layers)

        # Q2
        self.q2_layers = []
        self.q2_layers.append(nn.Linear(state_dim + action_dim, hidden_dim))
        self.q2_layers.append(nn.ReLU())
        
        for _ in range(n_hidden_layers - 1):
            self.q2_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.q2_layers.append(nn.ReLU())
            
        self.q2_layers.append(nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(*self.q2_layers)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        state_action = torch.cat([state, action], dim=1)
        
        q1 = self.q1(state_action)
        q2 = self.q2(state_action)
        
        return q1, q2
    
    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state_action = torch.cat([state, action], dim=1)
        return self.q1(state_action)

class SACPyTorch:
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        # Actor network
        hidden_dim_actor: int = 256,
        number_of_hidden_layers_actor: int = 2,
        # Critic network
        hidden_dim_critic: int = 256,
        number_of_hidden_layers_critic: int = 2,
        # SAC hyperparameters
        alpha_initial: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000,
        batch_size: int = 256,
        critic_learning_rate: float = 3e-4,
        actor_learning_rate: float = 3e-4,
        alpha_learning_rate: float = 3e-4,
        # Action bounds
        max_action: float = 1.0,
        # Device
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        # Flag for automatic entropy tuning
        auto_entropy_tuning: bool = True,
        # Flight phase
        flight_phase: str = "default",
        # Save frequency
        save_stats_frequency: int = 100,
        # Replay buffer 
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_annealing_steps: int = 100000,
        per_epsilon: float = 1e-6,
        # L1 or L2 loss
        use_l1_loss: bool = False,
        # Gradient clipping 
        clip_gradients: bool = False,
        max_grad_norm_actor: float = 1.0,
        max_grad_norm_critic: float = 1.0
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.device = device
        self.auto_entropy_tuning = auto_entropy_tuning
        self.flight_phase = flight_phase
        self.save_stats_frequency = save_stats_frequency
        self.use_per = use_per
        self.use_l1_loss = use_l1_loss
        self.clip_gradients = clip_gradients
        self.max_grad_norm_actor = max_grad_norm_actor
        self.max_grad_norm_critic = max_grad_norm_critic
        
        # 1. Initialize networks
        # 1.1. Actor
        self.actor = Actor(
            state_dim, 
            action_dim, 
            hidden_dim_actor, 
            number_of_hidden_layers_actor,
            max_action
        ).to(device)
        # 1.2. Critic
        self.critic = Critic(
            state_dim, 
            action_dim, 
            hidden_dim_critic, 
            number_of_hidden_layers_critic
        ).to(device)
        # 1.3. Critic target
        self.critic_target = Critic(
            state_dim, 
            action_dim, 
            hidden_dim_critic, 
            number_of_hidden_layers_critic
        ).to(device)
        # 1.4. Initialise target critic with critic parameters
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 2. Initialise optimisers and initialise temperature parameter alpha
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
        self.log_alpha = torch.tensor(np.log(alpha_initial), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_learning_rate)
        
        # 3. If auto_entropy_tuning is enabled, set target entropy, but in reality this is always used.
        if auto_entropy_tuning:
            self.target_entropy = -action_dim # Same as in Haarnoja et al.
        
        # 4. Initialise replay buffer
        if use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                buffer_size,
                state_dim,
                action_dim,
                alpha=per_alpha,
                beta=per_beta,
                beta_annealing_steps=per_beta_annealing_steps,
                epsilon=per_epsilon
            )
        else:
            self.replay_buffer = ReplayBuffer(buffer_size, state_dim, action_dim)
        
        # 5. Logging etc.
        self.updates = 0
        
        # 5.1. Directory structure, to avoid failing tests when saving
        os.makedirs(f"data/agent_saves/PyTorchSAC/{flight_phase}/runs", exist_ok=True)
        os.makedirs(f"data/agent_saves/PyTorchSAC/{flight_phase}/saves", exist_ok=True)
        os.makedirs(f"data/agent_saves/PyTorchSAC/{flight_phase}/learning_stats", exist_ok=True)
        
        # 5.2. Logging "learning_stats"
        self.learning_stats = {
            'step': [],
            # State
            'state_mean': [],
            'state_min': [],
            'state_max': [],
            'state_std': [],
            'state_kurtosis': [],
            # Action
            'action_mean': [],
            'action_min': [],
            'action_max': [],
            'action_std': [],
            'action_kurtosis': [],
            # Reward
            'reward_mean': [],
            'reward_min': [],
            'reward_max': [],
            'reward_std': [],
            'reward_kurtosis': [],
            # Losses    
            'critic_loss': [],
            'actor_loss': [],
            'alpha_loss': [],
            'alpha_value': [],
            # Q-value
            'q_value_mean': [],
            'q_value_min': [],
            'q_value_max': [],
            'q_value_std': [],
            # Log probability
            'log_prob_mean': [],
            'log_prob_min': [],
            'log_prob_max': [],
            'log_prob_std': [],
            # Target Q-value
            'target_q_mean': [],
            'target_q_min': [],
            'target_q_max': [],
            'target_q_std': [],
        }
        
        # 5.3. Add PER-specific stats if using PER
        if use_per:
            self.learning_stats['per_beta'] = []
            self.learning_stats['per_mean_weight'] = []
            self.learning_stats['per_max_priority'] = []
        
        # 5.4. Add gradient clipping stats if enabled
        if clip_gradients:
            self.learning_stats['actor_grad_norm'] = []
            self.learning_stats['critic_grad_norm'] = []
    
    @property # So easily called as a "getter" method. i.e. rocket = Rocketclass(); rocket.alpha <- alpha
    def alpha(self):
        # This corrects for the fact that the log_alpha is used.
        return self.log_alpha.exp()
    
    def select_action(self, state, deterministic=False):
        # Select an action from the policy
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state, deterministic)
            return action.cpu().numpy().flatten()
    
    def update(self):
        # This is batched! So check if buffer is full enough.
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 1. Sample from replay buffer
        if self.use_per:
            batch = self.replay_buffer.sample(self.batch_size)
            if batch is None: # If buffer is not full enough, return, extra redundancy left from debugging.
                return
                
            states, actions, rewards, next_states, dones, weights, indices = batch
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)
            weights = None
        
        # 2. Update critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        current_q1, current_q2 = self.critic(states, actions)
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
        '''
        So here there are four configurations:
        - PER, L1 loss
        - PER, L2 loss
        - No PER, L1 loss
        - No PER, L2 loss
        PER requires importance sampling weight correction.
        L2 loss is MSE, and L1 loss can just use the build in functional from PyTorch.
        '''
        if self.use_per:
            if self.use_l1_loss:
                critic_loss = (weights * F.l1_loss(current_q1, target_q, reduction='none') + 
                              weights * F.l1_loss(current_q2, target_q, reduction='none')).mean()
            else:
                critic_loss = (weights * F.mse_loss(current_q1, target_q, reduction='none') + 
                              weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()
        else:
            if self.use_l1_loss:
                critic_loss = F.l1_loss(current_q1, target_q) + F.l1_loss(current_q2, target_q)
            else:
                critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 3. Apply gradient clipping using torch's built in functions.
        critic_grad_norm = None
        if self.clip_gradients:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                max_norm=self.max_grad_norm_critic
            )
            if self.learning_stats.get('critic_grad_norm') is not None: # Redundancy left from debugging.
                self.learning_stats['critic_grad_norm'].append(critic_grad_norm.item())
        
        self.critic_optimizer.step()
        
        # 4. Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)

        # Again PER requires importance sampling weight correction.
        if self.use_per:
            actor_loss = (weights * (self.alpha * log_probs - q)).mean()
        else:
            actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 5. Apply gradient clipping using torch's built in functions.
        actor_grad_norm = None
        if self.clip_gradients:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                max_norm=self.max_grad_norm_actor
            )
            if self.learning_stats.get('actor_grad_norm') is not None: # Redundancy left from debugging.
                self.learning_stats['actor_grad_norm'].append(actor_grad_norm.item())
        
        self.actor_optimizer.step()
        
        # 6. Update temperature parameter alpha
        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_entropy_tuning:
            if self.use_per:
                alpha_loss = -(self.log_alpha * weights * (log_probs + self.target_entropy).detach()).mean()
            else:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
        
        # 7. Update priorities in PER
        if self.use_per:
            # Max of the two TD errors is used here.
            td_errors = torch.max(td_error1.abs(), td_error2.abs()).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # 8. Polyak averaging used to update the target networks.
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * param.data)
        
        # 9. logging.
        self._track_learning_stats(
            states=states,
            actions=actions,
            rewards=rewards,
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            alpha_loss=alpha_loss,
            q_values=q,
            target_q=target_q,
            log_probs=log_probs,
            weights=weights
        )
        
        # 10. Periodically save stats for efficiency.
        self.updates += 1
        run_id = getattr(self, 'run_id', None)
        if self.updates % self.save_stats_frequency == 0:
            self.save_learning_stats(run_id=run_id)
    
    def _track_learning_stats(self, states, actions, rewards, critic_loss, actor_loss, alpha_loss, q_values, target_q, log_probs, weights=None):
        # 1. Convert to numpy.
        states_np = states.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()
        rewards_np = rewards.detach().cpu().numpy().flatten()
        q_values_np = q_values.detach().cpu().numpy().flatten()
        target_q_np = target_q.detach().cpu().numpy().flatten()
        log_probs_np = log_probs.detach().cpu().numpy().flatten()
        
        # 2. Steps.
        self.learning_stats['step'].append(self.updates)
        
        # State statistics (for each dimension)
        state_mean = np.mean(states_np, axis=0)
        state_min = np.min(states_np, axis=0)
        state_max = np.max(states_np, axis=0)
        state_std = np.std(states_np, axis=0)
        # Kurtosis sometimes failed so try and expect is implemented.
        state_kurtosis = np.zeros_like(state_mean)
        for i in range(len(state_mean)):
            try:
                state_kurtosis[i] = scipy.stats.kurtosis(states_np[:, i])
            except:
                state_kurtosis[i] = 0
        
        # 3. State logging.
        self.learning_stats['state_mean'].append(np.mean(state_mean))
        self.learning_stats['state_min'].append(np.mean(state_min))
        self.learning_stats['state_max'].append(np.mean(state_max))
        self.learning_stats['state_std'].append(np.mean(state_std))
        self.learning_stats['state_kurtosis'].append(np.mean(state_kurtosis))
        
        # 4. Action logging.
        self.learning_stats['action_mean'].append(np.mean(actions_np))
        self.learning_stats['action_min'].append(np.min(actions_np))
        self.learning_stats['action_max'].append(np.max(actions_np))
        self.learning_stats['action_std'].append(np.std(actions_np))
        try:
            self.learning_stats['action_kurtosis'].append(scipy.stats.kurtosis(actions_np.flatten()))
        except:
            self.learning_stats['action_kurtosis'].append(0)
        
        # 5. Reward logging.
        self.learning_stats['reward_mean'].append(np.mean(rewards_np))
        self.learning_stats['reward_min'].append(np.min(rewards_np))
        self.learning_stats['reward_max'].append(np.max(rewards_np))
        self.learning_stats['reward_std'].append(np.std(rewards_np))
        try:
            self.learning_stats['reward_kurtosis'].append(scipy.stats.kurtosis(rewards_np))
        except:
            self.learning_stats['reward_kurtosis'].append(0)
        
        # 6. Loss logging.
        self.learning_stats['critic_loss'].append(critic_loss.item())
        self.learning_stats['actor_loss'].append(actor_loss.item())
        self.learning_stats['alpha_loss'].append(alpha_loss.item())
        self.learning_stats['alpha_value'].append(self.alpha.item())
        
        # 7. Q-value logging.
        self.learning_stats['q_value_mean'].append(np.mean(q_values_np))
        self.learning_stats['q_value_min'].append(np.min(q_values_np))
        self.learning_stats['q_value_max'].append(np.max(q_values_np))
        self.learning_stats['q_value_std'].append(np.std(q_values_np))
        
        # 8. Target Q-value logging.
        self.learning_stats['target_q_mean'].append(np.mean(target_q_np))
        self.learning_stats['target_q_min'].append(np.min(target_q_np))
        self.learning_stats['target_q_max'].append(np.max(target_q_np))
        self.learning_stats['target_q_std'].append(np.std(target_q_np))
        
        # 9. Log probability logging.
        self.learning_stats['log_prob_mean'].append(np.mean(log_probs_np))
        self.learning_stats['log_prob_min'].append(np.min(log_probs_np))
        self.learning_stats['log_prob_max'].append(np.max(log_probs_np))
        self.learning_stats['log_prob_std'].append(np.std(log_probs_np))
        
        # 10. PER logging.
        if self.use_per and weights is not None:
            weights_np = weights.detach().cpu().numpy()
            self.learning_stats['per_beta'].append(self.replay_buffer.beta)
            self.learning_stats['per_mean_weight'].append(np.mean(weights_np))
            self.learning_stats['per_max_priority'].append(self.replay_buffer.max_priority)
    
    def save_learning_stats(self, filename=None, run_id=None):
        run_dir = getattr(self, 'run_dir', None)
        if filename is None:
            if run_dir:
                filename = f"{run_dir}/learning_stats/sac_learning_stats.csv"
            elif run_id is None: # old code
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/learning_stats/sac_learning_stats_{timestamp}.csv"
            else:
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/learning_stats/sac_learning_stats_{run_id}.csv"
        
        if len(self.learning_stats['step']) == 0:
            print(f"No data")
            return
        if not hasattr(self, 'last_saved_step') or self.last_saved_step is None:
            self.last_saved_step = 0
        
        if self.last_saved_step == 0:
            start_idx = 0
        else:
            steps = self.learning_stats['step']
            start_idx = 0
            for i, step in enumerate(steps):
                if step > self.last_saved_step:
                    start_idx = i
                    break
        
        # 1. Extract all statistics since the last save.
        if start_idx < len(self.learning_stats['step']):
            latest_stats = {key: [self.learning_stats[key][i] for i in range(start_idx, len(self.learning_stats['step']))] 
                           for key in self.learning_stats}
            
            # 2. Update the last saved step.
            if len(self.learning_stats['step']) > 0:
                self.last_saved_step = self.learning_stats['step'][-1]
            
            stats_df = pd.DataFrame(latest_stats)
            
            # 3. Check it exists with headers
            file_exists = os.path.isfile(filename)
            
            # 4. Save
            if file_exists:
                stats_df.to_csv(filename, mode='a', header=False, index=False)
            else:
                # Create new file with headers
                stats_df.to_csv(filename, index=False)            
        else:
            print(f"Debug: No new stats to save.")
    
    def save(self, filename=None, run_id=None):
        run_dir = getattr(self, 'run_dir', None)
        
        if filename is None:
            if run_dir:
                filename = f"{run_dir}/agent_saves/sac_pytorch"
            elif run_id is None: # legacy code
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/saves/sac_pytorch_{timestamp}"
            else:
                filename = f"data/agent_saves/PyTorchSAC/{self.flight_phase}/saves/sac_pytorch_{run_id}"
        # 1. Save logs
        self.save_learning_stats(f"{filename}_learning_stats.csv", run_id)
        # 2. Save model
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'updates': self.updates
        }, filename)
        print(f"Model saved to {filename}")
    
    def load(self, filename):
        checkpoint = torch.load(filename)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
        self.updates = checkpoint['updates']
        
        print(f"Model loaded from {filename}") 