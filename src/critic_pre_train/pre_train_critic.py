import lmdb
import pickle
import jax
import jax.numpy as jnp
import random
import optax
from flax import linen as nn
from functools import partial
from typing import Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from src.agents.functions.networks import DoubleCritic


def load_experiences_lmdb(flight_phase,
                          batch_size = 64):
    folder_path=f"data/experience_buffer/{flight_phase}/experience_buffer.lmdb"
    env = lmdb.open(folder_path, readonly=True, lock=False)  # Open the folder, not the .mdb file

    with env.begin() as txn:
        cursor = txn.cursor()
        experiences = []

        for _, value in cursor:
            experience = pickle.loads(value)
            experiences.append(experience)

        random.shuffle(experiences)
        for i in range(0, len(experiences), batch_size):
            yield experiences[i:i + batch_size]

def preprocess_batch(batch):
    states, actions, rewards, next_states, next_actions = zip(*batch)
    actions = [action.detach().numpy() if hasattr(action, 'detach') else action for action in actions]
    next_actions = [next_action.detach().numpy() if hasattr(next_action, 'detach') else next_action for next_action in next_actions]
    
    return (
        jnp.array(states), 
        jnp.array(actions), 
        jnp.array(rewards).reshape(-1, 1), 
        jnp.array(next_states), 
        jnp.array(next_actions)
    )


def calculate_td_error(states : jnp.ndarray,
                       actions : jnp.ndarray,
                       next_states : jnp.ndarray,
                       next_actions : jnp.ndarray,
                       rewards : jnp.ndarray,
                       critic : nn.Module,
                       critic_params : jnp.ndarray,
                       critic_target_params : jnp.ndarray,
                       gamma : float):
    q1, q2 = critic.apply(critic_params, states, actions)
    next_q1, next_q2 = critic.apply(critic_target_params, next_states, next_actions)
    next_q_mean = jnp.minimum(next_q1, next_q2)
    td_target = rewards + gamma * next_q_mean
    td_errors = 0.5 * ((td_target - q1) ** 2 + (td_target - q2) ** 2)
    return jnp.mean(td_errors)



def critic_update(critic_optimiser,
                  calculate_td_lambda_func : Callable,
                  states : jnp.ndarray,
                  actions : jnp.ndarray,
                  next_states : jnp.ndarray,
                  next_actions : jnp.ndarray,
                  rewards : jnp.ndarray,
                  critic_params : jnp.ndarray,
                  critic_target_params : jnp.ndarray,
                  critic_opt_state : jnp.ndarray):
    def loss_fcn(params):
        return calculate_td_lambda_func(states = jax.lax.stop_gradient(states),
                                  actions = jax.lax.stop_gradient(actions),
                                  next_states = jax.lax.stop_gradient(next_states),
                                  next_actions = jax.lax.stop_gradient(next_actions),
                                  rewards = jax.lax.stop_gradient(rewards),
                                  critic_params = params,
                                  critic_target_params = jax.lax.stop_gradient(critic_target_params))
    
    grads = jax.grad(loss_fcn)(critic_params)
    updates, critic_opt_state = critic_optimiser.update(grads, critic_opt_state, critic_params)
    critic_params = optax.apply_updates(critic_params, updates)
    critic_loss = loss_fcn(critic_params)
    return critic_params, critic_opt_state, critic_loss

class pre_train_critic_from_pso_experiences:
    def __init__(self,
                 flight_phase : str,
                 state_dim : int,
                 action_dim : int,
                 hidden_dim_critic : int,
                 number_of_hidden_layers_critic : int,
                 gamma : float,
                 tau : float,
                 critic_learning_rate : float,
                 batch_size : int,
                 reinforcement_type : str):
        assert reinforcement_type in ['sac', 'td3']
        self.flight_phase = flight_phase
        critic = DoubleCritic(state_dim=state_dim,
                              action_dim=action_dim,
                              hidden_dim=hidden_dim_critic,
                              number_of_hidden_layers=number_of_hidden_layers_critic)
        critic_optimiser = optax.adam(learning_rate = critic_learning_rate)
        self.critic_params = critic.init(jax.random.PRNGKey(0), jnp.zeros((1, state_dim)), jnp.zeros((1, action_dim)))
        self.critic_target_params = self.critic_params
        self.critic_opt_state = critic_optimiser.init(self.critic_params)

        self.tau = tau
        self.batch_size = batch_size
        self.writer = SummaryWriter(log_dir = f'data/agent_saves/PreTrainCritic/{reinforcement_type}/{self.flight_phase}/runs')

        calculate_td_lambda_func = jax.jit(
            partial(calculate_td_error,
                    critic = critic,
                    gamma = gamma),
            static_argnames = ('critic', 'gamma')
        )

        self.critic_update_lambda_func = jax.jit(
            partial(critic_update,
                    critic_optimiser = critic_optimiser,
                    calculate_td_lambda_func = calculate_td_lambda_func),
            static_argnames = ('critic_optimiser', 'calculate_td_lambda_func')
        )

        self.critic_losses = []

    def __del__(self):
        self.writer.close()

    def __call__(self):
        pbar = tqdm(load_experiences_lmdb(self.flight_phase,
                                          batch_size=self.batch_size), desc='Pre-training critic')
        iteration = 0
        for batch in pbar:
            critic_loss = self.batch_to_update(batch)
            pbar.set_description(f'Pre-training critic, Critic Loss: {critic_loss:.4e}')
            self.writer.add_scalar('critic_loss', jax.device_get(critic_loss), iteration)
            iteration += 1
        self.save_critic(self.critic_params, self.critic_target_params, self.critic_opt_state)
        self.writer.flush()
        self.plot_critic_loss()

        return self.critic_params, self.critic_target_params, self.critic_opt_state

    def preprocess_batch(self, batch):
        return preprocess_batch(batch)
    
    def plot_critic_loss(self):
        plt.figure()
        plt.plot(self.critic_losses)
        plt.xlabel('Iteration')
        plt.ylabel('Critic Loss')
        plt.title('Critic Loss during Pre-training')
        plt.savefig(f'results/critic_pre_trains/{self.flight_phase}_critic_losses.png')
        plt.close()
    
    def save_critic(self,
                    critic_params,
                    critic_target_params,
                    critic_opt_state):
        self.critic_params = critic_params
        self.critic_target_params = critic_target_params
        self.critic_opt_state = critic_opt_state

        with open(f'data/agent_saves/critic_pre_trains/{self.flight_phase}/saves/critic_params.pkl', 'wb') as f:
            pickle.dump(self.critic_params, f)
        with open(f'data/agent_saves/critic_pre_trains/{self.flight_phase}/saves/critic_target_params.pkl', 'wb') as f:
            pickle.dump(self.critic_target_params, f)
        with open(f'data/agent_saves/critic_pre_trains/{self.flight_phase}/saves/critic_opt_state.pkl', 'wb') as f:
            pickle.dump(self.critic_opt_state, f)

    def batch_to_update(self, batch):
        states, actions, rewards, next_states, next_actions = self.preprocess_batch(batch)
        self.critic_params, self.critic_opt_state, critic_loss = self.critic_update_lambda_func(states = states,
                                                                                                actions = actions,
                                                                                                next_states = next_states,
                                                                                                next_actions = next_actions,
                                                                                                rewards = rewards,
                                                                                                critic_params = self.critic_params,
                                                                                                critic_target_params = self.critic_target_params,
                                                                                                critic_opt_state = self.critic_opt_state)

        self.critic_target_params = jax.tree_util.tree_map(lambda p, tp: self.tau * p + (1.0 - self.tau) * tp, self.critic_params, self.critic_target_params)
        self.critic_losses.append(float(critic_loss))
        return critic_loss
    

if __name__ == '__main__':
    runner = pre_train_critic_from_pso_experiences()
    critic_params, critic_target_params, critic_opt_state = runner()