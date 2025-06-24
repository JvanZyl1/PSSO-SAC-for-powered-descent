import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime

from src.agents.soft_actor_critic import SoftActorCritic

def main():
    env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    os.makedirs("results/VanillaSAC/LunarLander", exist_ok=True)
    os.makedirs("data/agent_saves/VanillaSAC/LunarLander/runs", exist_ok=True)
    os.makedirs("data/agent_saves/VanillaSAC/LunarLander/saves", exist_ok=True)
    
    agent = SoftActorCritic(
        state_dim=state_dim,
        action_dim=action_dim,
        flight_phase="LunarLander",
        hidden_dim_actor=256,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=256,
        number_of_hidden_layers_critic=2,
        temperature_initial=0.1,
        gamma=0.99,
        tau=0.005,
        alpha_buffer=0.6,
        beta_buffer=0.4,
        beta_decay_buffer=0.99,
        buffer_size=100000,
        trajectory_length=1,
        batch_size=256,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        temperature_learning_rate=3e-4,
        critic_grad_max_norm=10.0,
        actor_grad_max_norm=10.0,
        temperature_grad_max_norm=10.0,
        max_std=1.0,
        l2_reg_coef=0.0,
        expected_updates_to_convergence=100000
    )
    
    agent.use_uniform_sampling()
    print(f"Using uniform sampling: {agent.get_sampling_mode()}")
    
    num_episodes = 1000
    max_steps_per_episode = 1000
    evaluation_frequency = 10
    num_eval_episodes = 5
    
    episode_rewards = []
    eval_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        episode_updates = 0
        state = jnp.array(state, dtype=jnp.float32)
        for step in range(max_steps_per_episode):
            action = agent.select_actions(state)
            action_np = np.array(action)
            next_state, reward, done, truncated, _ = env.step(action_np)
            next_state = jnp.array(next_state, dtype=jnp.float32)
            reward = jnp.array(reward, dtype=jnp.float32)
            done_flag = jnp.array(1.0 if done else 0.0, dtype=jnp.float32)
            td_error = agent.calculate_td_error(
                state=jnp.expand_dims(state, 0),
                action=jnp.expand_dims(action, 0),
                reward=jnp.expand_dims(reward, 0),
                next_state=jnp.expand_dims(next_state, 0),
                done=jnp.expand_dims(done_flag, 0)
            )
            agent.buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done_flag,
                td_error=jnp.squeeze(td_error)
            )
            state = next_state
            episode_reward += reward
            if agent.buffer.current_size > agent.batch_size:
                agent.update()
                episode_updates += 1
            if done or truncated:
                break
        
        episode_rewards.append(float(episode_reward))
        print(f"Episode {episode}: Reward = {float(episode_reward):.2f}, Steps = {step+1}, Updates = {episode_updates}")
        if (episode + 1) % evaluation_frequency == 0:
            eval_reward = evaluate_agent(agent, env, num_eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"Evaluation after episode {episode}: Average reward = {eval_reward:.2f}")
            if (episode + 1) % (evaluation_frequency * 5) == 0:
                agent.save()
                plot_results(episode_rewards, eval_rewards, evaluation_frequency)
    
    eval_reward = evaluate_agent(agent, env, num_eval_episodes)
    print(f"Final evaluation: Average reward = {eval_reward:.2f}")
    agent.save()
    env.close()

def evaluate_agent(agent, env, num_episodes):
    eval_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        state = jnp.array(state, dtype=jnp.float32)
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_actions_no_stochastic(state)
            action_np = np.array(action)
            next_state, reward, done, truncated, _ = env.step(action_np)
            state = jnp.array(next_state, dtype=jnp.float32)
            episode_reward += reward
        eval_rewards.append(episode_reward)
    return np.mean(eval_rewards)

def plot_results(episode_rewards, eval_rewards, eval_freq):
    plt.figure(figsize=(12, 6))
    plt.suptitle('Lunar Lander Training and Evaluation Results', fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, color='blue', linewidth=2)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.title('Training Rewards', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.subplot(1, 2, 2)
    eval_episodes = [i * eval_freq for i in range(len(eval_rewards))]
    plt.plot(eval_episodes, eval_rewards, marker='o', color='red', linewidth=2)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Average Reward', fontsize=16)
    plt.title('Evaluation Rewards', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/VanillaSAC/LunarLander/training_results_{timestamp}.png")
    plt.show()

if __name__ == "__main__":
    main() 