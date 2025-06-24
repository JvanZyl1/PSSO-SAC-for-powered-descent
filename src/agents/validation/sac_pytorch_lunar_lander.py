import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import os
from datetime import datetime

from src.agents.sac_pytorch import SACPyTorch

def main():
    env = gym.make("LunarLander-v3", continuous=True, render_mode=None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    # Create results directory if it doesn't exist
    os.makedirs("results/PyTorchSAC/LunarLander", exist_ok=True)
    
    agent = SACPyTorch(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim_actor=256,
        number_of_hidden_layers_actor=2,
        hidden_dim_critic=256,
        number_of_hidden_layers_critic=2,
        alpha_initial=0.1,
        gamma=0.99,
        tau=0.005,
        buffer_size=100000,
        batch_size=256,
        critic_learning_rate=3e-4,
        actor_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        max_action=max_action,
        flight_phase="LunarLander",
        auto_entropy_tuning=True
    )
    
    num_episodes = 1000
    max_steps_per_episode = 2500
    evaluation_frequency = 10
    num_eval_episodes = 5
    
    episode_rewards = []
    eval_rewards = []
    episode_steps = []
    episode_updates = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        updates_this_episode = 0
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=1.0 if done else 0.0
            )
            state = next_state
            episode_reward += reward
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
                updates_this_episode += 1
            if done or truncated:
                break
        
        episode_rewards.append(float(episode_reward))
        episode_steps.append(step + 1)
        episode_updates.append(updates_this_episode)
        print(f"Episode {episode}: Reward = {float(episode_reward):.2f}, Steps = {step+1}, Updates = {updates_this_episode}")
        if (episode + 1) % evaluation_frequency == 0:
            eval_reward = evaluate_agent(agent, env, num_eval_episodes)
            eval_rewards.append(eval_reward)
            print(f"Evaluation after episode {episode}: Average reward = {eval_reward:.2f}")
            if (episode + 1) % (evaluation_frequency * 5) == 0:
                agent.save()
    
    eval_reward = evaluate_agent(agent, env, num_eval_episodes)
    print(f"Final evaluation: Average reward = {eval_reward:.2f}")
    agent.save()
    plot_results(episode_rewards, eval_rewards, episode_steps, episode_updates, evaluation_frequency)
    env.close()

def evaluate_agent(agent, env, num_episodes):
    eval_rewards = []
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state
            episode_reward += reward
        eval_rewards.append(episode_reward)
    return np.mean(eval_rewards)

def plot_results(episode_rewards, eval_rewards, episode_steps, episode_updates, eval_freq)
    plt.figure(figsize=(15, 10))
    plt.suptitle('Lunar Lander Training and Evaluation Results', fontsize=16)
    plt.subplot(2, 2, 1)
    plt.plot(episode_rewards, color='blue', linewidth=2)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.title('Training Rewards', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.subplot(2, 2, 2)
    eval_episodes = [(i+1) * eval_freq for i in range(len(eval_rewards))]
    plt.plot(eval_episodes, eval_rewards, marker='o', color='red', linewidth=2)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Average Reward', fontsize=16)
    plt.title('Evaluation Rewards', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.subplot(2, 2, 3)
    plt.plot(episode_steps, color='blue', linewidth=2)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Steps', fontsize=16)
    plt.title('Episode Length', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.subplot(2, 2, 4)
    plt.plot(episode_updates, color='blue', linewidth=2)
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Updates', fontsize=16)
    plt.title('Updates per Episode', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"results/PyTorchSAC/LunarLander/training_results_{timestamp}.png")
    plt.show()

if __name__ == "__main__":
    main() 