import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import json

from src.agents.sac_pytorch import SACPyTorch
from src.envs.rl.env_wrapped_rl_pytorch import rl_wrapped_env_pytorch
from src.envs.rl.env_wrapped_rl_pytorch import maximum_velocity as maximum_velocity_lambda

class pytorch_sac_inferred(SACPyTorch):
    def select_actions_no_stochastic(self, state):
        return self.select_action(state, deterministic=True)
    
    def select_actions(self, state):
        return self.select_action(state)

def main(use_l1_loss, gamma, per_alpha, per_beta):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    env = rl_wrapped_env_pytorch(
        flight_phase="landing_burn_pure_throttle",
        enable_wind=False,
        stochastic_wind=False,
        trajectory_length=1,
        discount_factor=0.99
    )
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    run_dir = f"data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/{run_id}"
    os.makedirs(f"{run_dir}/agent_saves", exist_ok=True)
    os.makedirs(f"{run_dir}/learning_stats", exist_ok=True)
    os.makedirs(f"{run_dir}/trajectories", exist_ok=True)
    os.makedirs(f"{run_dir}/metrics", exist_ok=True)
    os.makedirs(f"{run_dir}/plots", exist_ok=True)
    os.makedirs("results/PyTorchSAC/LandingBurnPureThrottle", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/saves", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/runs", exist_ok=True)
    os.makedirs("data/agent_saves/PyTorchSAC/LandingBurnPureThrottle/learning_stats", exist_ok=True)
    num_episodes = 750
    max_steps_per_episode = 2200
    evaluation_frequency = 10
    num_eval_episodes = 1 # As not stochastic, we can just do 1
    save_stats_frequency = 50
    
    early_log_check = True  
    early_log_check_frequency = 10  
    
    # Hyperparameters for SAC
    sac_hyperparams = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "hidden_dim_actor": 256,
        "number_of_hidden_layers_actor": 2,
        "hidden_dim_critic": 256,
        "number_of_hidden_layers_critic": 2,
        "alpha_initial": 0.2,
        "gamma": gamma,
        "tau": 0.005,
        "buffer_size": 1000000,
        "batch_size": 512,
        "critic_learning_rate": 0.005,
        "actor_learning_rate": 1e-5,
        "alpha_learning_rate": 3e-4,
        "max_action": 1.0,
        "auto_entropy_tuning": True,
        "use_per": True,
        "per_alpha": per_alpha,
        "per_beta": per_beta,
        "per_beta_annealing_steps": 100000,
        "per_epsilon": 1e-6,
        "use_l1_loss": use_l1_loss,
        "clip_gradients": False,
        "max_grad_norm_actor": 0.1,
        "max_grad_norm_critic": 0.1
    }
    
    agent = pytorch_sac_inferred(
        state_dim=sac_hyperparams["state_dim"],
        action_dim=sac_hyperparams["action_dim"],
        hidden_dim_actor=sac_hyperparams["hidden_dim_actor"],
        number_of_hidden_layers_actor=sac_hyperparams["number_of_hidden_layers_actor"],
        hidden_dim_critic=sac_hyperparams["hidden_dim_critic"],
        number_of_hidden_layers_critic=sac_hyperparams["number_of_hidden_layers_critic"],
        alpha_initial=sac_hyperparams["alpha_initial"],
        gamma=sac_hyperparams["gamma"],
        tau=sac_hyperparams["tau"],
        buffer_size=sac_hyperparams["buffer_size"],
        batch_size=sac_hyperparams["batch_size"],
        critic_learning_rate=sac_hyperparams["critic_learning_rate"],
        actor_learning_rate=sac_hyperparams["actor_learning_rate"],
        alpha_learning_rate=sac_hyperparams["alpha_learning_rate"],
        max_action=sac_hyperparams["max_action"],
        flight_phase="LandingBurnPureThrottle",
        auto_entropy_tuning=sac_hyperparams["auto_entropy_tuning"],
        use_l1_loss=sac_hyperparams["use_l1_loss"],
        use_per=sac_hyperparams["use_per"],
        per_alpha=sac_hyperparams["per_alpha"],
        per_beta=sac_hyperparams["per_beta"],
        per_beta_annealing_steps=sac_hyperparams["per_beta_annealing_steps"],
        per_epsilon=sac_hyperparams["per_epsilon"],
        save_stats_frequency=save_stats_frequency,
        clip_gradients=sac_hyperparams["clip_gradients"],
        max_grad_norm_actor=sac_hyperparams["max_grad_norm_actor"],
        max_grad_norm_critic=sac_hyperparams["max_grad_norm_critic"]
    )
    
    agent.run_id = run_id
    
    training_config = {
        "num_episodes": num_episodes,
        "max_steps_per_episode": max_steps_per_episode,
        "evaluation_frequency": evaluation_frequency,
        "num_eval_episodes": num_eval_episodes,
        "save_stats_frequency": save_stats_frequency,
        "environment": {
            "flight_phase": "landing_burn_pure_throttle",
            "enable_wind": False,
            "stochastic_wind": False,
            "trajectory_length": 1,
            "discount_factor": 0.99
        },
        "run_id": run_id
    }
    
    all_params = {
        "sac_hyperparameters": sac_hyperparams,
        "training_config": training_config
    }
    
    hyperparams_path = f"{run_dir}/hyperparameters.json"
    with open(hyperparams_path, 'w') as f:
        json.dump(all_params, f, indent=4)
    
    training_metrics_path = f"{run_dir}/metrics/training_metrics.csv"
    eval_metrics_path = f"{run_dir}/metrics/eval_metrics.csv"
    
    pd.DataFrame(columns=['episode', 'reward', 'steps', 'updates']).to_csv(training_metrics_path, index=False)
    pd.DataFrame(columns=['episode', 'eval_reward', 'success_rate']).to_csv(eval_metrics_path, index=False)
    
    learning_stats_path = f"{run_dir}/learning_stats/sac_learning_stats.csv"
    empty_stats = {key: [] for key in agent.learning_stats.keys()}
    pd.DataFrame(empty_stats).to_csv(learning_stats_path, index=False)
    print(f"Created learning stats file: {learning_stats_path}")
    
    if early_log_check:
        agent.save_stats_frequency = early_log_check_frequency
        print(f"Early log check enabled: Saving learning stats every {early_log_check_frequency} updates")
    
    agent.run_dir = run_dir
    
    episode_rewards = []
    eval_rewards = []
    episode_steps = []
    episode_updates = []
    landing_success_rate = []
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        updates_this_episode = 0
        
        for step in range(max_steps_per_episode):
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            agent.replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=1.0 if done else 0.0
            )
            state = next_state
            episode_reward += episode_reward
            if len(agent.replay_buffer) > agent.batch_size:
                agent.update()
                updates_this_episode += 1
            if done or truncated:
                break
        
        episode_rewards.append(float(episode_reward))
        episode_steps.append(step + 1)
        episode_updates.append(updates_this_episode)
        episode_data = pd.DataFrame({
            'episode': [episode + 1],
            'reward': [float(episode_reward)],
            'steps': [step + 1],
            'updates': [updates_this_episode]
        })
        episode_data.to_csv(training_metrics_path, mode='a', header=False, index=False)
        
        # Early logging check
        if early_log_check and episode == 2:
            print("Performing early logging check...")
            agent.save_learning_stats(run_id=run_id)
            agent.save_stats_frequency = save_stats_frequency
            print(f"Early logging check complete. Restored normal save frequency to {save_stats_frequency}")
            early_log_check = False
        
        if (episode + 1) % evaluation_frequency == 0:
            eval_reward, success_rate = evaluate_agent(agent, env, num_eval_episodes)
            eval_rewards.append(eval_reward)
            landing_success_rate.append(success_rate)
            print(f"Evaluation after episode {episode}: Average reward = {eval_reward:.2f}, Success rate = {success_rate:.2f}")
            eval_data = pd.DataFrame({
                'episode': [episode + 1],
                'eval_reward': [eval_reward],
                'success_rate': [success_rate]
            })
            eval_data.to_csv(eval_metrics_path, mode='a', header=False, index=False)
            
            if (episode + 1) % (evaluation_frequency * 5) == 0:
                agent.save(run_id=run_id)
                visualize_trajectory(agent, env, run_id=run_id, run_dir=run_dir)
                plot_results(training_metrics_path, eval_metrics_path, run_id, run_dir=run_dir)
    
    eval_reward, success_rate = evaluate_agent(agent, env, num_eval_episodes)
    print(f"Final evaluation: Average reward = {eval_reward:.2f}, Success rate = {success_rate:.2f}")
    final_eval_data = pd.DataFrame({
        'episode': [num_episodes],
        'eval_reward': [eval_reward],
        'success_rate': [success_rate]
    })
    final_eval_data.to_csv(eval_metrics_path, mode='a', header=False, index=False)
    agent.save(run_id=run_id)
    plot_results(training_metrics_path, eval_metrics_path, run_id, run_dir=run_dir)
    visualize_trajectory(agent, env, run_id=run_id, run_dir=run_dir, final=True)
    env.close()

def evaluate_agent(agent, env, num_episodes):
    eval_rewards = []
    successes = 0
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, info = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                successes += 1
        eval_rewards.append(episode_reward)
    success_rate = successes / num_episodes
    return np.mean(eval_rewards), success_rate

def save_trajectory_to_csv(states, actions, rewards, infos, dynamic_pressure, throttle_command, maximum_velocity, run_id, run_dir=None, final=False):
    if final:
        csv_path = f"{run_dir}/trajectories/trajectory_final.csv"
    else:
        csv_path = f"{run_dir}/trajectories/trajectory.csv"
    
    num_states = len(states)
    num_steps = len(rewards)
    
    # x, y, vx, vy, theta, theta_dot, gamma, alpha, mass, mass_propellant, time
    data = {
        'time': states[:, 10],
        'x': states[:, 0],
        'y': states[:, 1],
        'vx': states[:, 2],
        'vy': states[:, 3],
        'theta': states[:, 4],
        'theta_dot': states[:, 5],
        'gamma': states[:, 6],
        'alpha': states[:, 7],
        'mass': states[:, 8],
        'mass_propellant': states[:, 9],
    }
    
    if num_steps > 0:
        data['throttle'] = np.zeros(num_states)
        data['throttle'][:num_steps] = throttle_command
        data['throttle'][num_steps:] = throttle_command[-1] if num_steps > 0 else 0
        
        data['action'] = np.zeros(num_states)
        flattened_actions = np.array([a.flatten()[0] for a in actions])
        data['action'][:num_steps] = flattened_actions
        data['action'][num_steps:] = flattened_actions[-1] if num_steps > 0 else 0
        
        data['reward'] = np.zeros(num_states)
        data['reward'][:num_steps] = rewards
        
    
    data['dynamic_pressure'] = np.zeros(num_states)
    data['dynamic_pressure'][:len(dynamic_pressure)] = dynamic_pressure
    if len(dynamic_pressure) < num_states and len(dynamic_pressure) > 0:
        data['dynamic_pressure'][len(dynamic_pressure):] = dynamic_pressure[-1]
        
    data['maximum_velocity'] = np.zeros(num_states)
    data['maximum_velocity'][:len(maximum_velocity)] = maximum_velocity
    if len(maximum_velocity) < num_states and len(maximum_velocity) > 0:
        data['maximum_velocity'][len(maximum_velocity):] = maximum_velocity[-1]
    
    if len(infos) > 0:
        all_scalar_keys = set()
        all_nested_keys = {}
        for info in infos:
            for key, value in info.items():
                if key in ['state', 'actions']:
                    continue
                
                if isinstance(value, dict):
                    if key not in all_nested_keys:
                        all_nested_keys[key] = set()
                    
                    for subkey, subvalue in value.items():
                        if subkey == 'throttle':
                            continue
                        
                        if np.isscalar(subvalue):
                            all_nested_keys[key].add(subkey)
                elif np.isscalar(value):
                    all_scalar_keys.add(key)
        for key in all_scalar_keys:
            if key not in data:
                data[key] = np.full(num_states, np.nan)
        
        for dict_key, subkeys in all_nested_keys.items():
            for subkey in subkeys:
                col_name = f"{dict_key}_{subkey}"
                if col_name not in data:
                    data[col_name] = np.full(num_states, np.nan)
        
        for info_idx, info in enumerate(infos):
            for key in all_scalar_keys:
                if key in info and np.isscalar(info[key]):
                    data[key][info_idx] = info[key]
            for dict_key, subkeys in all_nested_keys.items():
                if dict_key in info and isinstance(info[dict_key], dict):
                    for subkey in subkeys:
                        col_name = f"{dict_key}_{subkey}"
                        if subkey in info[dict_key] and np.isscalar(info[dict_key][subkey]):
                            data[col_name][info_idx] = info[dict_key][subkey]
    
    lengths = [len(arr) for arr in data.values()]
    if len(set(lengths)) > 1:
        print(f"Warning: Arrays have different lengths: {dict(zip(data.keys(), lengths))}")
        min_length = min(lengths)
        data = {k: v[:min_length] for k, v in data.items()}
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Trajectory data saved to {csv_path}")
    
    return csv_path

def visualize_trajectory(agent, env, run_id, run_dir=None, final=False):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    dynamic_pressure = []
    throttle_command = []
    maximum_velocity = []
    infos = []
    done = False
    truncated = False
    while not (done or truncated):
        action = agent.select_action(state, deterministic=True)
        next_state, reward, done, truncated, info = env.step(action)
        raw_state = info['state']
        states.append(raw_state)
        actions.append(action)
        rewards.append(reward)
        dynamic_pressure.append(info['dynamic_pressure'])
        throttle_command.append(info['action_info']['throttle'])
        maximum_velocity_state = maximum_velocity_lambda(raw_state[1], raw_state[3])
        maximum_velocity.append(maximum_velocity_state)
        infos.append(info)
        state = next_state
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dynamic_pressure = np.array(dynamic_pressure)
    throttle_command = np.array(throttle_command)
    maximum_velocity = np.array(maximum_velocity)
    csv_path = save_trajectory_to_csv(
        states=states,
        actions=actions,
        rewards=rewards,
        infos=infos,
        dynamic_pressure=dynamic_pressure,
        throttle_command=throttle_command,
        maximum_velocity=maximum_velocity,
        run_id=run_id,
        run_dir=run_dir,
        final=final
    )
    
    # Plot
    plt.figure(figsize=(15, 10))
    plt.suptitle('Landing Burn Trajectory', fontsize=16)
    plt.subplot(3, 2, 1)
    plt.plot(states[:, 10], states[:, 1], color='blue', linewidth=2)  # time vs y
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Altitude (m)', fontsize=16)
    plt.title('Altitude vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.subplot(3, 2, 2)
    plt.plot(states[:, 10], states[:, 0], color='blue', linewidth=2)  # time vs x
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Horizontal Position (m)', fontsize=16)
    plt.title('Horizontal Position vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.subplot(3, 2, 3)
    plt.plot(states[:, 10], states[:, 3], color='blue', linewidth=2)  # time vs vy
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Vertical Velocity (m/s)', fontsize=16)
    plt.title('Vertical Velocity vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.subplot(3, 2, 4)
    plt.plot(states[:, 10], states[:, 2], color='blue', linewidth=2)  # time vs vx
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Horizontal Velocity (m/s)', fontsize=16)
    plt.title('Horizontal Velocity vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.subplot(3, 2, 5)
    plt.plot(states[:-1, 10], throttle_command[:len(states)-1], color='blue', linewidth=2)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Throttle Command', fontsize=16)
    plt.title('Throttle Command vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.subplot(3, 2, 6)
    plt.plot(states[:-1, 10], rewards[:len(states)-1], color='blue', linewidth=2)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Reward', fontsize=16)
    plt.title('Reward vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    plt.tight_layout()
    if final:
        plt.savefig(f"{run_dir}/plots/trajectory_final.png")
    else:
        plt.savefig(f"{run_dir}/plots/trajectory.png")
    plt.close()
    plt.figure(figsize=(15, 10))
    plt.plot(states[:, 10], dynamic_pressure/1000, color='blue', linewidth=2)
    plt.xlabel('Time (s)', fontsize=16)
    plt.ylabel('Dynamic Pressure (kPa)', fontsize=16)
    plt.title('Dynamic Pressure vs Time', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    if final:
        plt.savefig(f"{run_dir}/plots/dynamic_pressure_final.png")
    else:
        plt.savefig(f"{run_dir}/plots/dynamic_pressure.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    plt.plot(states[:, 0], states[:, 1], color='blue', linewidth=2)  # x vs y
    plt.scatter(states[0, 0], states[0, 1], color='green', s=100, marker='o', label='Start')
    plt.scatter(states[-1, 0], states[-1, 1], color='red', s=100, marker='x', label='End')
    plt.xlabel('Horizontal Position (m)', fontsize=16)
    plt.ylabel('Altitude (m)', fontsize=16)
    plt.title('X-Y Trajectory', fontsize=16)
    plt.legend(fontsize=14)
    plt.tick_params(axis='both', labelsize=16)
    plt.grid(True)
    if final:
        plt.savefig(f"{run_dir}/plots/xy_trajectory_final.png")
    else:
        plt.savefig(f"{run_dir}/plots/xy_trajectory.png")
    
    plt.close()
    total_reward = sum(rewards)
    return total_reward, states[:, 1]

def plot_results(training_metrics_path, eval_metrics_path, run_id, run_dir=None):
    training_df = pd.read_csv(training_metrics_path)
    eval_df = pd.read_csv(eval_metrics_path)
    
    # Plot
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 2, 1)
    plt.plot(training_df['episode'], training_df['reward'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.subplot(2, 2, 2)
    plt.plot(eval_df['episode'], eval_df['eval_reward'], marker='o', linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Average Reward', fontsize=16)
    plt.title('Evaluation Rewards', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)
    plt.subplot(2, 2, 3)
    plt.plot(training_df['episode'], training_df['steps'], linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Steps', fontsize=16)
    plt.title('Episode Length', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)
    plt.subplot(2, 2, 4)
    plt.plot(eval_df['episode'], eval_df['success_rate'], marker='o', linewidth=2, color='blue')
    plt.xlabel('Episode', fontsize=16)
    plt.ylabel('Success Rate', fontsize=16)
    plt.title('Landing Success Rate', fontsize=16)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/plots/training_results.png")
    plt.close()

if __name__ == "__main__":
    # Tests
    # use_l1_loss, gamma, per_alpha, per_beta
    # Want to try: 
    # alpha
    # False, 0.99, 0.7, 0.4
    # False, 0.99, 0.5, 0.4
    # L1
    # True, 0.99, 0.6, 0.4
    # gamma
    # False, 0.95, 0.6, 0.4
    # False, 0.9, 0.6, 0.4
    # beta
    # False, 0.99, 0.6, 0.6
    # False, 0.99, 0.6, 0.2
    tests = [
        (False, 0.99, 0.7, 0.4),
        (False, 0.99, 0.5, 0.4),
        (False, 0.95, 0.6, 0.4),
        (False, 0.9, 0.6, 0.4),
        (True, 0.99, 0.6, 0.4),
        (False, 0.99, 0.6, 0.6),
        (False, 0.99, 0.6, 0.2),
    ]
    # Final tests tomorrow : gradient clipping, 
    for test in tests:
        main(*test)