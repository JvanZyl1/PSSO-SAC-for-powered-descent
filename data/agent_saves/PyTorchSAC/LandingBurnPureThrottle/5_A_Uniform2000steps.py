import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window_size=20):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

base_path = "data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
run_dir = "5_A_Uniform_2000"

window_size = 20
heavy_window_size = 50
episode_limit = 2000

match_reward_y_axis = True

TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 10

key_episodes = [390, 730, 1070, 1420, 1610]
line_colors = ['red', 'green', 'orange', 'purple', 'cyan']

eval_max_reward = None
eval_min_reward = None
eval_path = os.path.join(base_path, run_dir, "metrics/eval_metrics.csv")
eval_metrics = pd.read_csv(eval_path)
eval_metrics = eval_metrics[eval_metrics['episode'] <= episode_limit]

eval_rewards = eval_metrics['eval_reward'].values
eval_max_reward = max(eval_rewards)
eval_min_reward = min(eval_rewards)
print(f"Evaluation reward range: {eval_min_reward} to {eval_max_reward}")

fig, axes = plt.subplots(2, 2, figsize=(8.27, 5.5))
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'

subplot_labels = ['A', 'B', 'C', 'D']
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.15, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

color = 'blue'
linestyle = '-'

axes[0, 0].set_xlabel(r"Steps (x$10^6$)", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Actor Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}"))

axes[0, 1].set_xlabel(r"Steps (x$10^6$)", fontsize=LABEL_SIZE)
axes[0, 1].set_ylabel("Critic Loss", fontsize=LABEL_SIZE)
axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}"))

axes[1, 0].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 0].set_ylabel("Training Reward", fontsize=LABEL_SIZE)
axes[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 1].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 1].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

training_metrics_path = os.path.join(base_path, run_dir, "metrics/training_metrics.csv")
training_metrics = pd.read_csv(training_metrics_path)
key_episodes_with_steps = []
for episode in key_episodes:
    closest_idx = np.abs(training_metrics['episode'].values - episode).argmin()
    closest_episode = training_metrics.iloc[closest_idx]['episode']
    steps_at_episode = np.sum(training_metrics['steps'].iloc[:closest_idx+1])
    if abs(closest_episode - episode) <= 5:
        key_episodes_with_steps.append((closest_episode, steps_at_episode))

learning_stats_path = os.path.join(base_path, run_dir, "agent_saves/sac_pytorch_learning_stats_reduced.csv")
learning_stats = pd.read_csv(learning_stats_path)

steps = learning_stats['step'].values
actor_loss = learning_stats['actor_loss'].values
smoothed_loss = moving_average(actor_loss, min(heavy_window_size, len(actor_loss)))
loss_std = pd.Series(actor_loss).rolling(window=min(heavy_window_size, len(actor_loss)), min_periods=1).std().values
axes[0, 0].plot(steps, actor_loss, color='purple', alpha=0.3)
axes[0, 0].plot(steps, smoothed_loss, color=color, linestyle=linestyle)
axes[0, 0].fill_between(
    steps,
    smoothed_loss - loss_std,
    smoothed_loss + loss_std,
    color=color, alpha=0.2
)
for i, (episode, step_value) in enumerate(key_episodes_with_steps):
    if i < len(line_colors):
        axes[0, 0].axvline(x=step_value, color=line_colors[i], linestyle='--', alpha=0.7,
                        linewidth=2)

critic_loss = learning_stats['critic_loss'].values
smoothed_loss = moving_average(critic_loss, min(heavy_window_size, len(critic_loss)))
loss_std = pd.Series(critic_loss).rolling(window=min(heavy_window_size, len(critic_loss)), min_periods=1).std().values
axes[0, 1].plot(steps, critic_loss, color='purple', alpha=0.3)
axes[0, 1].plot(steps, smoothed_loss, color=color, linestyle=linestyle)
axes[0, 1].fill_between(
    steps,
    smoothed_loss - loss_std,
    smoothed_loss + loss_std,
    color=color, alpha=0.2
)
for i, (episode, step_value) in enumerate(key_episodes_with_steps):
    if i < len(line_colors):
        axes[0, 1].axvline(x=step_value, color=line_colors[i], linestyle='--', alpha=0.7,
                        linewidth=2)

training_metrics = pd.read_csv(training_metrics_path)
training_episodes = training_metrics['episode'].values
training_rewards = training_metrics['reward'].values

mask = training_episodes <= episode_limit
training_episodes = training_episodes[mask]
training_rewards = training_rewards[mask]

smoothed_training_rewards = moving_average(training_rewards, min(window_size, len(training_rewards)))
training_std = pd.Series(training_rewards).rolling(window=min(window_size, len(training_rewards)), min_periods=1).std().values
axes[1, 0].plot(training_episodes, training_rewards, color='purple', alpha=0.3)
axes[1, 0].plot(training_episodes, smoothed_training_rewards, color=color, linestyle=linestyle)
axes[1, 0].fill_between(
    training_episodes,
    smoothed_training_rewards - training_std,
    smoothed_training_rewards + training_std,
    color=color, alpha=0.2
)
for i, episode in enumerate(key_episodes):
    if episode <= episode_limit:
        axes[1, 0].axvline(x=episode, color=line_colors[i % len(line_colors)], linestyle='--', alpha=0.7, 
                        linewidth=2)
train_min = min(training_rewards)
y_min = min(train_min, eval_min_reward)
axes[1, 0].set_ylim(y_min, eval_max_reward)
axes[1, 1].set_ylim(y_min, eval_max_reward)

eval_path = os.path.join(base_path, run_dir, "metrics/eval_metrics.csv")
eval_metrics = pd.read_csv(eval_path)
eval_metrics = eval_metrics[eval_metrics['episode'] <= episode_limit]
eval_episodes = eval_metrics['episode'].values
eval_rewards = eval_metrics['eval_reward'].values
if len(eval_rewards) > 0:
    smoothed_eval_rewards = moving_average(eval_rewards, min(window_size, len(eval_rewards)))
    eval_std = pd.Series(eval_rewards).rolling(window=min(window_size, len(eval_rewards)), min_periods=1).std().values
    axes[1, 1].plot(eval_episodes, eval_rewards, color='purple', alpha=0.3)
    axes[1, 1].plot(eval_episodes, smoothed_eval_rewards, color=color, linestyle=linestyle)
    axes[1, 1].fill_between(
        eval_episodes,
        smoothed_eval_rewards - eval_std,
        smoothed_eval_rewards + eval_std,
        color=color, alpha=0.2
    )
    for i, episode in enumerate(key_episodes):
        if episode <= episode_limit:
            axes[1, 1].axvline(x=episode, color=line_colors[i % len(line_colors)], linestyle='--', alpha=0.7, 
                            linewidth=2)

for row in axes:
    for ax in row:
        ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.5)
plt.savefig(os.path.join(base_path, "plots/uniform_2000s.png"), 
            dpi=300, 
            bbox_inches='tight',
            format='png')
plt.close()