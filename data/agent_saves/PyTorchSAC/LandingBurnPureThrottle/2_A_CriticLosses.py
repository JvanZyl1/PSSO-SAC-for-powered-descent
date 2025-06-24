import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window_size=20):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

def filter_outliers(data, k=1.5):
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    return np.clip(data, lower_bound, upper_bound)

base_path = "RocketTrajectoryOptimisation/data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
runs = {
    "run_1": "2_A_4",
    "run_2": "2_A_1",
    "run_3": "2_A_2",
    "run_4": "2_A_3"
}

legend_labels = {
    "run_1": r"$\alpha_{Q}$ = 0.005",
    "run_2": r"$\alpha_{Q}$ = 0.001",
    "run_3": r"$\alpha_{Q}$ = 0.0001",
    "run_4": r"$\alpha_{Q}$ = 0.00001"
}

window_size = 20
heavy_window_size = 50
episode_limit = 1000

TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 10

fig, axes = plt.subplots(2, 2, figsize=(8.27, 5.5))

plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'

subplot_labels = ['A', 'B', 'C', 'D']
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.15, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

colors = ['blue', 'red', 'green', 'purple']
linestyles = ['-', '-', '-', '-']

axes[0, 0].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Critic Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[0, 1].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 1].set_ylabel("Q-Value", fontsize=LABEL_SIZE)
axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 0].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 0].set_ylabel("Training Reward", fontsize=LABEL_SIZE)
axes[1, 0].set_yscale('log')  # Set log scale for y-axis
axes[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 1].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 1].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[1, 1].set_yscale('log')  # Set log scale for y-axis
axes[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

max_steps = {}

legend_handles = []
legend_labels_list = []

for i, (label, run_dir) in enumerate(runs.items()):
    training_path = os.path.join(base_path, run_dir, "metrics/training_metrics.csv")
    training_metrics = pd.read_csv(training_path)
    training_metrics = training_metrics[training_metrics['episode'] <= episode_limit]
    max_steps[label] = training_metrics['steps'].sum()
    print(f"{label} - Max steps at episode {episode_limit}: {max_steps[label]}")

    # Load learning stats
    learning_stats_path = os.path.join(base_path, run_dir, "agent_saves/sac_pytorch_learning_stats_reduced.csv")
    learning_stats = pd.read_csv(learning_stats_path)
    if label in max_steps:
        learning_stats = learning_stats[learning_stats['step'] <= max_steps[label]]
    steps = learning_stats['step'].values
    critic_loss = learning_stats['critic_loss'].values
    filtered_critic_loss = filter_outliers(critic_loss, k=1.5)
    smoothed_loss = moving_average(filtered_critic_loss, min(heavy_window_size, len(filtered_critic_loss)))
    loss_std = pd.Series(filtered_critic_loss).rolling(window=min(heavy_window_size, len(filtered_critic_loss)), min_periods=1).std().values
    line, = axes[0, 0].plot(steps/1000, smoothed_loss, 
                color=colors[i], linestyle=linestyles[i])
    if label not in legend_labels_list:
        legend_handles.append(line)
        legend_labels_list.append(legend_labels[label])
    lower_bound = np.maximum(0, smoothed_loss - loss_std)
    upper_bound = smoothed_loss + loss_std
    
    axes[0, 0].fill_between(
        steps/1000,
        lower_bound,
        upper_bound,
        color=colors[i], alpha=0.2
    )

    q_mean = learning_stats['q_value_mean'].fillna(0).values
    q_std = learning_stats['q_value_std'].fillna(0).values
    q_mean = filter_outliers(q_mean, k=2.0)
    smoothed_q_mean = moving_average(q_mean, window_size)
    base_color = colors[i]
    axes[0, 1].fill_between(
        steps/1000,
        smoothed_q_mean - q_std,
        smoothed_q_mean + q_std,
        color=base_color,
        alpha=0.3
    )
    line, = axes[0, 1].plot(
        steps/1000,
        smoothed_q_mean,
        color=base_color,
        linestyle=linestyles[i],
        linewidth=2
    )
                
    
    training_metrics = pd.read_csv(training_path)
    training_metrics = training_metrics[training_metrics['episode'] <= episode_limit]

    mask = training_metrics['reward'] > 0
    episodes = training_metrics['episode'][mask].values
    rewards = training_metrics['reward'][mask].values
    
    smoothed_rewards = moving_average(rewards, min(heavy_window_size, len(rewards)))
    
    log_rewards = np.log(rewards)
    log_std = pd.Series(log_rewards).rolling(window=min(heavy_window_size, len(rewards)), min_periods=1).std().values
    
    axes[1, 0].plot(episodes, smoothed_rewards, 
                color=colors[i], linestyle=linestyles[i])
    
    lower_bound = smoothed_rewards / np.exp(log_std)
    upper_bound = smoothed_rewards * np.exp(log_std)
    
    axes[1, 0].fill_between(
        episodes,
        lower_bound,
        upper_bound,
        color=colors[i], alpha=0.2
    )

    eval_path = os.path.join(base_path, run_dir, "metrics/eval_metrics.csv")
    eval_metrics = pd.read_csv(eval_path)
    eval_metrics = eval_metrics[eval_metrics['episode'] <= episode_limit]
    mask = eval_metrics['eval_reward'] > 0
    eval_episodes = eval_metrics['episode'][mask].values
    eval_rewards = eval_metrics['eval_reward'][mask].values
    
    smoothed_eval_rewards = moving_average(eval_rewards, min(window_size, len(eval_rewards)))
    
    log_eval_rewards = np.log(eval_rewards)
    log_eval_std = pd.Series(log_eval_rewards).rolling(window=min(window_size, len(eval_rewards)), min_periods=1).std().values
    
    axes[1, 1].plot(eval_episodes, smoothed_eval_rewards, 
                color=colors[i], linestyle=linestyles[i])
    
    lower_bound = smoothed_eval_rewards / np.exp(log_eval_std)
    upper_bound = smoothed_eval_rewards * np.exp(log_eval_std)
    
    axes[1, 1].fill_between(
        eval_episodes,
        lower_bound,
        upper_bound,
        color=colors[i], alpha=0.2
    )

for ax_row in axes:
    for ax in ax_row:
        ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.08, 1, 0.95], h_pad=0.1)
fig.legend(
    handles=legend_handles,
    labels=legend_labels_list,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.02),
    fontsize=LABEL_SIZE,
    ncol=len(legend_handles)
)
plt.savefig(os.path.join(base_path, "plots/critic_losses_comparison_2A.png"), dpi=300, bbox_inches='tight')
plt.close()