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

base_path = "data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
runs = {
    "polyak_0.999": "3_A_2",
    "polyak_0.995": "3_A_1",
    "polyak_0.98": "3_A_3",
    "polyak_0.995_fast": "3_A_4"
}

legend_labels = {
    "polyak_0.999": r"$\tau$ = 0.001",
    "polyak_0.995": r"$\tau$ = 0.005",
    "polyak_0.98": r"$\tau$ = 0.01",
    "polyak_0.995_fast": r"$\tau$ = 0.1"
}

window_size = 20
heavy_window_size = 50
episode_limit = 1000

TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 10

fig, axes = plt.subplots(3, 2, figsize=(8.27, 7.5))

plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'

subplot_labels = ['A', 'B', 'C', 'D', 'E', 'F']
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.15, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

colors = ['red', 'blue', 'green', 'purple']
linestyles = ['-', '-', '-', '-']

axes[0, 0].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Critic Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[0, 1].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 1].set_ylabel("Temperature", fontsize=LABEL_SIZE)
axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 0].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[1, 0].set_ylabel("Q-Value", fontsize=LABEL_SIZE)
axes[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 1].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[1, 1].set_ylabel("Target Q-Value", fontsize=LABEL_SIZE)
axes[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[2, 0].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[2, 0].set_ylabel("Training Reward", fontsize=LABEL_SIZE)
axes[2, 0].set_yscale('log')
axes[2, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[2, 1].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[2, 1].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[2, 1].set_yscale('log')
axes[2, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

max_steps = {}
legend_handles = []
legend_labels_list = []

for i, (label, run_dir) in enumerate(runs.items()):
    training_path = os.path.join(base_path, run_dir, "metrics/training_metrics.csv")
    if os.path.exists(training_path):
        training_metrics = pd.read_csv(training_path)
        
        training_metrics = training_metrics[training_metrics['episode'] <= episode_limit]
        
        if not training_metrics.empty:
            max_steps[label] = training_metrics['steps'].sum()
            print(f"{label} - Max steps at episode {episode_limit}: {max_steps[label]}")
    
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
    
    alpha_values = learning_stats['alpha_value'].fillna(0).values
    smoothed_alpha = moving_average(alpha_values, window_size)
    axes[0, 1].plot(
        steps/1000,
        smoothed_alpha,
        color=colors[i],
        linestyle=linestyles[i],
        linewidth=2
    )
    
    q_mean = learning_stats['q_value_mean'].fillna(0).values
    q_std = learning_stats['q_value_std'].fillna(0).values
    q_mean = filter_outliers(q_mean, k=2.0)
    smoothed_q_mean = moving_average(q_mean, window_size)
    axes[1, 0].fill_between(
        steps/1000,
        smoothed_q_mean - q_std,
        smoothed_q_mean + q_std,
        color=colors[i],
        alpha=0.3
    )
    axes[1, 0].plot(
        steps/1000,
        smoothed_q_mean,
        color=colors[i],
        linestyle=linestyles[i],
        linewidth=2
    )
    
    target_q_mean = learning_stats['target_q_mean'].fillna(0).values
    target_q_std = learning_stats['target_q_std'].fillna(0).values
    target_q_mean = filter_outliers(target_q_mean, k=2.0)
    smoothed_target_q_mean = moving_average(target_q_mean, window_size)
    axes[1, 1].fill_between(
        steps/1000,
        smoothed_target_q_mean - target_q_std,
        smoothed_target_q_mean + target_q_std,
        color=colors[i],
        alpha=0.3
    )
    axes[1, 1].plot(
        steps/1000,
        smoothed_target_q_mean,
        color=colors[i],
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
    axes[2, 0].plot(episodes, smoothed_rewards, 
                color=colors[i], linestyle=linestyles[i])
    lower_bound = smoothed_rewards / np.exp(log_std)
    upper_bound = smoothed_rewards * np.exp(log_std)
    axes[2, 0].fill_between(
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
    axes[2, 1].plot(eval_episodes, smoothed_eval_rewards, 
                color=colors[i], linestyle=linestyles[i])
    lower_bound = smoothed_eval_rewards / np.exp(log_eval_std)
    upper_bound = smoothed_eval_rewards * np.exp(log_eval_std)
    axes[2, 1].fill_between(
        eval_episodes,
        lower_bound,
        upper_bound,
        color=colors[i], alpha=0.2
    )

for ax_row in axes:
    for ax in ax_row:
        ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

fig.legend(
    handles=legend_handles,
    labels=legend_labels_list,
    loc='lower center',
    ncol=len(legend_handles),
    fontsize=LABEL_SIZE,
    frameon=True,
    bbox_to_anchor=(0.5, 0.04),
    columnspacing=6.0,
    handletextpad=1.0
)
plt.savefig(os.path.join(base_path, "plots/polyak_averaging_comparison.png"), dpi=300)
plt.close()