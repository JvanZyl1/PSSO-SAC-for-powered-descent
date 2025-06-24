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
    "actor_lr_0": "2_B_2",
    "actor_lr_1": "2_B_3",
    "actor_lr_2": "2_B_0",
    "actor_lr_3": "2_B_1"
}

legend_labels = {
    "actor_lr_0": r"$\alpha_\zeta = 4e-3$",
    "actor_lr_1": r"$\alpha_\zeta = 1e-3$",
    "actor_lr_2": r"$\alpha_\zeta = 1e-4$",
    "actor_lr_3": r"$\alpha_\zeta = 1e-5$"
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

colors = ['red', 'blue', 'green', 'purple']
linestyles = ['-', '-', '-', '-']

axes[0, 0].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Actor Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[0, 1].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 1].set_ylabel("Temperature", fontsize=LABEL_SIZE)
axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 0].set_ylabel("Training Reward", fontsize=LABEL_SIZE)
axes[1, 0].set_yscale('log')
axes[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 1].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 1].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[1, 1].set_yscale('log')
axes[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

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
    learning_stats = learning_stats[learning_stats['step'] <= max_steps[label]]
    steps = learning_stats['step'].values
    actor_loss = learning_stats['actor_loss'].values
    filtered_actor_loss = filter_outliers(actor_loss, k=1.5)

    line, = axes[0, 0].plot(steps/1000, filtered_actor_loss, 
                color=colors[i], linestyle=linestyles[i])

    if label not in legend_labels_list:
        legend_handles.append(line)
        legend_labels_list.append(legend_labels[label])

    alpha_mean = learning_stats['alpha_value'].fillna(0).values
    alpha_mean = filter_outliers(alpha_mean, k=2.0)
    
    smoothed_alpha_mean = moving_average(alpha_mean, window_size)
    
    axes[0, 1].plot(
        steps/1000,
        smoothed_alpha_mean,
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

plt.tight_layout(rect=[0, 0.08, 1, 0.95])

fig.legend(
    handles=legend_handles,
    labels=legend_labels_list,
    loc='lower center',
    ncol=len(legend_handles),
    fontsize=LABEL_SIZE,
    frameon=True,
    bbox_to_anchor=(0.5, 0.04),
    columnspacing=2.0,
    handletextpad=1.0
)

plt.savefig(os.path.join(base_path, "plots/actor_loss_comparison.png"), dpi=300)
plt.close()