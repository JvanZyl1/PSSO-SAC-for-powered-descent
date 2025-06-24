import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window_size=20):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

base_path = "data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
runs = {
    "alpha_lr_0.001": "1_A_2",
    "alpha_lr_0.0003": "1_A_1",
    "alpha_lr_0.0001": "1_A_3"
}

legend_labels = {
    "alpha_lr_0.001": r"$\alpha_{\log(\nu)}$ = 0.001",
    "alpha_lr_0.0003": r"$\alpha_{\log(\nu)}$ = 0.0003",
    "alpha_lr_0.0001": r"$\alpha_{\log(\nu)}$ = 0.0001"
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
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

subplot_labels = ['A', 'B', 'C', 'D']
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.15, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

colors = ['red', 'blue', 'green']
linestyles = ['-', '-', '-']

axes[0, 0].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 0].set_ylabel("Temperature Loss", fontsize=LABEL_SIZE)
axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[0, 1].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0, 1].set_ylabel("Temperature", fontsize=LABEL_SIZE)
axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1, 0].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[1, 0].set_ylabel("Training Reward", fontsize=LABEL_SIZE)
axes[1, 0].set_yscale('log')  # Set log scale for y-axis
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
    training_metrics = pd.read_csv(training_path)
    training_metrics = training_metrics[training_metrics['episode'] <= episode_limit]
    max_steps[label] = training_metrics['steps'].sum()
    
    learning_stats_path = os.path.join(base_path, run_dir, "agent_saves/sac_pytorch_learning_stats_reduced.csv")
    learning_stats = pd.read_csv(learning_stats_path)
    learning_stats = learning_stats[learning_stats['step'] <= max_steps[label]]
    steps = learning_stats['step'].values
    alpha_loss = learning_stats['alpha_loss'].values
    
    if len(alpha_loss) > 0:
        smoothed_loss = moving_average(alpha_loss, min(heavy_window_size, len(alpha_loss)))
        loss_std = pd.Series(alpha_loss).rolling(window=min(heavy_window_size, len(alpha_loss)), min_periods=1).std().values
        line, = axes[0, 0].plot(steps/1000, smoothed_loss, 
                    color=colors[i], linestyle=linestyles[i])
        if label not in legend_labels_list:
            legend_handles.append(line)
            legend_labels_list.append(legend_labels[label])
        lower_bound = smoothed_loss - loss_std
        upper_bound = smoothed_loss + loss_std
        
        axes[0, 0].fill_between(
            steps/1000,
            lower_bound,
            upper_bound,
            color=colors[i], alpha=0.2
        )
    
    if 'alpha_value' in learning_stats.columns and not learning_stats['alpha_value'].isnull().all():
        alpha_value = learning_stats['alpha_value'].fillna(0).values
        smoothed_value = moving_average(alpha_value, window_size)
        value_std = pd.Series(alpha_value).rolling(window=window_size, min_periods=1).std().values
        
        axes[0, 1].plot(steps/1000, smoothed_value, 
                    color=colors[i], linestyle=linestyles[i])
        axes[0, 1].fill_between(
            steps/1000,
            np.maximum(0, smoothed_value - value_std),
            smoothed_value + value_std,
            color=colors[i], alpha=0.2
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
    ncol=3,
    fontsize=LABEL_SIZE,
    frameon=True,
    bbox_to_anchor=(0.5, 0.04),
    columnspacing=6.0,
    handletextpad=1.0
)

save_path = base_path + "/plots/temperature_comparison_1000_episodes.png"	
plt.savefig(save_path, dpi=300)
plt.close()