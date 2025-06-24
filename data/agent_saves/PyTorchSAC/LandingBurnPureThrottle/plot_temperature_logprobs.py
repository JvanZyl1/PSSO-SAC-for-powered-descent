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

TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 10

key_episodes = [390, 730, 1070, 1420, 1610]
line_colors = ['red', 'green', 'orange', 'purple', 'cyan']

fig, axes = plt.subplots(1, 2, figsize=(8.27, 2.5))
plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'
subplot_labels = ['A', 'B']
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.15, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

color = 'blue'
linestyle = '-'

axes[0].set_xlabel(r"Steps ($10^6$)", fontsize=LABEL_SIZE)
axes[0].set_ylabel("Temperature", fontsize=LABEL_SIZE)
axes[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}"))
axes[0].grid(True, alpha=0.3)
axes[0].yaxis.set_major_locator(plt.MaxNLocator(5))

axes[1].set_xlabel(r"Steps ($10^6$)", fontsize=LABEL_SIZE)
axes[1].set_ylabel("Log probability", fontsize=LABEL_SIZE)
axes[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}"))
axes[1].grid(True, alpha=0.3)
axes[1].yaxis.set_major_locator(plt.MaxNLocator(5))

training_metrics_path = os.path.join(base_path, run_dir, "metrics/training_metrics.csv")
key_episodes_with_steps = []

if os.path.exists(training_metrics_path):
    training_metrics = pd.read_csv(training_metrics_path)
    
for episode in key_episodes:
    closest_idx = np.abs(training_metrics['episode'].values - episode).argmin()
    closest_episode = training_metrics.iloc[closest_idx]['episode']
    steps_at_episode = np.sum(training_metrics['steps'].iloc[:closest_idx+1])
    if abs(closest_episode - episode) <= 5:
        key_episodes_with_steps.append((closest_episode, steps_at_episode))

learning_stats_path = os.path.join(base_path, run_dir, "learning_stats/sac_learning_stats_reduced.csv")
learning_stats = pd.read_csv(learning_stats_path)
steps = learning_stats['step'].values
alpha_values = learning_stats['alpha_value'].values
smoothed_alpha = moving_average(alpha_values, min(heavy_window_size, len(alpha_values)))
axes[0].plot(steps, smoothed_alpha, color=color, linestyle=linestyle, linewidth=2)
for i, (episode, step_value) in enumerate(key_episodes_with_steps):
    if i < len(line_colors):
        axes[0].axvline(x=step_value, color=line_colors[i], linestyle='--', alpha=0.7,
                        linewidth=2)

log_prob_mean = learning_stats['log_prob_mean'].values
log_prob_min = learning_stats['log_prob_min'].values
log_prob_max = learning_stats['log_prob_max'].values
log_prob_std = learning_stats['log_prob_std'].values
smoothed_mean = moving_average(log_prob_mean, min(heavy_window_size, len(log_prob_mean)))
smoothed_min = moving_average(log_prob_min, min(heavy_window_size, len(log_prob_min)))
smoothed_max = moving_average(log_prob_max, min(heavy_window_size, len(log_prob_max)))
smoothed_std = moving_average(log_prob_std, min(heavy_window_size, len(log_prob_std)))
axes[1].plot(steps, smoothed_mean, color='blue', linestyle=linestyle, 
            label="Mean", linewidth=2)
axes[1].fill_between(
    steps,
    smoothed_min,
    smoothed_max,
    color='blue', alpha=0.2,
    label="Min-Max Range"
)
axes[1].fill_between(
    steps,
    smoothed_mean - smoothed_std,
    smoothed_mean + smoothed_std,
    color='green', alpha=0.3,
    label="std"
)
for i, (episode, step_value) in enumerate(key_episodes_with_steps):
    if i < len(line_colors):
        axes[1].axvline(x=step_value, color=line_colors[i], linestyle='--', alpha=0.7,
                        linewidth=2)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "plots/temperature_logprobs.png"), 
            dpi=300, 
            bbox_inches='tight',
            format='png')
plt.close()