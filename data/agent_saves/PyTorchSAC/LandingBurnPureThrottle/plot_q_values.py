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

plt.figure(figsize=(8.27, 2.5))
plt.xlabel(r"Steps ($10^6$)", fontsize=LABEL_SIZE)
plt.ylabel("Q-values", fontsize=LABEL_SIZE)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE)
plt.grid(True, alpha=0.3)

plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}"))

training_metrics_path = os.path.join(base_path, run_dir, "metrics/training_metrics.csv")
key_episodes_with_steps = []

training_metrics = pd.read_csv(training_metrics_path)
for episode in key_episodes:
    closest_idx = np.abs(training_metrics['episode'].values - episode).argmin()
    closest_episode = training_metrics.iloc[closest_idx]['episode']
    steps_at_episode = np.sum(training_metrics['steps'].iloc[:closest_idx+1])
    if abs(closest_episode - episode) <= 5:
        key_episodes_with_steps.append((closest_episode, steps_at_episode))
        
print(f"Steps @ key episodes: {key_episodes_with_steps}")

learning_stats_path = os.path.join(base_path, run_dir, "learning_stats/sac_learning_stats_reduced.csv")
learning_stats = pd.read_csv(learning_stats_path)
q_columns = ['q_value_mean', 'q_value_min', 'q_value_max', 'q_value_std']
steps = learning_stats['step'].values
q_mean = learning_stats['q_value_mean'].values
q_min = learning_stats['q_value_min'].values
q_max = learning_stats['q_value_max'].values
q_std = learning_stats['q_value_std'].values
smoothed_mean = moving_average(q_mean, min(heavy_window_size, len(q_mean)))
smoothed_min = moving_average(q_min, min(heavy_window_size, len(q_min)))
smoothed_max = moving_average(q_max, min(heavy_window_size, len(q_max)))
smoothed_std = moving_average(q_std, min(heavy_window_size, len(q_std)))

plt.plot(steps, smoothed_mean, color='blue', linestyle='-', 
        label="Mean Q-value", linewidth=2)
plt.fill_between(
    steps,
    smoothed_min,
    smoothed_max,
    color='blue', alpha=0.2,
    label="Min-Max Range"
)
plt.fill_between(
    steps,
    smoothed_mean - smoothed_std,
    smoothed_mean + smoothed_std,
    color='green', alpha=0.3,
    label=r"$\pm \sigma$"
)
for i, (episode, step_value) in enumerate(key_episodes_with_steps):
    if i < len(line_colors):
        plt.axvline(x=step_value, color=line_colors[i], linestyle='--', alpha=0.7,
                    linewidth=2)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "plots/q_values.png"), 
            dpi=300, 
            bbox_inches='tight',
            format='png')
plt.close()