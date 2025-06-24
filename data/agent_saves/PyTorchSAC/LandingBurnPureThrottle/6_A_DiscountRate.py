import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def moving_average(data, window_size=20):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

base_path = "agent_saves/PyTorchSAC/LandingBurnPureThrottle"
run_dir_A = ("5_A_PER_2500", r"$\gamma = 0.99$")
run_dir_B = ("7_A_gamma", r"$\gamma = 0.95$")
run_dir_C = ("7_B_gamma", r"$\gamma = 0.9$")

window_size = 20
heavy_window_size = 50
episode_limit = 750

TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 10

fig, axes = plt.subplots(1, 3, figsize=(8.57, 2.5))

subplot_labels = ['A', 'B', 'C', 'D']
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.15, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'

legend_handles = []
legend_labels_list = []

axes[0].set_xlabel(r"Steps ($x10^6$)", fontsize=LABEL_SIZE)
axes[0].set_ylabel("Actor Loss", fontsize=LABEL_SIZE)
axes[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}"))

axes[1].set_xlabel(r"Steps ($x10^6$)", fontsize=LABEL_SIZE)
axes[1].set_ylabel("Critic Loss", fontsize=LABEL_SIZE)
axes[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.1f}"))

axes[2].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[2].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

def plot_run_data(run_dir, color, linestyle, label_prefix):
    learning_stats_path = os.path.join(base_path, run_dir, "agent_saves/sac_pytorch_learning_stats_reduced.csv")
    learning_stats = pd.read_csv(learning_stats_path)
    steps = learning_stats['step'].values
    actor_loss = learning_stats['actor_loss'].values
    axes[0].plot(steps, actor_loss, color=color, alpha=0.5, linewidth=0.5)
    smoothed_loss = moving_average(actor_loss, min(heavy_window_size, len(actor_loss)))
    loss_std = pd.Series(actor_loss).rolling(window=min(heavy_window_size, len(actor_loss)), min_periods=1).std().values
    line, = axes[0].plot(steps, smoothed_loss, color=color, linestyle=linestyle)
    if label_prefix not in legend_labels_list:
        legend_handles.append(line)
        legend_labels_list.append(label_prefix)
    axes[0].fill_between(
        steps,
        smoothed_loss - loss_std,
        smoothed_loss + loss_std,
        color=color, alpha=0.3
    )
    print(f"{label_prefix} Actor Loss - Mean: {np.mean(smoothed_loss):.4f}, Std: {np.mean(loss_std):.4f}")
    
    critic_loss = learning_stats['critic_loss'].values
    axes[1].plot(steps, critic_loss, color=color, alpha=0.5, linewidth=0.5)
    smoothed_loss = moving_average(critic_loss, min(heavy_window_size, len(critic_loss)))
    loss_std = pd.Series(critic_loss).rolling(window=min(heavy_window_size, len(critic_loss)), min_periods=1).std().values
    axes[1].plot(steps, smoothed_loss, color=color, linestyle=linestyle)
    axes[1].fill_between(
        steps,
        smoothed_loss - loss_std,
        smoothed_loss + loss_std,
        color=color, alpha=0.3
    )

    eval_path = os.path.join(base_path, run_dir, "metrics/eval_metrics.csv")
    eval_metrics = pd.read_csv(eval_path)
    eval_metrics = eval_metrics[eval_metrics['episode'] <= episode_limit]
    eval_episodes = eval_metrics['episode'].values
    eval_rewards = eval_metrics['eval_reward'].values
    axes[2].plot(eval_episodes, eval_rewards, color=color, alpha=0.5, linewidth=0.5)
    smoothed_eval_rewards = moving_average(eval_rewards, min(window_size, len(eval_rewards)))
    eval_std = pd.Series(eval_rewards).rolling(window=min(window_size, len(eval_rewards)), min_periods=1).std().values
    axes[2].plot(eval_episodes, smoothed_eval_rewards, color=color, linestyle=linestyle)
    axes[2].fill_between(
        eval_episodes,
        smoothed_eval_rewards - eval_std,
        smoothed_eval_rewards + eval_std,
        color=color, alpha=0.3
    )

plot_run_data(run_dir_A[0], 'blue', '-', run_dir_A[1])
plot_run_data(run_dir_B[0], 'green', '-', run_dir_B[1])
plot_run_data(run_dir_C[0], 'purple', '-', run_dir_C[1])

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.legend(
    handles=legend_handles,
    labels=legend_labels_list,
    loc='lower center',
    ncol=4,
    fontsize=TICK_SIZE,
    frameon=True,
    bbox_to_anchor=(0.5, -0.1),
    columnspacing=2.0,
    handletextpad=0.2
)

save_path = base_path + "/plots/discount_rate_comparison.png"
plt.savefig(save_path, 
            dpi=300, 
            bbox_inches='tight',
            format='png')
plt.close()