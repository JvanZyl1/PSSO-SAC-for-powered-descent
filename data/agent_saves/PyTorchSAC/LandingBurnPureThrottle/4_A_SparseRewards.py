import pandas as pd
import matplotlib.pyplot as plt
import os

def moving_average(data, window_size=20):
    return pd.Series(data).rolling(window=window_size, min_periods=1).mean().values

base_path = "data/agent_saves/PyTorchSAC/LandingBurnPureThrottle"
run_dir = "4_A_Sparse"

window_size = 20
heavy_window_size = 50
episode_limit = 1000

TITLE_SIZE = 10
LABEL_SIZE = 10
TICK_SIZE = 10

fig, axes = plt.subplots(1, 3, figsize=(8.27, 2.5))

plt.rcParams['xtick.labelsize'] = TICK_SIZE
plt.rcParams['ytick.labelsize'] = TICK_SIZE
plt.rcParams['mathtext.fontset'] = 'dejavusans'

subplot_labels = ['A', 'B', 'C']
for i, ax in enumerate(axes.flat):
    ax.text(-0.1, 1.15, subplot_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight='bold', va='top')

color = 'blue'
linestyle = '-'

axes[0].set_title("Actor Loss", fontsize=TITLE_SIZE)
axes[0].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[0].set_ylabel("Actor Loss", fontsize=LABEL_SIZE)
axes[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[1].set_title("Critic Loss", fontsize=TITLE_SIZE)
axes[1].set_xlabel(r"Steps (x$10^3$)", fontsize=LABEL_SIZE)
axes[1].set_ylabel("Critic Loss", fontsize=LABEL_SIZE)
axes[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

axes[2].set_title("Evaluation Rewards", fontsize=TITLE_SIZE)
axes[2].set_xlabel("Episodes", fontsize=LABEL_SIZE)
axes[2].set_ylabel("Evaluation Reward", fontsize=LABEL_SIZE)
axes[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

max_steps = None

learning_stats_path = os.path.join(base_path, run_dir, "agent_saves/sac_pytorch_learning_stats_reduced.csv")
learning_stats = pd.read_csv(learning_stats_path)

steps = learning_stats['step'].values
actor_loss = learning_stats['actor_loss'].values
smoothed_loss = moving_average(actor_loss, min(heavy_window_size, len(actor_loss)))
loss_std = pd.Series(actor_loss).rolling(window=min(heavy_window_size, len(actor_loss)), min_periods=1).std().values
axes[0].plot(steps/1000, smoothed_loss, color=color, linestyle=linestyle)
axes[0].fill_between(
    steps/1000,
    smoothed_loss - loss_std,
    smoothed_loss + loss_std,
    color=color, alpha=0.2
)

critic_loss = learning_stats['critic_loss'].values
smoothed_loss = moving_average(critic_loss, min(heavy_window_size, len(critic_loss)))
loss_std = pd.Series(critic_loss).rolling(window=min(heavy_window_size, len(critic_loss)), min_periods=1).std().values
axes[1].plot(steps/1000, smoothed_loss, color=color, linestyle=linestyle)
axes[1].fill_between(
    steps/1000,
    smoothed_loss - loss_std,
    smoothed_loss + loss_std,
    color=color, alpha=0.2
)

eval_path = os.path.join(base_path, run_dir, "metrics/eval_metrics.csv")
eval_metrics = pd.read_csv(eval_path)
eval_metrics = eval_metrics[eval_metrics['episode'] <= episode_limit]
eval_episodes = eval_metrics['episode'].values
eval_rewards = eval_metrics['eval_reward'].values
smoothed_eval_rewards = moving_average(eval_rewards, min(window_size, len(eval_rewards)))

eval_std = pd.Series(eval_rewards).rolling(window=min(window_size, len(eval_rewards)), min_periods=1).std().values
axes[2].plot(eval_episodes, smoothed_eval_rewards, color=color, linestyle=linestyle)
axes[2].fill_between(
    eval_episodes,
    smoothed_eval_rewards - eval_std,
    smoothed_eval_rewards + eval_std,
    color=color, alpha=0.2
)
for ax in axes:
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(base_path, "plots/sparse_rewards_comparison.png"), 
            dpi=300, 
            bbox_inches='tight',
            format='png')
plt.close()