import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

def agent_plotter_sac(agent):
    save_path = agent.save_path

    # Plot critic losses
    plt.figure(figsize=(20,15))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.4, wspace=0.3)
    plt.suptitle('Reinforcement Learning', fontsize = 32)
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(agent.critic_losses_mean, label="Critic Loss", linewidth = 2, color = 'blue')
    ax1.plot(agent.critic_weighted_mse_losses_mean, label="Critic Weighted MSE Loss", linewidth = 4, color = 'red', linestyle = '--')
    ax1.plot(agent.critic_l2_regs_mean, label="Critic L2 Reg", linewidth = 4, color = 'green')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Loss", fontsize = 20)
    ax1.set_title("Critic Loss", fontsize = 22)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=20)
    ax1.grid()

    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(agent.actor_losses_mean, label="Actor Loss", linewidth = 2, color = 'blue')
    ax2.plot(agent.actor_entropy_losses_mean, label="Actor Entropy Loss", linewidth = 4, color = 'red', linestyle = '--')
    ax2.plot(agent.actor_q_losses_mean, label="Actor Q Loss", linewidth = 4, color = 'green')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("Loss", fontsize = 20)
    ax2.set_title("Actor Loss", fontsize = 22)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.legend(fontsize=20)
    ax2.grid()

    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(agent.temperature_losses_mean, label="Temperature Loss", linewidth = 4, color = 'blue')
    ax3.set_xlabel("Episode", fontsize = 20)
    ax3.set_ylabel("Loss", fontsize = 20)
    ax3.set_title("Temperature Loss", fontsize = 22)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.grid()

    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(agent.temperature_values_mean, label="Temperature Value", linewidth = 4, color = 'blue')
    ax4.set_xlabel("Episode", fontsize = 20)
    ax4.set_ylabel("Value", fontsize = 20)
    ax4.set_title("Temperature Value", fontsize = 22)
    ax4.tick_params(axis='both', which='major', labelsize=16)
    ax4.grid()
    plt.savefig(save_path + "sac_losses.png")
    plt.close()

    # Plot number of steps
    plt.figure(figsize=(10, 5))
    plt.plot(agent.number_of_steps, label="Steps", linewidth = 4, color = 'blue')
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Actor update steps", fontsize=20)
    plt.title("Number of Actor Update Steps", fontsize=22)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig(save_path + "number_of_steps.png")
    plt.close()

    # Now do the bounds on action std. and mean.
    min_y = min(min(agent.action_std_min), min(agent.action_mean_min))
    max_y = max(max(agent.action_std_max), max(agent.action_mean_max))
    steps = np.arange(len(agent.action_std_mean))
    plt.figure(figsize=(10, 5))
    plt.suptitle('Action Randomness', fontsize = 32)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.4, hspace=1.0)
    ax1 = plt.subplot(gs[0])
    ax1.plot(steps, agent.action_std_mean, label=r"$\mu(\sigma_a)$", linewidth = 4, color = 'blue')
    # min-max band in light tint , action_std
    ax1.fill_between(steps, agent.action_std_min, agent.action_std_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(steps, np.array(agent.action_std_mean) - np.array(agent.action_std_std), np.array(agent.action_std_mean) + np.array(agent.action_std_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Steps", fontsize = 20)
    ax1.set_ylabel("Action Standard Deviation", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_ylim(min_y, max_y)
    ax1.grid()
    # Now same for action mean.
    ax2 = plt.subplot(gs[1])
    ax2.plot(steps, agent.action_mean_mean, label=r"$\mu$", linewidth = 4, color = 'blue')
    ax2.fill_between(steps, agent.action_mean_min, agent.action_mean_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(steps, np.array(agent.action_mean_mean) - np.array(agent.action_mean_std), np.array(agent.action_mean_mean) + np.array(agent.action_mean_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Steps", fontsize = 20)
    ax2.set_ylabel("Action Mean", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()
    ax2.set_ylim(min_y, max_y)
    
    # Create a single legend for both plots and place it to the right of the right plot
    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=16)
    
    plt.savefig(save_path + "action_std_and_mean.png", bbox_inches='tight')
    plt.close()

    # Now log probabilities, same style uncertainty bands.
    plt.figure(figsize=(10, 5))
    plt.suptitle('Log Probabilities', fontsize = 32)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.45)
    ax1 = plt.subplot(gs[0])
    ax1.plot(steps, agent.log_probabilities_mean, label=r"$- \mu_{\log p(a|\mu, \sigma)}$", linewidth = 4, color = 'blue')
    ax1.fill_between(steps, agent.log_probabilities_min, agent.log_probabilities_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(steps, np.array(agent.log_probabilities_mean) - np.array(agent.log_probabilities_std), np.array(agent.log_probabilities_mean) + np.array(agent.log_probabilities_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Steps", fontsize = 20)
    ax1.set_ylabel("Log Probabilities", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()
    ax2 = plt.subplot(gs[1])
    ax2.plot(steps, agent.log_probabilities_mean, label=r"$- \mu_{\log p(a|\mu, \sigma)}$", linewidth = 4, color = 'blue')
    ax2.fill_between(steps, np.array(agent.log_probabilities_mean) - np.array(agent.log_probabilities_std), np.array(agent.log_probabilities_mean) + np.array(agent.log_probabilities_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Steps", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()

    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "log_probabilities.png", bbox_inches='tight')
    plt.close()

    # Now td errors, same style uncertainty bands.
    plt.figure(figsize=(10, 5))
    plt.suptitle('TD Errors', fontsize = 32)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.2, hspace=0.45)
    ax1 = plt.subplot(gs[0])
    ax1.plot(steps, agent.td_errors_mean, label=r"$\mu$", linewidth = 2, color = 'blue')
    ax1.fill_between(steps, agent.td_errors_min, agent.td_errors_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(steps, np.array(agent.td_errors_mean) - np.array(agent.td_errors_std), np.array(agent.td_errors_mean) + np.array(agent.td_errors_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Steps", fontsize = 20)
    ax1.set_ylabel("TD Errors", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()

    ax2 = plt.subplot(gs[1])
    ax2.plot(steps, agent.td_errors_mean, label=r"$\mu$", linewidth = 2, color = 'blue')
    ax2.fill_between(steps, np.array(agent.td_errors_mean) - np.array(agent.td_errors_std), np.array(agent.td_errors_mean) + np.array(agent.td_errors_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Steps", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()

    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "td_errors.png", bbox_inches='tight')
    plt.close()

    # Now same for sampled rewards.
    plt.figure(figsize=(10, 5))
    plt.suptitle('Sampled Experiences', fontsize = 32)
    gs = gridspec.GridSpec(3,1, height_ratios=[1, 1, 1], wspace=0.4, hspace=0.4)
    ax1 = plt.subplot(gs[0])
    ax1.plot(steps, agent.sampled_rewards_mean, label=r"$\mu_r$", linewidth = 2, color = 'blue')
    ax1.fill_between(steps, agent.sampled_rewards_min, agent.sampled_rewards_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(steps, np.array(agent.sampled_rewards_mean) - np.array(agent.sampled_rewards_std), np.array(agent.sampled_rewards_mean) + np.array(agent.sampled_rewards_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_ylabel("Rewards", fontsize = 18)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()

    # Now same for sampled states.
    ax2 = plt.subplot(gs[1])
    ax2.plot(steps, agent.sampled_states_mean, label=r"$\mu_s$", linewidth = 2, color = 'blue')
    ax2.fill_between(steps, agent.sampled_states_min, agent.sampled_states_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(steps, np.array(agent.sampled_states_mean) - np.array(agent.sampled_states_std), np.array(agent.sampled_states_mean) + np.array(agent.sampled_states_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_ylabel("States", fontsize = 18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()

    # Now same for sampled actions.
    ax3 = plt.subplot(gs[2])
    ax3.plot(steps, agent.sampled_actions_mean, label=r"$\mu$", linewidth = 2, color = 'blue')
    ax3.fill_between(steps, agent.sampled_actions_min, agent.sampled_actions_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax3.fill_between(steps, np.array(agent.sampled_actions_mean) - np.array(agent.sampled_actions_std), np.array(agent.sampled_actions_mean) + np.array(agent.sampled_actions_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax3.set_ylabel("Actions", fontsize = 18)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.grid()

    handles, labels = ax3.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "sampled_experiences.png", bbox_inches='tight')
    plt.close()
    
    # Critic losses uncertainty bands.
    episodes = np.arange(len(agent.critic_losses_mean))
    plt.figure(figsize=(10, 5))
    plt.suptitle('Critic Losses', fontsize = 16)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.25, hspace=0.6)
    ax1 = plt.subplot(gs[0])
    ax1.plot(episodes, agent.critic_losses_mean, label="Critic Loss", linewidth = 2, color = 'blue')
    ax1.fill_between(episodes, agent.critic_losses_min, agent.critic_losses_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(episodes, np.array(agent.critic_losses_mean) - np.array(agent.critic_losses_std), np.array(agent.critic_losses_mean) + np.array(agent.critic_losses_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Loss", fontsize = 20)
    ax1.set_title("Critic Loss", fontsize = 16)
    ax1.set_yscale('log')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()

    ax2 = plt.subplot(gs[1])
    ax2.plot(episodes, agent.critic_weighted_mse_losses_mean, label="Critic Weighted MSE Loss", linewidth = 2, color = 'blue')
    ax2.fill_between(episodes, agent.critic_weighted_mse_losses_min, agent.critic_weighted_mse_losses_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(episodes, np.array(agent.critic_weighted_mse_losses_mean) - np.array(agent.critic_weighted_mse_losses_std), np.array(agent.critic_weighted_mse_losses_mean) + np.array(agent.critic_weighted_mse_losses_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_title("Critic Weighted MSE Loss", fontsize = 16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()
    ax2.set_yscale('log')
    ax3 = plt.subplot(gs[2])
    ax3.plot(episodes, agent.critic_l2_regs_mean, label=r"$\mu$", linewidth = 2, color = 'blue')
    ax3.fill_between(episodes, agent.critic_l2_regs_min, agent.critic_l2_regs_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax3.fill_between(episodes, np.array(agent.critic_l2_regs_mean) - np.array(agent.critic_l2_regs_std), np.array(agent.critic_l2_regs_mean) + np.array(agent.critic_l2_regs_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax3.set_xlabel("Episode", fontsize = 20)
    ax3.set_title("Critic L2 Reg", fontsize = 16)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.grid()

    handles, labels = ax3.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "critic_losses.png", bbox_inches='tight')
    plt.close()

    # Actor losses uncertainty bands.
    plt.figure(figsize=(10, 5))
    plt.suptitle('Actor Losses', fontsize = 16)
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.25, hspace=0.6)
    ax1 = plt.subplot(gs[0])
    ax1.plot(episodes, agent.actor_losses_mean, label="Actor Loss", linewidth = 2, color = 'blue')
    ax1.fill_between(episodes, agent.actor_losses_min, agent.actor_losses_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(episodes, np.array(agent.actor_losses_mean) - np.array(agent.actor_losses_std), np.array(agent.actor_losses_mean) + np.array(agent.actor_losses_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Loss", fontsize = 20)
    ax1.set_title("Actor Loss", fontsize = 16)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()

    ax2 = plt.subplot(gs[1])
    ax2.plot(episodes, agent.actor_entropy_losses_mean, label="Actor Entropy Loss", linewidth = 2, color = 'blue')
    ax2.fill_between(episodes, agent.actor_entropy_losses_min, agent.actor_entropy_losses_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(episodes, np.array(agent.actor_entropy_losses_mean) - np.array(agent.actor_entropy_losses_std), np.array(agent.actor_entropy_losses_mean) + np.array(agent.actor_entropy_losses_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_title("Actor Entropy Loss", fontsize = 16)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()

    ax3 = plt.subplot(gs[2])
    ax3.plot(episodes, agent.actor_q_losses_mean, label=r"$\mu$", linewidth = 2, color = 'blue')
    ax3.fill_between(episodes, agent.actor_q_losses_min, agent.actor_q_losses_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax3.fill_between(episodes, np.array(agent.actor_q_losses_mean) - np.array(agent.actor_q_losses_std), np.array(agent.actor_q_losses_mean) + np.array(agent.actor_q_losses_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax3.set_xlabel("Episode", fontsize = 20)
    ax3.set_title("Actor Q Loss", fontsize = 16)
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.grid()

    handles, labels = ax3.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "actor_losses.png", bbox_inches='tight')
    plt.close()

    # Temperature losses uncertainty bands.
    plt.figure(figsize=(10, 5))
    plt.suptitle('Temperature Losses', fontsize = 16)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.4, hspace=0.6)
    ax1 = plt.subplot(gs[0])
    ax1.plot(episodes, agent.temperature_losses_mean, label=r"$\mu$", linewidth = 2, color = 'blue')
    ax1.fill_between(episodes, agent.temperature_losses_min, agent.temperature_losses_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(episodes, np.array(agent.temperature_losses_mean) - np.array(agent.temperature_losses_std), np.array(agent.temperature_losses_mean) + np.array(agent.temperature_losses_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Loss", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()

    ax2 = plt.subplot(gs[1])
    ax2.plot(episodes, agent.temperature_values_mean, label=r"$\mu$", linewidth = 2, color = 'blue')
    ax2.fill_between(episodes, agent.temperature_values_min, agent.temperature_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(episodes, np.array(agent.temperature_values_mean) - np.array(agent.temperature_values_std), np.array(agent.temperature_values_mean) + np.array(agent.temperature_values_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("Temperature", fontsize = 20)
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()

    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "temperature_losses.png", bbox_inches='tight')
    plt.close()

    # Q1 and Q2 values
    plt.figure(figsize=(10, 5))
    plt.suptitle('Q1 and Q2 Values', fontsize = 16)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.4, hspace=0.6)
    ax1 = plt.subplot(gs[0])
    ax1.plot(episodes, agent.q1_values_mean, label="Q1", linewidth = 2, color = 'blue')
    ax1.fill_between(episodes, agent.q1_values_min, agent.q1_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(episodes, np.array(agent.q1_values_mean) - np.array(agent.q1_values_std), np.array(agent.q1_values_mean) + np.array(agent.q1_values_std),  
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Q1", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()
    
    ax2 = plt.subplot(gs[1])
    ax2.plot(episodes, agent.q2_values_mean, label="Q2", linewidth = 2, color = 'blue')
    ax2.fill_between(episodes, agent.q2_values_min, agent.q2_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(episodes, np.array(agent.q2_values_mean) - np.array(agent.q2_values_std), np.array(agent.q2_values_mean) + np.array(agent.q2_values_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("Q2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()
    
    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "q1_and_q2_values.png", bbox_inches='tight')
    plt.close()
    
    # Add Next Q1 and Next Q2 values plot
    plt.figure(figsize=(10, 5))
    plt.suptitle('Next Q1 and Next Q2 Values', fontsize = 16)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.4, hspace=0.6)
    ax1 = plt.subplot(gs[0])
    ax1.plot(episodes, agent.next_q1_values_mean, label="Next Q1", linewidth = 2, color = 'blue')
    ax1.fill_between(episodes, agent.next_q1_values_min, agent.next_q1_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(episodes, np.array(agent.next_q1_values_mean) - np.array(agent.next_q1_values_std), np.array(agent.next_q1_values_mean) + np.array(agent.next_q1_values_std),  
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Next Q1", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()
    
    ax2 = plt.subplot(gs[1])
    ax2.plot(episodes, agent.next_q2_values_mean, label="Next Q2", linewidth = 2, color = 'blue')
    ax2.fill_between(episodes, agent.next_q2_values_min, agent.next_q2_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(episodes, np.array(agent.next_q2_values_mean) - np.array(agent.next_q2_values_std), np.array(agent.next_q2_values_mean) + np.array(agent.next_q2_values_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("Next Q2", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()
    
    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)

    plt.savefig(save_path + "next_q1_and_q2_values.png", bbox_inches='tight')
    plt.close()
    
    # Combined Q and Next Q values plot (4 columns)
    plt.figure(figsize=(20, 5))
    plt.suptitle('Q Values and Next Q Values', fontsize = 16)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.4, hspace=0.6)
    
    # Q1
    ax1 = plt.subplot(gs[0])
    ax1.plot(episodes, agent.q1_values_mean, label="Q1", linewidth = 2, color = 'blue')
    ax1.fill_between(episodes, agent.q1_values_min, agent.q1_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(episodes, np.array(agent.q1_values_mean) - np.array(agent.q1_values_std), np.array(agent.q1_values_mean) + np.array(agent.q1_values_std),  
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Episode", fontsize = 16)
    ax1.set_ylabel("Q1", fontsize = 16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid()
    
    # Q2
    ax2 = plt.subplot(gs[1])
    ax2.plot(episodes, agent.q2_values_mean, label="Q2", linewidth = 2, color = 'blue')
    ax2.fill_between(episodes, agent.q2_values_min, agent.q2_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax2.fill_between(episodes, np.array(agent.q2_values_mean) - np.array(agent.q2_values_std), np.array(agent.q2_values_mean) + np.array(agent.q2_values_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Episode", fontsize = 16)
    ax2.set_ylabel("Q2", fontsize = 16)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.grid()
    
    # Next Q1
    ax3 = plt.subplot(gs[2])
    ax3.plot(episodes, agent.next_q1_values_mean, label="Next Q1", linewidth = 2, color = 'red')
    ax3.fill_between(episodes, agent.next_q1_values_min, agent.next_q1_values_max,
                facecolor='C3', alpha=0.20,
                label='min-max')
    ax3.fill_between(episodes, np.array(agent.next_q1_values_mean) - np.array(agent.next_q1_values_std), np.array(agent.next_q1_values_mean) + np.array(agent.next_q1_values_std),  
                facecolor='C3', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax3.set_xlabel("Episode", fontsize = 16)
    ax3.set_ylabel("Next Q1", fontsize = 16)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    ax3.grid()
    
    # Next Q2
    ax4 = plt.subplot(gs[3])
    ax4.plot(episodes, agent.next_q2_values_mean, label="Next Q2", linewidth = 2, color = 'red')
    ax4.fill_between(episodes, agent.next_q2_values_min, agent.next_q2_values_max,
                facecolor='C3', alpha=0.20,
                label='min-max')
    ax4.fill_between(episodes, np.array(agent.next_q2_values_mean) - np.array(agent.next_q2_values_std), np.array(agent.next_q2_values_mean) + np.array(agent.next_q2_values_std),
                facecolor='C3', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax4.set_xlabel("Episode", fontsize = 16)
    ax4.set_ylabel("Next Q2", fontsize = 16)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    ax4.grid()
    
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.05, 0.5), fontsize=14)

    plt.savefig(save_path + "combined_q_values.png", bbox_inches='tight')
    plt.close()
    
    # Plot for abs(Q1-Q2) values
    plt.figure(figsize=(10, 5))
    plt.suptitle('Absolute Difference between Q1 and Q2 Values', fontsize = 16)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.4, hspace=0.6)
    
    # |Q1-Q2|
    ax1 = plt.subplot(gs[0])
    ax1.plot(episodes, agent.q_diff_values_mean, label="|Q1-Q2|", linewidth = 2, color = 'blue')
    ax1.fill_between(episodes, agent.q_diff_values_min, agent.q_diff_values_max,
                facecolor='C0', alpha=0.20,
                label='min-max')
    ax1.fill_between(episodes, np.array(agent.q_diff_values_mean) - np.array(agent.q_diff_values_std), 
                     np.array(agent.q_diff_values_mean) + np.array(agent.q_diff_values_std),
                facecolor='C0', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("|Q1-Q2|", fontsize = 20)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.grid()
    
    # |next_Q1-next_Q2|
    ax2 = plt.subplot(gs[1])
    ax2.plot(episodes, agent.next_q_diff_values_mean, label="|next_Q1-next_Q2|", linewidth = 2, color = 'red')
    ax2.fill_between(episodes, agent.next_q_diff_values_min, agent.next_q_diff_values_max,
                facecolor='C3', alpha=0.20,
                label='min-max')
    ax2.fill_between(episodes, np.array(agent.next_q_diff_values_mean) - np.array(agent.next_q_diff_values_std), 
                     np.array(agent.next_q_diff_values_mean) + np.array(agent.next_q_diff_values_std),
                facecolor='C3', alpha=0.45,
                label=r'$\pm 1 \sigma$')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("|next_Q1-next_Q2|", fontsize = 20)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()
    
    handles, labels = ax1.get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=20)
    
    plt.savefig(save_path + "q_diff_values.png", bbox_inches='tight')
    plt.close()
    
    # Combined plot with both differences
    plt.figure(figsize=(12, 6))
    plt.suptitle('Q-Value Differences', fontsize = 18)
    
    plt.plot(episodes, agent.q_diff_values_mean, label="|Q1-Q2|", linewidth = 2, color = 'blue')
    plt.fill_between(episodes, agent.q_diff_values_min, agent.q_diff_values_max,
                facecolor='C0', alpha=0.15,
                label='min-max (current)')
    plt.fill_between(episodes, np.array(agent.q_diff_values_mean) - np.array(agent.q_diff_values_std), 
                     np.array(agent.q_diff_values_mean) + np.array(agent.q_diff_values_std),
                facecolor='C0', alpha=0.3,
                label=r'$\pm 1 \sigma$ (current)')
    
    plt.plot(episodes, agent.next_q_diff_values_mean, label="|next_Q1-next_Q2|", linewidth = 2, color = 'red')
    plt.fill_between(episodes, agent.next_q_diff_values_min, agent.next_q_diff_values_max,
                facecolor='C3', alpha=0.15,
                label='min-max (next)')
    plt.fill_between(episodes, np.array(agent.next_q_diff_values_mean) - np.array(agent.next_q_diff_values_std), 
                     np.array(agent.next_q_diff_values_mean) + np.array(agent.next_q_diff_values_std),
                facecolor='C3', alpha=0.3,
                label=r'$\pm 1 \sigma$ (next)')
    
    plt.xlabel("Episode", fontsize = 20)
    plt.ylabel("Absolute Difference", fontsize = 20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.grid(True)
    plt.legend(fontsize=14, loc='upper right')
    
    plt.savefig(save_path + "combined_q_diff_values.png", bbox_inches='tight')
    plt.close()
    

    
    
def agent_plotter_td3(agent):
    save_path = agent.save_path

    # Plot critic and actor losses
    plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.3)
    plt.suptitle('Reinforcement Learning', fontsize = 32)
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(agent.critic_losses, label="Critic Loss", linewidth = 4, color = 'blue')
    ax1.plot(agent.critic_mse_losses, label="Critic MSE Loss", linewidth = 4, color = 'red')
    ax1.plot(agent.critic_l2_regs, label="Critic L2 Reg", linewidth = 4, color = 'green')
    ax1.set_xlabel("Episode", fontsize = 20)
    ax1.set_ylabel("Loss", fontsize = 20)
    ax1.set_title("Critic Loss", fontsize = 22)
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.legend(fontsize=20)
    ax1.grid()

    ax2 = plt.subplot(gs[1])
    ax2.plot(agent.actor_losses, label="Actor Loss", linewidth = 4, color = 'blue')
    ax2.set_xlabel("Episode", fontsize = 20)
    ax2.set_ylabel("Loss", fontsize = 20)
    ax2.set_title("Actor Loss", fontsize = 22)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    ax2.grid()
    
    plt.savefig(save_path + "td3_losses.png")
    plt.close()

    # Plot number of steps
    plt.figure(figsize=(10, 5))
    plt.plot(agent.number_of_steps, label="Steps", linewidth = 4, color = 'blue')
    plt.xlabel("Episode", fontsize=20)
    plt.ylabel("Steps", fontsize=20)
    plt.title("Number of Steps", fontsize=22)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    plt.savefig(save_path + "number_of_steps.png")
    plt.close()