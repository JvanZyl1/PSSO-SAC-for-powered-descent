import pickle
from src.agents.soft_actor_critic import SoftActorCritic
from src.agents.td3 import TD3
from torch.utils.tensorboard import SummaryWriter

def load_sac(agent_path : str):
    with open(agent_path, 'rb') as f:
        agent_state = pickle.load(f)
    sac = SoftActorCritic(**agent_state['inputs'])

    sac.rng_key = agent_state['misc']['rng_key']
    sac.run_id = agent_state['misc']['run_id']
    sac.writer = SummaryWriter(log_dir=f'data/agent_saves/VanillaSAC/{sac.flight_phase}/runs/{sac.run_id}')


    # Update logging all
    sac.critic_loss_episode = agent_state['logging']['critic_loss_episode']
    sac.actor_loss_episode = agent_state['logging']['actor_loss_episode']
    sac.temperature_loss_episode = agent_state['logging']['temperature_loss_episode']
    sac.td_errors_episode = agent_state['logging']['td_errors_episode']
    sac.temperature_values_all_episode = agent_state['logging']['temperature_values_all_episode']
    sac.number_of_steps_episode = agent_state['logging']['number_of_steps_episode']
    sac.critic_losses = agent_state['logging']['critic_losses']
    sac.actor_losses = agent_state['logging']['actor_losses']
    sac.temperature_losses = agent_state['logging']['temperature_losses']
    sac.td_errors = agent_state['logging']['td_errors']
    sac.temperature_values = agent_state['logging']['temperature_values']
    sac.number_of_steps = agent_state['logging']['number_of_steps']
    sac.episode_idx = agent_state['logging']['episode_idx']
    sac.step_idx = agent_state['logging']['step_idx']

    # Update update all
    sac.critic_params = agent_state['update']['critic_params']
    sac.critic_opt_state = agent_state['update']['critic_opt_state']
    sac.critic_target_params = agent_state['update']['critic_target_params']
    sac.actor_params = agent_state['update']['actor_params']
    sac.actor_opt_state = agent_state['update']['actor_opt_state']
    sac.temperature = agent_state['update']['temperature']
    sac.temperature_opt_state = agent_state['update']['temperature_opt_state']
    sac.buffer = agent_state['update']['buffer']  
    
    return sac

def load_td3(agent_path : str):
    with open(agent_path, 'rb') as f:
        agent_state = pickle.load(f)
    td3 = TD3(**agent_state['inputs'])

    td3.rng_key = agent_state['misc']['rng_key']
    td3.run_id = agent_state['misc']['run_id']
    td3.writer = SummaryWriter(log_dir=f'data/agent_saves/TD3/{td3.flight_phase}/runs/{td3.run_id}')

    # Update logging all
    td3.critic_loss_episode = agent_state['logging']['critic_loss_episode']
    td3.actor_loss_episode = agent_state['logging']['actor_loss_episode']
    td3.td_errors_episode = agent_state['logging']['td_errors_episode']
    td3.number_of_steps_episode = agent_state['logging']['number_of_steps_episode']
    td3.critic_losses = agent_state['logging']['critic_losses']
    td3.actor_losses = agent_state['logging']['actor_losses']
    td3.td_errors = agent_state['logging']['td_errors']
    td3.number_of_steps = agent_state['logging']['number_of_steps']
    td3.episode_idx = agent_state['logging']['episode_idx']
    td3.step_idx = agent_state['logging']['step_idx']

    td3.critic_params = agent_state['update']['critic_params']
    td3.critic_opt_state = agent_state['update']['critic_opt_state']
    td3.critic_target_params = agent_state['update']['critic_target_params']
    td3.actor_params = agent_state['update']['actor_params']
    td3.actor_opt_state = agent_state['update']['actor_opt_state']
    td3.buffer = agent_state['update']['buffer']
    return td3