import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from functools import partial
from typing import Callable, Tuple

@jax.jit
def gaussian_likelihood(actions: jnp.ndarray,
                        mean: jnp.ndarray,
                        std: jnp.ndarray) -> jnp.ndarray:
    log_prob = -0.5 * (((actions - mean) ** 2) / (std ** 2) + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
    return log_prob.sum(axis=-1)

@jax.jit
def clip_grads(grads: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    clipped_grads = jax.tree_util.tree_map(lambda x: x * scale, grads)
    return clipped_grads

def calculate_td_error(states,
                       actions,
                       rewards,
                       next_states,
                       dones,
                       temperature: float,
                       gamma: float,
                       critic_params: jnp.ndarray,
                       critic_target_params: jnp.ndarray,
                       critic: nn.Module,
                       next_actions: jnp.ndarray,
                       next_log_policy: jnp.ndarray) -> jnp.ndarray:
    q1, q2 = critic.apply(critic_params, states, actions)
    next_q1, next_q2 = critic.apply(critic_target_params, next_states, next_actions)
    next_q_mean = jnp.minimum(next_q1, next_q2)
    entropy_term = jax.lax.stop_gradient(temperature) * jnp.expand_dims(jax.lax.stop_gradient(next_log_policy), axis=1)  
    td_target = rewards + gamma * (1 - dones) * (next_q_mean - entropy_term)
    td_errors = 0.5 * ((td_target - q1)**2 + (td_target - q2)**2)
    return td_errors

def critic_update(critic_optimiser,
                  calculate_td_error_fcn : Callable,
                  critic_params : jnp.ndarray,
                  critic_opt_state : jnp.ndarray,
                  critic_grad_max_norm : float,
                  buffer_weights : jnp.ndarray,
                  states : jnp.ndarray,
                  actions : jnp.ndarray,
                  rewards : jnp.ndarray,
                  next_states : jnp.ndarray,
                  dones : jnp.ndarray,
                  temperature : float,
                  critic_target_params : jnp.ndarray,
                  next_actions : jnp.ndarray,
                  next_log_policy : jnp.ndarray,
                  l2_reg_coef : float) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def loss_fcn(params):
        td_errors = calculate_td_error_fcn(states = jax.lax.stop_gradient(states),
                                           actions = jax.lax.stop_gradient(actions),
                                           rewards = jax.lax.stop_gradient(rewards),
                                           next_states = jax.lax.stop_gradient(next_states),
                                           dones = jax.lax.stop_gradient(dones),
                                           temperature = jax.lax.stop_gradient(temperature),
                                           critic_params = params,
                                           critic_target_params = jax.lax.stop_gradient(critic_target_params),
                                           next_actions = jax.lax.stop_gradient(next_actions),
                                           next_log_policy = jax.lax.stop_gradient(next_log_policy))
        weighted_td_error_loss = jnp.mean(jax.lax.stop_gradient(buffer_weights) * td_errors)
        # L2 regularisation
        l2_reg = 0.0
        for param in jax.tree_util.tree_leaves(params):
            l2_reg += jnp.sum(param**2)
        
        l2_reg_loss = l2_reg * l2_reg_coef
        loss = weighted_td_error_loss + l2_reg_loss
        return loss, (td_errors, weighted_td_error_loss, l2_reg_loss)

    grads, (_,_,_)= jax.grad(loss_fcn, has_aux=True)(critic_params)
    clipped_grads = clip_grads(grads, max_norm=critic_grad_max_norm)
    updates, critic_opt_state = critic_optimiser.update(clipped_grads, critic_opt_state, critic_params)
    critic_params = optax.apply_updates(critic_params, updates)
    critic_loss, (td_errors, weighted_td_error_loss, l2_reg_loss) = loss_fcn(critic_params)
    return critic_params, critic_opt_state, critic_loss, td_errors, weighted_td_error_loss, l2_reg_loss

def actor_update(actor_optimiser,
                 actor : nn.Module,
                 critic : nn.Module,
                 actor_grad_max_norm : float,
                 temperature : float,
                 states : jnp.ndarray,
                 normal_distribution : jnp.ndarray,
                 critic_params : jnp.ndarray,
                 actor_params : jnp.ndarray,
                 actor_opt_state : jnp.ndarray,
                 max_std : float):
    def loss_fcn(params):
        action_mean, action_std = actor.apply(params, states)
        noise   = jnp.clip(normal_distribution, -max_std, max_std)
        actions = jnp.clip(noise * action_std + action_mean, -1, 1)
        action_std = jnp.maximum(action_std, 1e-6) # avoid crazy log probabilities.
        q1, q2 = critic.apply(jax.lax.stop_gradient(critic_params), jax.lax.stop_gradient(states), actions)
        q_min = jnp.minimum(q1, q2)
        log_probability = gaussian_likelihood(normal_distribution * action_std + action_mean, action_mean, action_std)
        squash_corr = jnp.sum(jnp.log1p(-actions**2 + 1e-6), axis=-1)
        log_probability = log_probability - squash_corr 
        entropy_loss = (jax.lax.stop_gradient(temperature) * log_probability).mean()
        q_loss = (-q_min).mean()
        return (jax.lax.stop_gradient(temperature) * log_probability - q_min).mean(), (log_probability, action_std, action_mean, entropy_loss, q_loss,q1,q2)
    grads, aux_values = jax.grad(loss_fcn, has_aux=True)(actor_params)
    clipped_grads = clip_grads(grads, max_norm=actor_grad_max_norm)
    updates, actor_opt_state = actor_optimiser.update(clipped_grads, actor_opt_state, actor_params)
    actor_params = optax.apply_updates(actor_params, updates)
    actor_loss, (current_log_probabilities, action_std, action_mean, entropy_loss, q_loss,q1,q2) = loss_fcn(actor_params)
    return actor_params, actor_opt_state, actor_loss, current_log_probabilities, action_std, action_mean, entropy_loss, q_loss,q1,q2

def temperature_update(temperature_optimiser,
                       temperature_grad_max_norm: float,
                       current_log_probabilities: jnp.ndarray,
                       target_entropy: float,
                       temperature_opt_state: jnp.ndarray,
                       temperature: float) -> Tuple[jnp.ndarray, optax.OptState, jnp.ndarray]:
    log_alpha = jnp.log(temperature)
    def loss_fn(log_alpha):
        diff = jax.lax.stop_gradient(current_log_probabilities) + jax.lax.stop_gradient(target_entropy)
        return (-log_alpha * diff).mean()

    grads = jax.grad(loss_fn)(log_alpha)
    grads = clip_grads(grads, max_norm=temperature_grad_max_norm)
    updates, temperature_opt_state = temperature_optimiser.update(grads, temperature_opt_state, log_alpha)
    log_alpha = optax.apply_updates(log_alpha, updates)
    temperature = jnp.exp(log_alpha)
    temperature_loss = loss_fn(log_alpha)
    return temperature, temperature_opt_state, temperature_loss

def update_sac(actor : nn.Module,
               critic : nn.Module,
               actor_params: jnp.ndarray,
               actor_opt_state: jnp.ndarray,
               normal_distribution_for_next_actions: jnp.ndarray,
               normal_distribution_for_actions: jnp.ndarray,
               states: jnp.ndarray,
               actions: jnp.ndarray,
               rewards: jnp.ndarray,
               next_states: jnp.ndarray,
               dones: jnp.ndarray,
               buffer_weights: jnp.ndarray,
               temperature: float,
               temperature_opt_state: jnp.ndarray,
               critic_params: jnp.ndarray,
               critic_target_params: jnp.ndarray,
               critic_opt_state: jnp.ndarray,
               critic_update_lambda: Callable,
               actor_update_lambda: Callable,
               temperature_update_lambda: Callable,
               tau: float,
               max_std: float,
               first_step_bool: bool):
    # 0. Sample next actions
    next_action_mean, next_action_std = actor.apply(actor_params, next_states)
    noise   = jnp.clip(normal_distribution_for_next_actions, -max_std, max_std)
    next_actions = jnp.clip(noise * next_action_std + next_action_mean, -1, 1)

    # 1. Find next actions log probabilities.
    next_log_probabilities = gaussian_likelihood(noise * next_action_std + next_action_mean, next_action_mean, next_action_std)
    squash_corr = jnp.sum(jnp.log1p(-next_actions**2 + 1e-6), axis=-1)
    next_log_probabilities = next_log_probabilities - squash_corr 

    # 2. Update the critic.
    critic_params, critic_opt_state, critic_loss, td_errors, weighted_td_error_loss, l2_reg = critic_update_lambda(critic_params = critic_params,
                                                                                   critic_opt_state = critic_opt_state,
                                                                                   buffer_weights = buffer_weights,
                                                                                   states = states,
                                                                                   actions = actions,
                                                                                   rewards = rewards,
                                                                                   next_states = next_states,
                                                                                   dones = dones,
                                                                                   temperature = temperature,
                                                                                   critic_target_params = critic_target_params,
                                                                                   next_actions = next_actions,
                                                                                   next_log_policy = next_log_probabilities)
    next_q1, next_q2 = critic.apply(critic_target_params, next_states, next_actions)
    critic_loss = jax.lax.stop_gradient(critic_loss)
    td_errors = jax.lax.stop_gradient(td_errors)

    # 2. Update the actor.
    actor_params, actor_opt_state, actor_loss, current_log_probabilities, action_std, action_mean, actor_entropy_loss, actor_q_loss,q1,q2 = actor_update_lambda(temperature = temperature,
                                                                                                           states = states,
                                                                                                           normal_distribution = normal_distribution_for_actions,
                                                                                                           critic_params = critic_params,
                                                                                                           actor_params = actor_params,
                                                                                                           actor_opt_state = actor_opt_state)
    actor_loss = jax.lax.stop_gradient(actor_loss)

    # 3. Update the temperature.
    temperature, temperature_opt_state, temperature_loss = jax.lax.cond(
        first_step_bool,
        lambda : (temperature, temperature_opt_state, 0.0),
        lambda : temperature_update_lambda(current_log_probabilities = current_log_probabilities,
                                          temperature_opt_state = temperature_opt_state,
                                          temperature = temperature)
    )
    temperature_loss = jax.lax.stop_gradient(temperature_loss)

    # 4. Update the target critic.
    critic_target_params = jax.tree_util.tree_map(lambda p, tp: tau * p + (1.0 - tau) * tp, critic_params, critic_target_params)

    # 5. Return values.
    return critic_params, critic_opt_state, critic_loss, td_errors, \
            actor_params, actor_opt_state, actor_loss, actor_entropy_loss, actor_q_loss,q1,q2, next_q1, next_q2, \
            temperature, temperature_opt_state, temperature_loss, \
            critic_target_params, \
            current_log_probabilities, action_std, action_mean, \
            weighted_td_error_loss, l2_reg


def critic_warm_up_update(actor : nn.Module,
                          actor_params: jnp.ndarray,
                          normal_distribution_for_next_actions: jnp.ndarray,
                          states: jnp.ndarray,
                          actions: jnp.ndarray,
                          rewards: jnp.ndarray,
                          next_states: jnp.ndarray,
                          dones: jnp.ndarray,
                          initial_temperature: float,
                          critic_params: jnp.ndarray,
                          critic_target_params: jnp.ndarray,
                          critic_opt_state: jnp.ndarray,
                          critic_update_lambda: Callable,
                          tau: float,
                          max_std: float):
    # 0. Sample next actions
    next_action_mean, next_action_std = actor.apply(actor_params, next_states)
    noise   = jnp.clip(normal_distribution_for_next_actions, -max_std, max_std)
    next_actions = jnp.clip(noise * next_action_std + next_action_mean, -1, 1)

    # 1. Find next actions log probabilities.
    next_log_probabilities = gaussian_likelihood(noise * next_action_std + next_action_mean, next_action_mean, next_action_std)
    squash_corr = jnp.sum(jnp.log1p(-next_actions**2 + 1e-6), axis=-1)
    next_log_probabilities = next_log_probabilities - squash_corr 

    # 2. Update the critic.
    buffer_weights_ones = jnp.ones_like(rewards)
    critic_params, critic_opt_state, critic_loss, td_errors, weighted_td_error_loss, l2_reg = critic_update_lambda(critic_params = critic_params,
                                                                                   critic_opt_state = critic_opt_state,
                                                                                   buffer_weights = buffer_weights_ones,
                                                                                   states = states,
                                                                                   actions = actions,
                                                                                   rewards = rewards,
                                                                                   next_states = next_states,
                                                                                   dones = dones,
                                                                                   temperature = initial_temperature,
                                                                                   critic_target_params = critic_target_params,
                                                                                   next_actions = next_actions,
                                                                                   next_log_policy = next_log_probabilities)
    critic_loss = jax.lax.stop_gradient(critic_loss)
    td_errors = jax.lax.stop_gradient(td_errors)

    # 4. Update the target critic.
    critic_target_params = jax.tree_util.tree_map(lambda p, tp: tau * p + (1.0 - tau) * tp, critic_params, critic_target_params)

    # 5. Return values.
    return critic_params, critic_opt_state, critic_target_params, critic_loss, \
            weighted_td_error_loss, l2_reg

def lambda_compile_sac(critic_optimiser,
                       critic: nn.Module,
                       critic_grad_max_norm: float,
                       actor_optimiser,
                       actor: nn.Module,
                       actor_grad_max_norm: float,
                       temperature_optimiser,
                       temperature_grad_max_norm: float,
                       gamma: float,
                       tau: float,
                       target_entropy: float,
                       initial_temperature: float,
                       max_std: float,
                       l2_reg_coef: float):
    calculate_td_error_lambda = jax.jit(
        partial(calculate_td_error,
                critic = critic,
                gamma = gamma),
        static_argnames = ['critic', 'gamma']
    )

    critic_update_lambda = jax.jit(
        partial(critic_update,
                critic_optimiser = critic_optimiser,
                calculate_td_error_fcn = calculate_td_error_lambda,
                critic_grad_max_norm = critic_grad_max_norm,
                l2_reg_coef = l2_reg_coef),
        static_argnames = ['critic_optimiser', 'calculate_td_error_fcn', 'critic_grad_max_norm', 'l2_reg_coef']
    )

    actor_update_lambda = jax.jit(
          partial(actor_update,
                  actor_optimiser = actor_optimiser,
                  actor = actor,
                  critic = critic,
                  actor_grad_max_norm = actor_grad_max_norm,
                  max_std = max_std),
          static_argnames = ['actor_optimiser', 'actor', 'critic', 'actor_grad_max_norm', 'max_std']
    )

    temperature_update_lambda = jax.jit(
        partial(temperature_update,
                temperature_optimiser = temperature_optimiser,
                temperature_grad_max_norm = temperature_grad_max_norm,
                target_entropy = target_entropy),
        static_argnames = ['temperature_optimiser', 'temperature_grad_max_norm', 'target_entropy']
    )

    update_sac_lambda = jax.jit(
        partial(update_sac,
                actor = actor,
                critic = critic,
                critic_update_lambda = critic_update_lambda,
                actor_update_lambda = actor_update_lambda,
                temperature_update_lambda = temperature_update_lambda,
                tau = tau,
                max_std = max_std),
        static_argnames = ['critic_update_lambda', 'actor_update_lambda', 'temperature_update_lambda', 'tau', 'max_std', 'critic', 'actor']
    )

    critic_warm_up_update_lambda = jax.jit(
        partial(critic_warm_up_update,
                actor = actor,
                initial_temperature = initial_temperature,
                critic_update_lambda = critic_update_lambda,
                tau = tau,
                max_std = max_std),
        static_argnames = ['actor', 'initial_temperature', 'critic_update_lambda', 'tau', 'max_std']
    )

    return update_sac_lambda, calculate_td_error_lambda, critic_warm_up_update_lambda