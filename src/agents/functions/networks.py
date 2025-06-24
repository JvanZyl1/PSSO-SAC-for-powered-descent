import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple, Dict, Any

### ACTOR ###
class GaussianActor(nn.Module):
    action_dim: int
    hidden_dim: int = 10
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(state)
        x = nn.relu(x)
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        mean = nn.tanh(nn.Dense(self.action_dim)(x))
        # initial bias of 0.5/number_of_hidden_layers
        # initial weights are 0.001/number_of_hidden_layers
        std = nn.sigmoid(nn.Dense(self.action_dim, 
                                  kernel_init=lambda *_: jnp.ones((self.hidden_dim, self.action_dim)) * 0.001/self.number_of_hidden_layers,
                                  bias_init=lambda *_: jnp.ones((self.action_dim,)) * 0.001/self.number_of_hidden_layers)(x))
        return mean, std

class ClassicalActor(nn.Module):
    action_dim: int
    hidden_dim: int = 10
    number_of_hidden_layers: int = 3
    @nn.compact
    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(state)
        x = nn.relu(x)
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        return nn.tanh(nn.Dense(self.action_dim)(x))

### CRITIC ###
class Critic(nn.Module):
    hidden_dim: int
    number_of_hidden_layers: int

    @nn.compact
    def __call__(self, sa: jnp.ndarray) -> jnp.ndarray:
        x = sa
        for _ in range(self.number_of_hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.xavier_uniform())(x)
            x = nn.relu(x)
        q = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())(x)
        return jnp.squeeze(q, axis=-1)

class DoubleCritic(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    number_of_hidden_layers: int = 3

    def setup(self):
        self.q1_net = Critic(self.hidden_dim, self.number_of_hidden_layers)
        self.q2_net = Critic(self.hidden_dim, self.number_of_hidden_layers)

    def __call__(self, state: jnp.ndarray, action: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        sa = jnp.concatenate([state, action], axis=-1)
        
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)

        if len(state.shape) > 1:  # For batches
            q1 = jnp.reshape(q1, (-1, 1))
            q2 = jnp.reshape(q2, (-1, 1))
        
        return q1, q2

