"""Actor-Critic networks for continuous action spaces."""
from typing import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp


class GaussianActor(nn.Module):
    """Actor network outputting Gaussian policy for continuous actions.

    Attributes:
        hidden_dims: Hidden layer dimensions
        action_dim: Continuous action dimension
        activation: Activation function (e.g., nn.relu, nn.tanh, nn.elu)
        log_std_min: Minimum log std for stability
        log_std_max: Maximum log std for stability
    """
    hidden_dims: Sequence[int]
    action_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass.

        Args:
            x: Observations (batch_size, obs_dim)

        Returns:
            Tuple of (mean, log_std), each (batch_size, action_dim)
        """
        # Shared hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)

        # Mean head
        mean = nn.Dense(self.action_dim)(x)

        log_std = self.param("log_std", nn.initializers.constant(0.1), (self.action_dim,))
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        log_std = jnp.broadcast_to(log_std, mean.shape)

        return mean, log_std


class ValueCritic(nn.Module):
    """Critic network outputting state value V(s).

    Attributes:
        hidden_dims: Hidden layer dimensions
        activation: Activation function (e.g., nn.relu, nn.tanh, nn.elu)
    """
    hidden_dims: Sequence[int]
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Observations (batch_size, obs_dim)

        Returns:
            State values (batch_size,)
        """
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)

        # Output layer (scalar value)
        value = nn.Dense(1)(x)
        return jnp.squeeze(value, axis=-1)
