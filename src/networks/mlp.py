"""Multi-layer perceptron Q-network using Flax."""
from typing import Callable, Sequence

import jax.numpy as jnp
from flax import linen as nn


class MLPQNetwork(nn.Module):
    """MLP-based Q-network for discrete action spaces.

    Attributes:
        hidden_dims: Sequence of hidden layer dimensions
        num_actions: Number of discrete actions
        activation: Activation function (e.g., nn.relu, nn.tanh, nn.elu)
    """

    hidden_dims: Sequence[int]
    action_dim: int
    activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the Q-network.

        Args:
            x: Input observation of shape (batch_size, observation_dim)

        Returns:
            Q-values for all actions, shape (batch_size, num_actions)
        """
        # Hidden layers
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)

        # Output layer: Q-values for each action
        q_values = nn.Dense(self.action_dim)(x)

        return q_values
