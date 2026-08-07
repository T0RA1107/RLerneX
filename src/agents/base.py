"""Base agent abstract class for RL algorithms."""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    @abstractmethod
    def select_action(
        self,
        observation: Float[Array, "n_env ..."],
        rng: jax.random.PRNGKey,
        training: bool = True,
    ) -> Float[Array, "n_env action_dim"]:
        """Select an action given an observation.

        Args:
            observation: Current observation
            rng: JAX random number generator key
            training: Whether in training mode (for exploration)

        Returns:
            Tuple of (action, info_dict, updated_rng)
            - action: int for discrete, jnp.ndarray for continuous
            - info_dict: Optional additional info (e.g., log_prob, value)
            - updated_rng: Updated RNG key
        """
        pass

    @abstractmethod
    def update(self, batch: dict[str, jnp.ndarray]) -> dict[str, float]:
        """Update agent parameters using a batch of transitions.

        Args:
            batch: Dictionary containing transitions with keys:
                - observations: (batch_size, observation_dim)
                - actions: (batch_size,)
                - rewards: (batch_size,)
                - next_observations: (batch_size, observation_dim)
                - dones: (batch_size,)

        Returns:
            Dictionary of metrics (e.g., loss, q_values)
        """
        pass

    @abstractmethod
    def check_action_type(self, action_type: str) -> None:
        """Check if the action type is compatible with the agent.

        Args:
            action_type: Type of action space (e.g., 'discrete', 'continuous')
        """
        pass

    @property
    @abstractmethod
    def isonpolicy(self) -> bool:
        """Return whether the agent is on-policy.

        Returns:
            True if on-policy, False if off-policy
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent parameters to file.

        Args:
            path: File path to save parameters
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent parameters from file.

        Args:
            path: File path to load parameters from
        """
        pass

    def compute_log_prob(
        self,
        observation: Float[Array, "n_env ..."],
        action: Float[Array, "n_env action_dim"],
    ) -> Float[Array, " n_env"] | None:
        """Compute log probability of action under current policy.

        Args:
            observation: Current observation
            action: Action taken

        Returns:
            Log probability per environment, or None if not supported
        """
        return None
