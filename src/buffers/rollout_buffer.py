"""Rollout buffer for on-policy algorithms (A2C)."""

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from .base import BaseBuffer


class RolloutBuffer(BaseBuffer):
    """Buffer for storing episode rollouts (on-policy).

    Stores: observations, actions, rewards, next_observations, dones

    """

    def __init__(
        self,
        n_env: int,
        max_episode_length: int,
        observation_dim: int,
        action_dim: int,
        store_log_prob: bool = False,
    ):
        """Initialize rollout buffer.

        Args:
            n_env: Number of parallel environments
            max_episode_length: Maximum episode length
            observation_dim: Observation dimension
            action_dim: Action dimension
            store_log_prob: Whether to store log probabilities
        """
        self.n_env = n_env
        self.max_episode_length = max_episode_length
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.store_log_prob = store_log_prob

        # Pre-allocate JAX arrays
        self.observations = jnp.zeros((n_env, max_episode_length, observation_dim))
        self.actions = jnp.zeros((n_env, max_episode_length, action_dim))
        self.rewards = jnp.zeros((n_env, max_episode_length))
        self.next_observations = jnp.zeros((n_env, max_episode_length, observation_dim))
        self.dones = jnp.zeros((n_env, max_episode_length))
        if store_log_prob:
            self.log_probs = jnp.zeros((n_env, max_episode_length))

        self.position = 0

    def __len__(self) -> int:
        """Return current number of transitions."""
        return self.position * self.n_env

    def add(
        self,
        observation: Float[Array, "n_env ..."],
        action: Float[Array, "n_env action_dim"],
        reward: Float[Array, " n_env"],
        next_observation: Float[Array, "n_env ..."],
        done: Bool[Array, " n_env"],
        log_prob: Float[Array, " n_env"] | None = None,
    ):
        """Add a transition to the buffer.

        Args:
            observation: Observation
            action: Action taken
            reward: Reward received
            next_observation: Next observation
            done: Whether episode is done
            log_prob: Log probability of the action
        """
        self.observations = self.observations.at[:, self.position].set(observation)
        self.actions = self.actions.at[:, self.position].set(action)
        self.rewards = self.rewards.at[:, self.position].set(reward)
        self.next_observations = self.next_observations.at[:, self.position].set(
            next_observation
        )
        self.dones = self.dones.at[:, self.position].set(done)
        if self.store_log_prob and log_prob is not None:
            self.log_probs = self.log_probs.at[:, self.position].set(log_prob)

        self.position += 1

    def get_batch(self, batch_size: int) -> dict[str, jnp.ndarray]:
        """Get all stored transitions as a batch.

        Args:
            batch_size: Size of the batch to return

        Returns:
            Dictionary with observations, actions, rewards, next_observations, dones
        """
        batch = {
            "observations": self.observations[:, :batch_size],
            "actions": self.actions[:, :batch_size],
            "rewards": self.rewards[:, :batch_size],
            "next_observations": self.next_observations[:, :batch_size],
            "dones": self.dones[:, :batch_size],
        }
        if self.store_log_prob:
            batch["log_probs"] = self.log_probs[:, :batch_size]
        self.observations = jnp.roll(self.observations, -batch_size, axis=1)
        self.actions = jnp.roll(self.actions, -batch_size, axis=1)
        self.rewards = jnp.roll(self.rewards, -batch_size, axis=1)
        self.next_observations = jnp.roll(self.next_observations, -batch_size, axis=1)
        self.dones = jnp.roll(self.dones, -batch_size, axis=1)
        if self.store_log_prob:
            self.log_probs = jnp.roll(self.log_probs, -batch_size, axis=1)
        self.position -= batch_size
        return batch

    def reset(self):
        """Clear buffer for next episode."""
        self.position = 0
