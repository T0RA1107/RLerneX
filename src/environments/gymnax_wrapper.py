"""Gymnax environment wrapper for consistent interface."""
from pathlib import Path
from typing import Any, Tuple, Union

import gymnax
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.visualize import Visualizer


class GymnaxWrapper:
    """Wrapper for Gymnax environments providing a consistent interface.

    Supports both discrete and continuous action spaces.
    """

    def __init__(self, env_name: str, env_kwargs: dict = None):
        """Initialize the Gymnax environment wrapper.

        Args:
            env_name: Name of the Gymnax environment (e.g., "CartPole-v1")
            env_kwargs: Additional keyword arguments for environment creation
        """
        self.env_name = env_name
        self.env_kwargs = env_kwargs or {}

        # Create the Gymnax environment
        self.env, self.env_params = gymnax.make(env_name, **self.env_kwargs)

        # Vectorized methods (vmap over batch dimension)
        self._vmap_reset = jax.vmap(
            lambda rng: self.env.reset(rng, self.env_params)
        )
        self._vmap_step = jax.vmap(
            lambda state, action, rng: self.env.step(rng, state, action, self.env_params)
        )

        # Rendering state
        self._recording = False
        self._states: list = []

    def reset(self, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, Any]:
        """Reset the environment (vectorized).

        Args:
            rng: JAX random keys with shape (n_envs,) or (n_envs, 2)

        Returns:
            Tuple of (observations, states) with batch dimension
        """
        obs, state = self._vmap_reset(rng)
        return obs, state

    def step(
        self,
        state: Any,
        action: Union[int, jnp.ndarray],
        rng: jax.random.PRNGKey
    ) -> Tuple[jnp.ndarray, Any, float, bool, dict]:
        """Execute one step (vectorized).

        Args:
            state: Batched environment states
            action: Batched actions
            rng: Batched random keys

        Returns:
            Tuple of batched (next_obs, next_state, reward, done, info)
        """
        if self._recording:
            self._states.append(state)

        next_obs, next_state, reward, done, info = self._vmap_step(state, action, rng)
        return next_obs, next_state, reward, done, info

    def start_recording(self) -> None:
        """Start recording environment states for video generation."""
        self._recording = True
        self._states = []

    def stop_recording(self) -> None:
        """Stop recording environment states."""
        self._recording = False

    def save_video(self, save_path: str | Path) -> None:
        """Save recorded states as an animation to the given path.

        The actual file format is determined by the provided filename/extension
        and the underlying Gymnax ``Visualizer.animate`` implementation (e.g.,
        in current usage this is typically a GIF).

        Args:
            save_path: Path where the animation file will be saved.
        """
        if not self._states:
            return

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        states = []
        for state in self._states:
            first_state  = jax.tree_util.tree_map(lambda x: x[0], state)
            states.append(first_state)

        vis = Visualizer(self.env, self.env_params, states)
        vis.animate(str(save_path))

        # Clear buffer and stop recording
        self._states = []
        self._recording = False

    def get_observation_space(self) -> int:
        """Get the observation space dimension.

        Returns:
            Dimension of observation space
        """
        return self.env.observation_space(self.env_params).shape[0]

    def get_action_space(self) -> int:
        """Get the action space dimension.

        Returns:
            Number of discrete actions
        """
        return self.env.action_space(self.env_params).n

    @property
    def observation_dim(self) -> int:
        """Observation dimension."""
        return self.get_observation_space()

    @property
    def is_continuous_action(self) -> bool:
        """Check if action space is continuous."""
        return isinstance(self.env.action_space(self.env_params), gymnax.environments.spaces.Box)

    @property
    def action_dim(self) -> int:
        """Get action dimension (for continuous action spaces).

        Returns:
            Action dimension
        """
        if self.is_continuous_action:
            space = self.env.action_space(self.env_params)
            return int(np.prod(space.shape))
        else:
            return self.get_action_space()

    @property
    def action_low(self) -> jnp.ndarray:
        """Get lower bounds for continuous actions.

        Returns:
            Lower bounds as JAX array

        Raises:
            TypeError: If action space is not continuous
        """
        if self.is_continuous_action:
            low = jnp.array(self.env.action_space(self.env_params).low)
            # Ensure it's at least 1D
            if low.ndim == 0:
                low = jnp.array([low])
            return low
        raise TypeError("action_low is only for continuous action spaces")

    @property
    def action_high(self) -> jnp.ndarray:
        """Get upper bounds for continuous actions.

        Returns:
            Upper bounds as JAX array

        Raises:
            TypeError: If action space is not continuous
        """
        if self.is_continuous_action:
            high = jnp.array(self.env.action_space(self.env_params).high)
            # Ensure it's at least 1D
            if high.ndim == 0:
                high = jnp.array([high])
            return high
        raise TypeError("action_high is only for continuous action spaces")

    def scale_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """Scale action from [-1, 1] to environment's action bounds.

        Args:
            action: Action in range [-1, 1]

        Returns:
            Scaled action within environment's action bounds
        """
        if not self.is_continuous_action:
            raise TypeError("scale_action is only for continuous action spaces")

        low = self.action_low
        high = self.action_high
        scaled_action = low + (0.5 * (action + 1.0) * (high - low))
        return jnp.clip(scaled_action, low, high)

    def normalize_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """Normalize action from environment's action bounds to [-1, 1].

        Args:
            action: Action within environment's action bounds

        Returns:
            Normalized action in range [-1, 1]
        """
        if not self.is_continuous_action:
            raise TypeError("normalize_action is only for continuous action spaces")

        low = self.action_low
        high = self.action_high
        normalized_action = 2.0 * (action - low) / (high - low) - 1.0
        return jnp.clip(normalized_action, -1.0, 1.0)
