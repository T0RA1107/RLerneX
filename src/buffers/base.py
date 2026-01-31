from abc import ABC, abstractmethod

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


class BaseBuffer(ABC):
    """Abstract base class for all buffers."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def add(
        self,
        observation: Float[Array, "n_env ..."],
        action: Float[Array, "n_env action_dim"],
        reward: Float[Array, " n_env"],
        next_observation: Float[Array, "n_env ..."],
        done: Bool[Array, " n_env"],
    ):
        pass

    @abstractmethod
    def get_batch(self, batch_size: int) -> dict[str, jnp.ndarray]:
        pass
