"""DQN agent implementation using JAX, Flax, and rlax."""
import pickle

import hydra
import jax
import jax.numpy as jnp
import optax
import rlax
from flax.training import train_state
from jaxtyping import Array, Bool, Float, Int
from omegaconf import DictConfig

from src.agents.base import BaseAgent
from src.buffers.replay_buffer import ReplayBuffer


class TrainState(train_state.TrainState):
    """Extended TrainState for DQN with target network."""
    target_params: dict[str, jnp.ndarray] = None


class DQNAgent(BaseAgent):
    """DQN agent with epsilon-greedy exploration and target network."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        network_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        discount_gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        max_grad_norm: float = 0.5,
        n_env: int = 8,
        batch_size: int = 64,
        buffer_size: int = 10000,
        learning_starts: int = 1000,
        target_update_frequency: int = 100,
        seed: int = 42,
    ):
        """Initialize DQN agent.

        Args:
            observation_dim: Dimension of observation space
            action_dim: Number of discrete actions
            network_cfg: Configuration for Q-network
            optimizer_cfg: Configuration for optimizer
            discount_gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            batch_size: Batch size for training
            buffer_size: Replay buffer size
            learning_starts: Steps before starting training
            target_update_frequency: Steps between target network updates
            seed: Random seed
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discount_gamma = discount_gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.max_grad_norm = max_grad_norm
        self.n_env = n_env
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_frequency = target_update_frequency

        # Initialize RNG
        self.rng = jax.random.PRNGKey(seed)

        # Create replay buffer
        self.rng, buffer_rng = jax.random.split(self.rng)
        self.buffer = ReplayBuffer(
            n_env, buffer_size, observation_dim, 1, buffer_rng
        )

        # Initialize networks
        self.rng, network_rng = jax.random.split(self.rng)
        self.q_network = hydra.utils.instantiate(network_cfg, action_dim=action_dim)

        # Initialize network parameters
        dummy_obs = jnp.zeros((1, observation_dim))
        params = self.q_network.init(network_rng, dummy_obs)

        # Create optimizer
        optimizer = hydra.utils.instantiate(optimizer_cfg)
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optimizer
        )

        # Create train state with target network
        self.state = TrainState.create(
            apply_fn=self.q_network.apply,
            params=params,
            target_params=params,
            tx=optimizer,
        )

        # Training step counter
        self.training_steps = 0

    def select_action(
        self,
        observation: Float[Array, "n_env ..."],
        rng: jax.random.PRNGKey,
        training: bool = True
    ) -> Float[Array, "n_env action_dim"]:
        """Select action using epsilon-greedy policy.

        Args:
            observation: Current observation
            rng: JAX random number generator key
            training: Whether in training mode

        Returns:
            Tuple of (action, info_dict, updated_rng)
        """
        rng = jax.vmap(lambda k: jax.random.split(k, 2))(rng)
        epsilon_rng, action_rng = rng[:, 0], rng[:, 1]
        return self._calc_action(
            self.state,
            observation,
            jax.vmap(lambda k: jax.random.uniform(k) < self.epsilon)(epsilon_rng),
            jax.vmap(lambda k: jax.random.randint(k, (), 0, self.action_dim))(action_rng),
            training
        )

    @staticmethod
    @jax.jit
    def _calc_action(
        state: train_state.TrainState,
        observation: Float[Array, "n_env ..."],
        explore: Bool[Array, " n_env"],
        random_action: Float[Array, "n_env action_dim"],
        training: Bool[Array, ""]
    ):
        q_values = state.apply_fn(state.params, observation)
        greedy_action = jnp.argmax(q_values, axis=-1)
        action = jnp.where(explore & training, random_action, greedy_action)

        return action

    def update(self, batch: dict[str, jnp.ndarray]) -> dict[str, float]:
        """Update Q-network using a batch of transitions.

        Args:
            batch: Dictionary containing transitions

        Returns:
            Dictionary of training metrics
        """
        # Convert numpy arrays to JAX arrays
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]

        # Update Q-network
        self.state, loss, q_values = self._update_step(
            self.state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        )

        self.training_steps += 1

        # Update target network periodically
        if self.training_steps % self.target_update_frequency == 0:
            self.state = self.state.replace(target_params=self.state.params)

        return {
            "loss/total": float(loss),
            "mean_q_value": float(q_values.mean()),
            "epsilon": self.epsilon,
        }

    @staticmethod
    @jax.jit
    def _update_step(
        state: TrainState,
        observations: Float[Array, "n_env T ..."],
        actions: Int[Array, "n_env T"],
        rewards: Float[Array, "n_env T"],
        next_observations: Float[Array, "n_env T ..."],
        dones: Bool[Array, "n_env T"],
    ) -> tuple[TrainState, Float[Array, ""], Float[Array, "n_env T"]]:
        """JIT-compiled training step.

        Args:
            state: Current training state
            observations: Batch of observations
            actions: Batch of actions
            rewards: Batch of rewards
            next_observations: Batch of next observations
            dones: Batch of done flags

        Returns:
            Tuple of (updated_state, loss, q_values)
        """
        observations = observations.reshape(-1, observations.shape[-1])
        actions = actions.reshape(-1)
        rewards = rewards.reshape(-1)
        next_observations = next_observations.reshape(-1, next_observations.shape[-1])
        dones = dones.reshape(-1)

        def loss_fn(params):
            # Compute Q-values for current observations
            q_values = state.apply_fn(params, observations)

            # Compute target Q-values using target network
            target_q_values = state.apply_fn(state.target_params, next_observations)

            # Use rlax q_learning for TD error computation (vectorized)
            td_errors = jax.vmap(rlax.q_learning)(
                q_tm1=q_values,
                a_tm1=actions,
                r_t=rewards,
                discount_t=(1.0 - dones) * 0.99,
                q_t=target_q_values
            )

            # Mean squared TD error
            loss = jnp.mean(td_errors ** 2)

            return loss, q_values

        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, q_values), grads = grad_fn(state.params)

        # Apply gradients
        state = state.apply_gradients(grads=grads)

        return state, loss, q_values

    def check_action_type(self, action_type: str) -> None:
        assert action_type == "discrete", "DQNAgent only supports discrete action spaces."

    @property
    def isonpolicy(self) -> bool:
        """Return whether the agent is on-policy.

        Returns:
            True if on-policy, False if off-policy
        """
        return False

    def decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save agent parameters to file.

        Args:
            path: File path to save parameters
        """
        checkpoint = {
            "params": self.state.params,
            "target_params": self.state.target_params,
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str) -> None:
        """Load agent parameters from file.

        Args:
            path: File path to load parameters from
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        self.state = self.state.replace(
            params=checkpoint["params"],
            target_params=checkpoint["target_params"],
        )
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
