"""PPO (Proximal Policy Optimization) agent for continuous action spaces."""

import pickle
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from jaxtyping import Array, Bool, Float
from omegaconf import DictConfig

from src.agents.base import BaseAgent
from src.buffers.rollout_buffer import RolloutBuffer
from src.estimators.multistep import generalized_advantage_estimation


class PPOAgent(BaseAgent):
    """PPO agent for continuous action spaces."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        network_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        discount_gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_env: int = 8,
        batch_size: int = 64,
        minibatch_size: int = 64,
        internal_epoch: int = 3,
        max_episode_length: int = 200,
        learning_starts: int = 64,
        norm_advantages: bool = True,
        seed: int = 42,
    ):
        """Initialize PPO agent.

        Args:
            observation_dim: Observation space dimension
            action_dim: Action space dimension
            network_cfg: Network configuration
            optimizer_cfg: Optimizer configuration
            discount_gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_eps: PPO clipping epsilon
            entropy_coef: Entropy regularization coefficient
            value_loss_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_env: Number of parallel environments
            batch_size: Batch size for updates
            minibatch_size: Size of shuffled minibatches within each epoch
            internal_epoch: Number of epochs per update
            max_episode_length: Maximum episode length
            learning_starts: Number of steps before learning starts
            norm_advantages: Whether to normalize advantages
            seed: Random seed
        """
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.n_env = n_env
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.discount_gamma = discount_gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.norm_advantages = norm_advantages
        self.learning_starts = learning_starts
        self.internal_epoch = internal_epoch
        self.rng = jax.random.PRNGKey(seed)

        self.buffer = RolloutBuffer(
            n_env,
            max_episode_length,
            observation_dim,
            action_dim,
            store_log_prob=True,
        )

        self.actor_network = hydra.utils.instantiate(
            network_cfg.actor, action_dim=action_dim
        )
        self.critic_network = hydra.utils.instantiate(network_cfg.critic)

        self.rng, actor_rng, critic_rng = jax.random.split(self.rng, 3)
        dummy_obs = jnp.zeros((1, observation_dim))

        actor_params = self.actor_network.init(actor_rng, dummy_obs)
        critic_params = self.critic_network.init(critic_rng, dummy_obs)

        actor_tx = hydra.utils.instantiate(optimizer_cfg.actor)
        actor_tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            actor_tx,
        )
        critic_tx = hydra.utils.instantiate(optimizer_cfg.critic)
        critic_tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            critic_tx,
        )

        self.actor_state = train_state.TrainState.create(
            apply_fn=self.actor_network.apply,
            params=actor_params,
            tx=actor_tx,
        )
        self.critic_state = train_state.TrainState.create(
            apply_fn=self.critic_network.apply,
            params=critic_params,
            tx=critic_tx,
        )

    def select_action(
        self,
        observation: Float[Array, "n_env ..."],
        rng: jax.random.PRNGKey,
        training: bool = True,
    ) -> Float[Array, "n_env action_dim"]:
        """Select action from Gaussian policy.

        Args:
            observation: Current observation
            rng: JAX random key
            training: Whether in training mode

        Returns:
            Tuple of (action, None, updated_rng)
        """
        return self._calc_action(
            self.actor_state,
            observation,
            jax.vmap(lambda k: jax.random.normal(k, self.action_dim))(rng),
            jnp.array(float(training)),
        )

    @staticmethod
    @jax.jit
    def _calc_action(
        actor_state: train_state.TrainState,
        observation: Float[Array, "n_env ..."],
        z: Float[Array, "n_env action_dim"],
        training: Float[Array, ""],
    ):
        mean, log_std = actor_state.apply_fn(actor_state.params, observation)

        std = jnp.exp(log_std)
        pre_tanh_action = mean + std * z * training
        return jnp.tanh(pre_tanh_action)

    def compute_log_prob(
        self,
        observation: Float[Array, "n_env ..."],
        action: Float[Array, "n_env action_dim"],
    ) -> Float[Array, " n_env"]:
        """Compute log probability of action under current policy."""
        return self._compute_log_prob(self.actor_state, observation, action)

    @staticmethod
    @jax.jit
    def _compute_log_prob(
        actor_state: train_state.TrainState,
        observation: Float[Array, "n_env ..."],
        action: Float[Array, "n_env action_dim"],
    ) -> Float[Array, " n_env"]:
        mean, log_std = actor_state.apply_fn(actor_state.params, observation)
        return PPOAgent._gaussian_log_prob(action, mean, log_std)

    @staticmethod
    def _gaussian_log_prob(
        action: Float[Array, "... action_dim"],
        mean: Float[Array, "... action_dim"],
        log_std: Float[Array, "... action_dim"],
    ) -> Float[Array, "..."]:
        """Compute log probability of a tanh-squashed Gaussian action.

        The stored action is ``a = tanh(u)`` with ``u ~ N(mean, std)``. The
        pre-tanh sample is recovered via ``atanh`` and the change-of-variables
        term ``-sum(log(1 - tanh(u)^2))`` is added, using the numerically stable
        form ``log(1 - tanh(u)^2) = 2 * (log 2 - u - softplus(-2u))``.

        Args:
            action: Squashed action taken, in (-1, 1)
            mean: Mean of the pre-tanh Gaussian
            log_std: Log standard deviation of the pre-tanh Gaussian

        Returns:
            Log probability
        """
        eps = 1e-6
        pre_tanh_action = jnp.arctanh(jnp.clip(action, -1 + eps, 1 - eps))
        var = jnp.exp(2 * log_std)
        log_prob = -0.5 * (
            jnp.square(pre_tanh_action - mean) / var + 2 * log_std + jnp.log(2 * jnp.pi)
        )
        log_prob -= 2 * (
            jnp.log(2.0) - pre_tanh_action - jax.nn.softplus(-2 * pre_tanh_action)
        )
        return jnp.sum(log_prob, axis=-1)

    def update(self, batch: dict[str, jnp.ndarray]) -> dict[str, float]:
        """Update actor and critic using PPO.

        Args:
            batch: Batch with observations, actions, rewards, next_observations, dones

        Returns:
            Dictionary of training metrics
        """
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_observations = batch["next_observations"]
        dones = batch["dones"]
        old_log_probs = batch["log_probs"]

        self.rng, update_rng = jax.random.split(self.rng)
        self.actor_state, self.critic_state, losses = self._update_step(
            self.actor_state,
            self.critic_state,
            observations,
            actions,
            rewards,
            next_observations,
            dones,
            old_log_probs,
            update_rng,
            self.discount_gamma,
            self.gae_lambda,
            self.clip_eps,
            self.entropy_coef,
            self.value_loss_coef,
            self.norm_advantages,
            internal_epoch=self.internal_epoch,
            minibatch_size=self.minibatch_size,
        )

        return {
            "loss/total": float(losses["loss"]),
            "loss/actor": float(losses["actor_loss"]),
            "loss/critic": float(losses["critic_loss"]),
            "loss/entropy": float(losses["entropy"]),
        }

    @staticmethod
    @partial(jax.jit, static_argnames=("internal_epoch", "minibatch_size"))
    def _update_step(
        actor_state: train_state.TrainState,
        critic_state: train_state.TrainState,
        observations: Float[Array, "n_env T ..."],
        actions: Float[Array, "n_env T action_dim"],
        rewards: Float[Array, "n_env T"],
        next_observations: Float[Array, "n_env T ..."],
        dones: Bool[Array, "n_env T"],
        old_log_probs: Float[Array, "n_env T"],
        rng: jax.random.PRNGKey,
        discount_gamma: float,
        gae_lambda: float,
        clip_eps: float,
        entropy_coef: float,
        value_loss_coef: float,
        norm_advantages: bool,
        internal_epoch: int,
        minibatch_size: int,
    ):
        """JIT-compiled PPO update step with shuffled minibatch epochs."""
        values = critic_state.apply_fn(critic_state.params, observations).squeeze()
        next_values = critic_state.apply_fn(
            critic_state.params, next_observations
        ).squeeze()

        advantages = generalized_advantage_estimation(
            rewards=rewards,
            values=values,
            next_values=next_values,
            dones=dones,
            discount_gamma=discount_gamma,
            lambda_=gae_lambda,
        ).reshape(-1)
        returns = advantages + values.reshape(-1)

        observations = observations.reshape(-1, observations.shape[-1])
        actions = actions.reshape(-1, actions.shape[-1])
        old_log_probs = old_log_probs.reshape(-1)

        # if norm_advantages: x = (x - mean) / (std + 1e-8)
        advantages = jax.lax.cond(
            norm_advantages,
            lambda x: (x - jnp.mean(x)) / (jnp.std(x) + 1e-8),
            lambda x: x,
            advantages,
        )

        num_samples = observations.shape[0]
        num_minibatches = max(num_samples // minibatch_size, 1)

        def minibatch_update_step(carry, minibatch_indices):
            actor_state, critic_state = carry
            observations_mb = observations[minibatch_indices]
            actions_mb = actions[minibatch_indices]
            old_log_probs_mb = old_log_probs[minibatch_indices]
            advantages_mb = advantages[minibatch_indices]
            returns_mb = returns[minibatch_indices]

            def actor_loss_fn(actor_params):
                mean, log_std = actor_state.apply_fn(actor_params, observations_mb)
                log_probs = PPOAgent._gaussian_log_prob(actions_mb, mean, log_std)

                # PPO clipped objective
                ratio = jnp.exp(log_probs - old_log_probs_mb)
                surr1 = ratio * jax.lax.stop_gradient(advantages_mb)
                surr2 = jnp.clip(
                    ratio, 1 - clip_eps, 1 + clip_eps
                ) * jax.lax.stop_gradient(advantages_mb)
                policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                entropy = jnp.mean(
                    jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
                )

                actor_loss = policy_loss - entropy_coef * entropy
                return actor_loss, entropy

            def critic_loss_fn(critic_params):
                values_mb = critic_state.apply_fn(
                    critic_params, observations_mb
                ).squeeze()
                critic_loss = jnp.mean(
                    jnp.square(values_mb - jax.lax.stop_gradient(returns_mb))
                )
                return critic_loss

            # Actor gradients and update
            (actor_loss, entropy), actor_grads = jax.value_and_grad(
                actor_loss_fn, has_aux=True
            )(actor_state.params)
            actor_state = actor_state.apply_gradients(grads=actor_grads)

            # Critic gradients and update
            critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(
                critic_state.params
            )
            critic_state = critic_state.apply_gradients(grads=critic_grads)
            return (actor_state, critic_state), (actor_loss, critic_loss, entropy)

        def epoch_update_step(carry, epoch_rng):
            permutation = jax.random.permutation(epoch_rng, num_samples)
            minibatch_indices = permutation[: num_minibatches * minibatch_size].reshape(
                num_minibatches, minibatch_size
            )
            return jax.lax.scan(minibatch_update_step, carry, minibatch_indices)

        epoch_rngs = jax.random.split(rng, internal_epoch)
        (actor_state, critic_state), (actor_loss, critic_loss, entropy) = jax.lax.scan(
            epoch_update_step, (actor_state, critic_state), epoch_rngs
        )

        actor_loss = actor_loss.mean()
        critic_loss = critic_loss.mean()
        entropy = entropy.mean()

        losses = {
            "loss": actor_loss + value_loss_coef * critic_loss,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "entropy": entropy,
        }

        return actor_state, critic_state, losses

    def check_action_type(self, action_type: str) -> None:
        assert action_type == "continuous", "PPOAgent only supports continuous actions."

    @property
    def isonpolicy(self) -> bool:
        return True

    def decay_epsilon(self) -> None:
        pass

    def save(self, path: str):
        checkpoint = {
            "actor_params": self.actor_state.params,
            "critic_params": self.critic_state.params,
        }
        with open(path, "wb") as f:
            pickle.dump(checkpoint, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)
        self.actor_state = self.actor_state.replace(
            params=checkpoint["actor_params"],
        )
        self.critic_state = self.critic_state.replace(
            params=checkpoint["critic_params"],
        )
