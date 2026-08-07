"""Training script for RL agents."""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import hydra
import jax
import jax.numpy as jnp
from loguru import logger
from omegaconf import DictConfig
from pyinstrument import Profiler

from src.utils.logger import init_logger
from src.utils.wandb_logger import finish_wandb, init_wandb, log_metrics


class RLTrainer:
    """RL training orchestrator."""

    def __init__(self, cfg: DictConfig):
        """Initialize trainer.

        Args:
            cfg: Hydra configuration
        """
        # Initialize WandB
        self.wandb_enabled = cfg.wandb.enabled
        self.wandb_run = init_wandb(cfg)

        # Set random seed
        self.rng = jax.random.PRNGKey(cfg.seed)

        # Initialize environment
        logger.info(f"Initializing environment: {cfg.environment._target_}")
        self.environment = hydra.utils.instantiate(cfg.environment)

        # Initialize agent with environment dimensions
        logger.info(f"Initializing agent: {cfg.agent._target_}")
        observation_dim = self.environment.observation_dim
        action_dim = self.environment.action_dim
        self.agent = hydra.utils.instantiate(
            cfg.agent,
            observation_dim=observation_dim,
            action_dim=action_dim,
        )

        # Training configuration
        self.num_episodes = cfg.training.num_episodes
        self.max_steps_per_episode = cfg.training.max_steps_per_episode
        self.eval_frequency = cfg.training.eval_frequency
        self.eval_episodes = cfg.training.eval_episodes

        # Checkpoint configuration
        self.checkpoint_save_frequency = cfg.checkpoint.save_frequency
        self.checkpoint_dir = Path(cfg.path.run_dir) / cfg.path.ckpt_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Rendering configuration
        self.rendering_enabled = cfg.rendering.enabled
        self.video_dir = Path(cfg.path.run_dir) / "gifs"
        if self.rendering_enabled:
            self.video_dir.mkdir(parents=True, exist_ok=True)

        # Metrics tracking
        self.episode_rewards = []
        self.episode_lengths = []

        logger.info("Trainer initialized successfully")

    def run_episode(self, n_env: int, training: bool = True) -> tuple[float, int]:
        """Run a single episode (supports both on-policy and off-policy).

        Args:
            training: Whether in training mode

        Returns:
            Tuple of (total_reward, episode_length)
        """
        # Detect if agent is on-policy (has rollout_buffer) or off-policy (has replay_buffer)
        is_onpolicy = self.agent.isonpolicy

        # Reset environment
        rng = jax.random.split(self.rng, n_env + 1)
        self.rng, reset_rng = rng[0], rng[1:]
        obs, env_state = self.environment.reset(reset_rng)

        episode_reward = jnp.zeros(n_env)
        episode_length = jnp.zeros(n_env, dtype=jnp.int32)
        loss_info = {}
        done = jnp.full(n_env, False)

        # Clear rollout buffer
        if training and is_onpolicy:
            self.agent.buffer.reset()

        while not done.any() and (episode_length < self.max_steps_per_episode).any():
            # Select action
            rng = jax.random.split(self.rng, n_env + 1)
            self.rng, action_rng = rng[0], rng[1:]
            normed_action = self.agent.select_action(obs, action_rng, training=training)

            # Clip action to bounds (continuous)
            if self.environment.is_continuous_action:
                action = self.environment.scale_action(normed_action)
                normed_action = self.environment.normalize_action(action)
            else:
                action = normed_action

            # Step environment
            rng = jax.random.split(self.rng, n_env + 1)
            self.rng, step_rng = rng[0], rng[1:]
            next_obs, env_state, reward, done, env_info = self.environment.step(
                env_state, action, step_rng
            )

            # Store in buffer
            if training:
                log_prob = self.agent.compute_log_prob(obs, normed_action)
                self.agent.buffer.add(
                    obs,
                    normed_action,
                    reward / 100,
                    next_obs,
                    env_info["terminated"],
                    log_prob=log_prob,
                )

            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1

            # Update agent with collected rollout
            if training and len(self.agent.buffer) >= self.agent.learning_starts:
                # Update
                batch = self.agent.buffer.get_batch(self.agent.batch_size // n_env)
                results = self.agent.update(batch)
                for k, v in results.items():
                    loss_info[k] = loss_info.get(k, 0.0) + v

        # TODO: fix episode length averaging
        loss_info = {k: v / episode_length.mean() for k, v in loss_info.items()}
        return episode_reward, episode_length, loss_info

    def evaluate(self, episode: int | None = None) -> dict:
        """Run evaluation episodes.

        Args:
            episode: Current training episode (for video filename)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Running evaluation...")

        if self.rendering_enabled:
            self.environment.start_recording()

        eval_rewards, eval_lengths, _ = self.run_episode(
            n_env=self.eval_episodes, training=False
        )

        metrics = {
            "eval/mean_reward": eval_rewards.mean(),
            "eval/std_reward": eval_rewards.std(),
            "eval/mean_length": eval_lengths.mean(),
        }

        logger.info(
            f"Evaluation: mean_reward={metrics['eval/mean_reward']:.2f} "
            f"(±{metrics['eval/std_reward']:.2f}), "
            f"mean_length={metrics['eval/mean_length']:.1f}"
        )

        if self.rendering_enabled:
            episode_str = f"episode_{episode}" if episode is not None else "final"
            video_path = self.video_dir / f"eval_{episode_str}.gif"
            self.environment.save_video(video_path)
            logger.info(f"Saved evaluation video: {video_path}")

        return metrics

    def save_checkpoint(self, episode: int) -> None:
        """Save agent checkpoint.

        Args:
            episode: Current episode number
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint_episode_{episode}.pkl"
        self.agent.save(str(checkpoint_path))
        logger.info(f"Saved checkpoint: {checkpoint_path}")

    def run_training(self) -> None:
        """Run the main training loop."""
        logger.info(f"Starting training for {self.num_episodes} episodes")

        for episode in range(1, self.num_episodes + 1):
            # Run training episode
            episode_reward, episode_length, loss_info = self.run_episode(
                n_env=self.agent.n_env, training=True
            )

            # Decay epsilon
            self.agent.decay_epsilon()

            # Track metrics
            self.episode_rewards.append(episode_reward.mean())
            self.episode_lengths.append(episode_length.mean())

            # Log training metrics
            train_metrics = {
                "episode": episode,
                "train/episode_reward": episode_reward.mean(),
                "train/episode_length": episode_length.mean(),
            }

            # Add agent-specific metrics (DQN has epsilon and buffer)
            if hasattr(self.agent, "epsilon"):
                train_metrics["train/epsilon"] = self.agent.epsilon
            if hasattr(self.agent, "buffer"):
                train_metrics["train/buffer_size"] = len(self.agent.buffer)

            train_metrics.update(loss_info)
            log_metrics(train_metrics, episode, self.wandb_enabled)

            # Log to console
            if episode % 10 == 0:
                mean_reward = float(jnp.mean(jnp.array(self.episode_rewards[-10:])))
                log_str = (
                    f"Episode {episode}/{self.num_episodes} | "
                    f"Reward: {episode_reward.mean():.2f} | "
                    f"Mean(10): {mean_reward:.2f} | "
                )
                if "loss/total" in loss_info:
                    for loss_name, loss_value in loss_info.items():
                        log_str += f"{loss_name}: {loss_value:.4f} | "
                if hasattr(self.agent, "epsilon"):
                    log_str += f" | Epsilon: {self.agent.epsilon:.3f}"
                if hasattr(self.agent, "buffer"):
                    log_str += f" | Buffer: {len(self.agent.buffer)}"
                logger.info(log_str)

            # Periodic evaluation
            if episode % self.eval_frequency == 0:
                eval_metrics = self.evaluate(episode=episode)
                log_metrics(eval_metrics, episode, self.wandb_enabled)

            # Save checkpoint
            if episode % self.checkpoint_save_frequency == 0:
                self.save_checkpoint(episode)

        # Final evaluation
        logger.info("Training complete! Running final evaluation...")
        final_metrics = self.evaluate()
        log_metrics(final_metrics, self.num_episodes, self.wandb_enabled)

        # Save final checkpoint
        self.save_checkpoint(self.num_episodes)

        # Finish WandB
        finish_wandb(self.wandb_enabled)

        logger.info("Training finished successfully!")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
@logger.catch
def main(cfg: DictConfig):
    """Main entry point.

    Args:
        cfg: Hydra configuration
    """
    init_logger(cfg)

    if cfg.profile:
        profiler = Profiler()
        profiler.start()

    trainer = RLTrainer(cfg)
    trainer.run_training()

    if cfg.profile:
        profiler.stop()
        profiler.print()
        html_output = profiler.output_html()
        output_path = Path(cfg.path.run_dir) / "profiling_report.html"
        output_path.write_text(html_output)


if __name__ == "__main__":
    main()
