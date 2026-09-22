# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RLerneX is a JAX-native reinforcement learning framework (JAX/Flax/rlax/optax/Gymnax) with Hydra configuration management and optional Weights & Biases tracking. Python 3.12 only, managed with `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Install pre-commit hooks (runs ruff format + ruff --fix on commit)
uv run pre-commit install

# Lint / format manually
uv run ruff check --fix .
uv run ruff format .

# Train (Hydra CLI overrides select components and hyperparameters)
uv run ./src/train.py environment=cartpole agent=dqn network=mlp optimizer=adam
uv run ./src/train.py environment=pendulum agent=ppo network=actor_critic optimizer=actor_critic_adam

# Hyperparameter sweep (comma-separated values + --multirun)
uv run ./src/train.py agent.discount_gamma=0.9,0.99,0.999 --multirun

# Pre-configured runs
bash scripts/train_cartpole_dqn.sh    # DQN + CartPole (discrete)
bash scripts/train_acrobot_dqn.sh     # DQN + Acrobot (discrete)
bash scripts/train_pendulum_a2c.sh    # A2C + Pendulum (continuous, multirun over gamma)
bash scripts/train_pendulum_ppo.sh    # PPO + Pendulum (continuous)
```

There is no test suite.

Useful override flags: `wandb.enabled=true`, `rendering.enabled=true` (saves eval GIFs), `profile=true` (pyinstrument report), `training.num_episodes=N`.

Outputs go to Hydra run dirs: `logs/train/runs/<date>/<time>/` (or `logs/train/multiruns/...`) containing `.hydra/` config snapshots, `train.log`, checkpoints in `ckpt/`, and GIFs in `gifs/`.

## Architecture

### Hydra composition drives everything

`configs/train.yaml` is the root config; its `defaults` list picks one YAML from each config group (`environment/`, `agent/`, `network/`, `optimizer/`, `logger/`, `hydra/`). The `configs/` groups mirror the `src/` packages, and every component YAML has a `_target_` that `hydra.utils.instantiate` resolves to a class in `src/`.

Important pattern: agent configs set `_recursive_: False` and pass `network_cfg`/`optimizer_cfg` as raw DictConfigs (interpolated from the `network`/`optimizer` groups, e.g. `network_cfg: ${network}`). The agent's `__init__` instantiates its own networks and optimizers via `hydra.utils.instantiate`. Actor-critic agents (A2C/PPO) expect `network.actor` / `network.critic` sub-configs (`network=actor_critic`, `optimizer=actor_critic_adam`); DQN expects a flat network config (`network=mlp`, `optimizer=adam`). Mismatched network/optimizer group selection for an agent fails at instantiation.

### Training loop (`src/train.py`)

`RLTrainer.run_episode` is a single generic loop that handles both on-policy and off-policy agents, branching on `agent.isonpolicy` (on-policy: rollout buffer reset per episode) and `environment.is_continuous_action` (continuous: actions are scaled/normalized between agent and env). Everything is vectorized over `agent.n_env` parallel environments via JAX `vmap`; RNG keys are split per-env each step. Rewards are scaled by 1/100 before being stored in the buffer. Updates start once `len(agent.buffer) >= agent.learning_starts`, drawing `agent.batch_size // n_env` batches.

### Agent contract (`src/agents/base.py`)

New algorithms subclass `BaseAgent` and implement: `select_action`, `update(batch)`, `check_action_type`, `isonpolicy` property, `save`/`load` (pickle), and optionally `compute_log_prob` (needed for policy-gradient methods; base returns None). Agents own their buffer (`RolloutBuffer` for on-policy, `ReplayBuffer` for off-policy from `src/buffers/`) plus `n_env`, `batch_size`, `learning_starts` attributes that the trainer reads. To add an algorithm: implement the agent in `src/agents/`, add `configs/agent/<name>.yaml` with its `_target_`, and optionally a `scripts/train_*.sh`.

### Environments (`src/environments/gymnax_wrapper.py`)

`GymnaxWrapper` wraps Gymnax envs with vmapped `reset`/`step`, exposes `observation_dim`/`action_dim`/`is_continuous_action`, action scaling/normalization for continuous spaces, and GIF recording. Adding an environment is usually just a new `configs/environment/<name>.yaml` with the Gymnax `env_name`.

### Other pieces

- `src/estimators/multistep.py`: GAE (generalized advantage estimation) used by A2C/PPO.
- `src/networks/`: Flax modules (`MLPQNetwork`, `GaussianActor`, `ValueCritic`); activation functions are injected via Hydra `hydra.utils.get_method` config, not hardcoded.
- `docs/<ALGO>.md`: per-algorithm notes linking the original papers and the papers behind implementation choices (why tanh squash, why truncation bootstrapping, etc.). Add one when adding an algorithm; index in `docs/README.md`.
- Shapes are annotated with `jaxtyping` (`Float[Array, "n_env ..."]`); ruff ignores `F722` to allow this syntax.

## Caveats

- Environment rendering (`rendering.enabled=true`) requires copying `misc/visualize/*` into the installed gymnax package's `visualize/` directory (see README "Visualization").
- Match algorithm to action space: DQN is discrete-only (CartPole, Acrobot); A2C/PPO are continuous (Pendulum). Agents validate this via `check_action_type`.
