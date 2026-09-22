# RLerneX - JAX/Flax/rlax Reinforcement Learning Framework

> **RL** + **Lerne** (Lernaean in French) + **X** (JAX)

A modular reinforcement learning framework built with JAX, Flax, rlax, and Gymnax. Features Hydra configuration management and Weights & Biases integration for experiment tracking.

## Features

- **JAX-native implementation**: Leverages JAX for JIT compilation and automatic differentiation
- **Modular design**: Easy to extend with new algorithms, environments, and network architectures
- **Configuration-driven**: All hyperparameters managed via Hydra YAML files
- **Multiple algorithms**: DQN (off-policy) and A2C (on-policy) with unified training interface
- **Gymnax environments**: Fast, JAX-based gym environments (CartPole, Pendulum, etc.)
- **Experiment tracking**: Weights & Biases integration for metrics and visualization
- **Vectorized training**: Support for parallel environment execution via JAX vmap

## Implemented Algorithms

| Algorithm | Type | Action Space | Description |
|-----------|------|--------------|-------------|
| **DQN** | Off-policy, Value-based | Discrete | Deep Q-Network with experience replay and target network |
| **A2C** | On-policy, Policy gradient | Continuous | Advantage Actor-Critic with GAE and entropy regularization |

## Project Structure

```
RLerneX/
├── configs/                      # Hydra configuration files
│   ├── train.yaml               # Main training config
│   ├── environment/             # Environment configs
│   ├── agent/                   # Agent/algorithm configs
│   ├── network/                 # Network architecture configs
│   ├── optimizer/               # Optimizer configs
│   ├── logger/                  # Logger configs
│   └── hydra/                   # Hydra framework configs
├── src/                         # Source code
│   ├── agents/                  # RL agent implementations
│   ├── environments/            # Environment wrappers
│   ├── networks/                # Neural network architectures
│   ├── buffers/                 # Experience buffers
│   ├── estimators/              # Value estimation utilities
│   ├── utils/                   # Utility functions
│   └── train.py                 # Training orchestration
├── scripts/                     # Training scripts
│   ├── train_cartpole_dqn.sh   # DQN on CartPole
│   └── train_pendulum_a2c.sh   # A2C on Pendulum
├── pyproject.toml              # Project dependencies
└── README.md                    # This file
```

## Installation
Install dependencies.
```bash
uv sync
```
Install pre-commit
```bash
sudo apt install pre-commit
uv run pre-commit install
```

## Quick Start

### Training Scripts

Pre-configured training scripts are available in the `scripts/` directory:

```bash
# DQN on CartPole (discrete actions)
bash scripts/train_cartpole_dqn.sh

# A2C on Pendulum (continuous actions)
bash scripts/train_pendulum_a2c.sh
```

## Configuration System

The project uses Hydra for hierarchical configuration management.

## Outputs

Training outputs are organized by Hydra:

```
logs/
└── train/
    └── runs/
        └── YYYY-MM-DD/          # Date
            └── HH-MM-SS/        # Time
                ├── .hydra/      # Hydra config snapshots
                ├── train.log    # Training logs
                └── ckpt/        # Model checkpoints
                    └── checkpoint_episode_X.pkl
```

## Current Implementation

### Algorithms
| Algorithm | Type | Action Space | Buffer | Key Features |
|-----------|------|--------------|--------|--------------|
| **DQN** | Off-policy | Discrete | Replay Buffer | Target network, epsilon-greedy exploration |
| **A2C** | On-policy | Continuous | Rollout Buffer | GAE, entropy regularization, gradient clipping |

### Environments
- **CartPole-v1**: Discrete action space (2 actions), 4D observation
- **Pendulum-v1**: Continuous action space (1D), 3D observation

### Networks
- **MLP Q-Network**: For value-based methods (DQN)
- **Gaussian Actor**: Outputs mean and log_std for continuous policy
- **Actor Critic**: Outputs scalar V(s) for advantage calculation
- **Custom Activations**: Mish activation function

### Buffers
- **ReplayBuffer**: Circular buffer for off-policy experience replay
- **RolloutBuffer**: Episode storage for on-policy methods

### Estimators
- **GAE**: Generalized Advantage Estimation for policy gradients

## Visualization

To visualize Gymnax environments, you need to copy the files from `misc/visualize/` to the `gymnax/visualize/` directory in your gymnax library installation.

```bash
# Find the gymnax library installation path
python -c "import gymnax; print(gymnax.__path__[0])"

# Copy misc/visualize contents to gymnax/visualize
cp misc/visualize/* <gymnax_path>/visualize/
```

This enables gym environment rendering through gymnax's visualization functionality.

## Roadmap

Future improvements:

- [ ] Add PPO (on-policy, both discrete and continuous)
- [ ] Add SAC (off-policy actor-critic for continuous control)
- [ ] CNN networks for visual observations
- [ ] Distributed training
- [ ] Advanced exploration strategies
- [ ] Model checkpointing with best model selection

### Import Errors

Ensure you're in the project root and dependencies are installed:

```bash
uv sync
```

### Action Space Mismatch

Ensure you use the correct algorithm for your environment:
- **Discrete actions** (CartPole): Use DQN with `agent=dqn`
- **Continuous actions** (Pendulum): Use A2C with `agent=a2c`

## References

Per-algorithm paper notes (original papers + papers behind implementation choices) live in [`docs/`](docs/README.md).

- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
- [rlax Documentation](https://github.com/google-deepmind/rlax)
- [Gymnax Documentation](https://github.com/RobertTLange/gymnax)
- [Hydra Documentation](https://hydra.cc/)
- [Weights & Biases Documentation](https://docs.wandb.ai/)

## License

This project is open source and available under the MIT License.
