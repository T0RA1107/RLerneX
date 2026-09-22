# DQN (Deep Q-Network)

## Overview

Off-policy, value-based method for discrete action spaces. A neural network
approximates the action-value function Q(s, a); experience replay and a target
network stabilise the bootstrapped regression. Exploration is ε-greedy.

## Original papers

- Mnih et al., **"Playing Atari with Deep Reinforcement Learning"** (2013)
  <https://arxiv.org/abs/1312.5602>
  Experience replay combined with a convolutional Q-network; first human-level results on Atari.
- Mnih et al., **"Human-level control through deep reinforcement learning"** (Nature, 2015)
  <https://www.nature.com/articles/nature14236>
  Introduces the target network: bootstrap targets are computed with a periodically
  frozen copy of the parameters, which removes the instability of regressing onto a moving target.

## Implementation in this repository

- Agent: `src/agents/dqn.py` (`DQNAgent`)
- Network: `src/networks/mlp.py` (`MLPQNetwork`), an MLP rather than a CNN since the environments are low-dimensional classic control tasks
- Buffer: `src/buffers/replay_buffer.py` (`ReplayBuffer`), one circular buffer per parallel environment
- Config: `configs/agent/dqn.yaml`

| Aspect | Implementation |
|---|---|
| TD error | `rlax.q_learning` under `vmap`; plain one-step Q-learning (not Double DQN) |
| Loss | mean squared TD error (not Huber) |
| Target network | hard copy of the online parameters at a fixed step interval |
| Exploration | ε decays multiplicatively per episode down to a floor |
| Gradient clipping | `optax.clip_by_global_norm` |
| Reward scale | rewards are scaled down in the trainer (shared by all algorithms) |

## Related papers and resources

- Lin, **"Self-Improving Reactive Agents Based on Reinforcement Learning, Planning and Teaching"** (1992)
  <https://link.springer.com/article/10.1007/BF00992699>
  Origin of experience replay: breaks sample correlation and improves data efficiency. Basis for `ReplayBuffer`.
- van Hasselt, Guez & Silver, **"Deep Reinforcement Learning with Double Q-learning"** (AAAI 2016)
  <https://arxiv.org/abs/1509.06461>
  Decouples action selection from action evaluation to reduce the overestimation caused by the max operator.
  **Not implemented**; switching to `rlax.double_q_learning` would be sufficient.
- Wang et al., **"Dueling Network Architectures for Deep Reinforcement Learning"** (ICML 2016)
  <https://arxiv.org/abs/1511.06581>
  Decomposes Q into a state value and an advantage stream. **Not implemented**.
- rlax documentation <https://rlax.readthedocs.io/>
  Argument conventions of `q_learning` (`q_tm1, a_tm1, r_t, discount_t, q_t`). Passing `(1 - done) * γ`
  as `discount_t` handles terminal transitions.

## Notes

- The discount factor inside `_update_step` is hard-coded rather than read from
  `discount_gamma`, so the config value does not affect the TD target.
- DQN is discrete-only and asserts this in `check_action_type`. Use A2C or PPO for continuous action spaces.
