# A2C (Advantage Actor-Critic)

## Overview

On-policy actor-critic. The actor outputs an action distribution, the critic
estimates the state value, and the advantage computed from the critic reduces the
variance of the policy gradient. A2C is the synchronous variant of A3C: several
environments are stepped in parallel and updated as one batch.

## Original papers

- Mnih et al., **"Asynchronous Methods for Deep Reinforcement Learning"** (ICML 2016)
  <https://arxiv.org/abs/1602.01783>
  A3C. Multiple workers send gradients asynchronously; combines advantage actor-critic with entropy regularisation.
- OpenAI Baselines (A2C) <https://github.com/openai/baselines>
  Synchronous re-implementation of A3C. The asynchrony was found to be non-essential; batching over
  parallel environments is enough. The `n_env`-parallel synchronous update here follows this design.

## Implementation in this repository

- Agent: `src/agents/a2c.py` (`A2CAgent`)
- Networks: `src/networks/actor_critic.py` (`GaussianActor`, `ValueCritic`); separate networks and separate optimisers for actor and critic
- Buffer: `src/buffers/rollout_buffer.py` (`RolloutBuffer`)
- Estimator: `src/estimators/multistep.py` (GAE, see [GAE.md](GAE.md))
- Configs: `configs/agent/a2c.yaml`, `configs/network/actor_critic.yaml`, `configs/optimizer/actor_critic_adam.yaml`

| Aspect | Implementation |
|---|---|
| Policy | diagonal Gaussian; the mean is state-dependent, `log_std` is a state-independent parameter clipped to a range |
| Actor loss | `-mean(log π(a|s) · A)` minus an entropy bonus; A is `stop_gradient`-ed |
| Critic loss | squared error between V(s) and the λ-return `A + V(s)` |
| Advantage | GAE, optionally standardised within the batch |
| Update frequency | every fixed number of environment steps (short rollout slices), one epoch per update |
| Action bounds | **none in the policy**; the trainer clips to the action box before stepping the environment, and the log-probability is computed on the clipped action |

## Related papers and resources

- Williams, **"Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"** (1992)
  <https://link.springer.com/article/10.1007/BF00992696>
  REINFORCE, the origin of the policy gradient ∇J = E[∇log π · G]. Subtracting a baseline keeps the estimator
  unbiased, which is what justifies using the advantage.
- Schulman et al., **"High-Dimensional Continuous Control Using Generalized Advantage Estimation"** (ICLR 2016)
  <https://arxiv.org/abs/1506.02438>
  GAE; see [GAE.md](GAE.md).
- Fujita & Maeda, **"Clipped Action Policy Gradient"** (ICML 2018)
  <https://proceedings.mlr.press/v80/fujita18a.html>
  Points out that policies are usually optimised as if actions were not clipped, although the environment clips them.
  A2C in this repository still has this issue; PPO resolves it with a tanh-squashed policy (see [PPO.md](PPO.md)).
- Andrychowicz et al., **"What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study"** (ICLR 2021)
  <https://arxiv.org/abs/2006.05990>
  Large-scale comparison of design choices (advantage normalisation, initial policy std, action-distribution
  parameterisation, and more). Useful as a source of default recommendations.

## Notes

- Time limits are treated as truncation, not termination, and are bootstrapped through (see [GAE.md](GAE.md)).
- Short rollout slices with a single epoch make A2C noticeably higher-variance than PPO on the same task.
- `_calc_action` still contains the commented-out tanh squash. Enabling it requires changing
  `_gaussian_log_prob` at the same time to include the Jacobian correction, as done in PPO.
