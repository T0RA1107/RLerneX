# GAE (Generalized Advantage Estimation) and time-limit handling

The advantage estimator shared by A2C and PPO. Implementation: `src/estimators/multistep.py`.

## Overview

With the TD error δ_t = r_t + γ V(s_{t+1}) − V(s_t), the advantage is the
exponentially weighted sum A_t = Σ_k (γλ)^k δ_{t+k}. λ = 0 gives one-step TD
(low variance, high bias); λ = 1 gives the Monte-Carlo return (high variance, low bias).

## Original paper

- Schulman et al., **"High-Dimensional Continuous Control Using Generalized Advantage Estimation"** (ICLR 2016)
  <https://arxiv.org/abs/1506.02438>
  Defines the estimator above and evaluates it with TRPO on continuous control.

## Implementation in this repository

- `generalized_advantage_estimation(rewards, values, next_values, dones, discount_gamma, lambda_)`
  runs a reverse-time `jax.lax.scan` over a `(n_env, T)` batch.
- `next_values` is passed explicitly, and `dones` contains **terminations only**, not truncations.
- The return used as the critic target is recovered as `advantages + values` (the λ-return).

## Related papers and resources

- Pardo et al., **"Time Limits in Reinforcement Learning"** (ICML 2018)
  <https://arxiv.org/abs/1712.00378>
  Treating a time limit as a terminal state makes the value of a state depend on the remaining time,
  which corrupts value learning. At a time limit one should bootstrap from the next state's value instead.
  Environments with a fixed episode length (such as Pendulum) are directly affected.
- Gymnasium `terminated` / `truncated` split <https://gymnasium.farama.org/api/env/>
  API-level reflection of the above. `GymnaxWrapper` exposes `info["terminated"]` and `info["truncated"]` with the same meaning.

## Notes

- `GymnaxWrapper` steps through `step_env` (no auto-reset) so that `next_obs` at a time limit is the true
  successor state; the trainer stores only `terminated` in the buffers. This distinction matters for both A2C and PPO.
- Rewards are scaled down in the trainer, so absolute return and critic-loss magnitudes are small compared to
  the raw environment reward.
