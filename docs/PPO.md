# PPO (Proximal Policy Optimization)

## Overview

On-policy actor-critic. The size of each policy update is limited by clipping the
importance ratio r_t(θ) = π_θ(a|s) / π_old(a|s), which makes it safe to run several
epochs of minibatch updates on the same rollout. PPO is a first-order approximation
of the trust-region constraint of TRPO.

## Original papers

- Schulman et al., **"Proximal Policy Optimization Algorithms"** (2017)
  <https://arxiv.org/abs/1707.06347>
  Clipped objective L^CLIP = E[min(r A, clip(r, 1−ε, 1+ε) A)] with multiple epochs of minibatch SGD.
- Schulman et al., **"Trust Region Policy Optimization"** (ICML 2015)
  <https://arxiv.org/abs/1502.05477>
  Predecessor: KL-constrained policy optimisation solved with conjugate gradients. PPO replaces the constraint with clipping.

## Implementation in this repository

- Agent: `src/agents/ppo.py` (`PPOAgent`)
- Networks, buffer and GAE are shared with A2C ([A2C.md](A2C.md), [GAE.md](GAE.md))
- Config: `configs/agent/ppo.yaml`

| Aspect | Implementation |
|---|---|
| Rollout | a full episode from all parallel environments is collected, then one update is run |
| Epochs / minibatches | several epochs of shuffled minibatches, implemented as a nested `lax.scan` |
| Policy | **tanh-squashed diagonal Gaussian**: a = tanh(u), u ~ N(μ(s), σ) |
| Log-probability | the pre-tanh sample is recovered from the stored action via `atanh` (with a small clip away from ±1), and the Jacobian term −Σ log(1 − tanh²u) is added in the numerically stable form 2(log 2 − u − softplus(−2u)) |
| Actor loss | clipped surrogate minus an entropy bonus; the entropy is that of the pre-tanh Gaussian (an approximation, not the entropy of the squashed distribution) |
| Critic loss | squared error between V and the λ-return; no value clipping |
| Advantage | GAE, standardised within the batch |
| Gradient clipping | global-norm clipping on both actor and critic |
| Early stopping | **none**; epochs are not cut short on an approximate-KL threshold |

## Related papers and resources

### tanh squashing and the change of variables

- Haarnoja et al., **"Soft Actor-Critic"** (ICML 2018), Appendix C "Enforcing Action Bounds"
  <https://arxiv.org/abs/1801.01290>
  Applies an invertible tanh to an unbounded Gaussian and corrects the density by the change-of-variables formula:
  log π(a|s) = log N(u; μ, σ²) − Σ log(1 − tanh²(u_i)). Source of the formula in `_gaussian_log_prob`.
- OpenAI Spinning Up, SAC implementation <https://spinningup.openai.com/en/latest/algorithms/sac.html>
  Source of the numerically stable form 2(log 2 − u − softplus(−2u)) of the Jacobian term.

### Likelihood of clipped actions

- Fujita & Maeda, **"Clipped Action Policy Gradient"** (ICML 2018)
  <https://proceedings.mlr.press/v80/fujita18a.html>
  Actions are clipped before being sent to the environment, but the policy is optimised as if they were not.
  Actions on the boundary should be treated as probability mass beyond the bound (a CDF), not as a density.
  The trainer here computes the log-probability on the **clipped** action, so before tanh squashing was introduced
  PPO had exactly this problem; the squash makes the stored action an exact, invertible function of the sample.
- Huang et al., **"The 37 Implementation Details of Proximal Policy Optimization"** (ICLR Blog Track 2022)
  <https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/>
  For continuous actions the standard recipe stores the raw sample and clips only when stepping the environment.
  Also covers advantage normalisation, value clipping, learning-rate annealing and other details.

### Gaussian policies on bounded action spaces

- Chou, Maturana & Scherer, **"Improving Stochastic Policy Gradients in Continuous Control ... using the Beta Distribution"** (ICML 2017)
  <https://proceedings.mlr.press/v70/chou17a.html>
  Shows the bias a Gaussian policy incurs near the boundary of a bounded action space and proposes a Beta policy.
- Hsu, Mendler-Dünner & Hardt, **"Revisiting Design Choices in Proximal Policy Optimization"** (2020)
  <https://arxiv.org/abs/2009.10897>
  Analyses failure modes of PPO. One of them is a Gaussian policy drifting away from the high-reward region of a
  bounded action space and collapsing; KL-regularised PPO or a Beta policy mitigate it. The unbounded mean drifting far
  outside the action box, observed here before tanh squashing, is an instance of this failure mode.

### PPO clipping does not bound the ratio

- Wang, He & Tan, **"Truly Proximal Policy Optimization"** (UAI 2019)
  <https://arxiv.org/abs/1903.07940>
  The clipped objective lets the ratio leave the clip range by a large margin. For A < 0 and r ≫ 1 the objective is
  deliberately left unclipped as a pessimistic bound, so loss and gradient are unbounded on that side.
- Engstrom et al., **"Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO"** (ICLR 2020)
  <https://arxiv.org/abs/2005.12729>
  PPO's performance depends heavily on code-level details; the clipping alone does not enforce a trust region.
- Andrychowicz et al., **"What Matters In On-Policy Reinforcement Learning?"** (ICLR 2021)
  <https://arxiv.org/abs/2006.05990>
  Large-scale study; source of recommendations on epoch count, minibatch count, advantage normalisation and action distribution.

## Notes

- Divergence mechanism observed with the unsquashed policy: as the policy std shrinks and the unbounded mean moves outside
  the action box, the log-probability of the clipped boundary action becomes extremely negative and extremely sensitive
  to the parameters. Many Adam steps per rollout then drive |log r| far beyond the clip range, the unclipped side of the
  objective blows up, `exp` overflows in float32, and the NaN propagates through `clip_by_global_norm` into the parameters.
  This is an optimisation failure rather than overfitting; evaluation return stays flat until the crash.
  The tanh-squashed policy removes both the clipped-likelihood mismatch and the unbounded action.
- Remaining levers: approximate-KL early stopping of epochs, a larger entropy coefficient (the std still decays monotonically),
  and monitoring saturation of the pre-tanh mean.
- The logged `loss/*` values are divided by the episode length in the trainer, so they are not the raw per-update losses.
- PPO is implemented for continuous actions only (`check_action_type`); a discrete variant does not exist.
