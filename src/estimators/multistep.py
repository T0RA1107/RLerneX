import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


@jax.jit
def generalized_advantage_estimation(
    rewards: Float[Array, "n_env T"],
    values: Float[Array, "n_env T"],
    next_values: Float[Array, "n_env T"],
    dones: Bool[Array, "n_env T"],
    discount_gamma: float,
    lambda_: float,
) -> Float[Array, "n_env T"]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards at each timestep
        values: Value estimates at each timestep
        next_values: Value estimates at next timestep
        dones: Done flags at each timestep
        discount_gamma: Discount factor
        lambda_: GAE lambda parameter

    Returns:
        Advantages at each timestep
    """
    n_env, T = rewards.shape
    dones = dones.astype(jnp.float32)
    last_gae_lam = jnp.zeros(n_env)

    def trace(last_gae_lam: Float[Array, " n_env"], t: Int[Array, " T"]):
        non_terminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + discount_gamma * next_values[:, t] * non_terminal - values[:, t]
        last_gae_lam = delta + discount_gamma * lambda_ * non_terminal * last_gae_lam
        return last_gae_lam, last_gae_lam

    _, gae_lam = jax.lax.scan(
        trace,
        last_gae_lam,
        jnp.arange(T).astype(jnp.int32),
        reverse=True
    )

    return gae_lam.T


if __name__ == "__main__":
    # Example usage
    n_env = 2
    T = 5
    rewards = jnp.array([
        [1.0, 0.5, 0.0, 1.0, 0.5],
        [0.0, 1.0, 0.5, 0.0, 1.0]
    ])
    values = jnp.array([
        [0.5, 0.6, 0.4, 0.7, 0.8],
        [0.3, 0.4, 0.5, 0.6, 0.7]
    ])
    next_values = jnp.array([
        [0.6, 0.4, 0.7, 0.8, 0.0],
        [0.4, 0.5, 0.6, 0.7, 0.0]
    ])
    dones = jnp.array([
        [0, 0, 0, 1, 1],
        [0, 0, 1, 0, 1]
    ]).astype(jnp.bool)
    discount_gamma = 0.99
    lambda_ = 0.95

    advantages = generalized_advantage_estimation(
        rewards,
        values,
        next_values,
        dones,
        discount_gamma,
        lambda_
    )

    print(advantages.shape)  # Should be (n_env, T)
    print("Advantages:\n", advantages)