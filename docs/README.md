# docs/

One file per implemented algorithm. Each file links the original papers and the
papers behind the implementation choices made in this repository.
How the code is organised is covered by `CLAUDE.md` and `README.md`; these notes
only explain **why** the implementation looks the way it does.

| File | Contents |
|---|---|
| [DQN.md](DQN.md) | Deep Q-Network (discrete actions, off-policy) |
| [A2C.md](A2C.md) | Advantage Actor-Critic (continuous actions, on-policy) |
| [PPO.md](PPO.md) | Proximal Policy Optimization (continuous actions, on-policy) |
| [GAE.md](GAE.md) | Generalized Advantage Estimation and time-limit handling (shared by A2C / PPO) |

## Structure of each file

1. **Overview** — the method in a few lines
2. **Original papers** — links and the key idea of each
3. **Implementation in this repository** — relevant files and how the implementation differs from the standard recipe
4. **Related papers and resources** — link, short summary, and how it relates to this implementation
5. **Notes** — known limitations and pitfalls

When adding an algorithm, create `docs/<NAME>.md` with the same structure and add a row to the table above.
