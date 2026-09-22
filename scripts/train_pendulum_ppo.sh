uv run ./src/train.py \
  environment=pendulum \
  agent=ppo \
  network=actor_critic \
  optimizer=actor_critic_adam \
  training.num_episodes=500 \
  wandb.enabled=false \
  rendering.enabled=true
