uv run ./src/train.py \
  environment=pendulum \
  agent=a2c \
  network=actor_critic \
  optimizer=actor_critic_adam \
  agent.discount_gamma=0.9,0.99,0.999 \
  training.num_episodes=500 \
  wandb.enabled=false \
  rendering.enabled=true \
  --multirun
