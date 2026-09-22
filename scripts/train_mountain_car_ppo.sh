uv run ./src/train.py \
  environment=mountain_car_continuous \
  agent=ppo \
  network=actor_critic \
  optimizer=actor_critic_adam \
  agent.batch_size=1600 \
  agent.learning_starts=1600 \
  +agent.max_episode_length=500 \
  agent.discount_gamma=0.99 \
  training.max_steps_per_episode=500 \
  training.num_episodes=300 \
  training.eval_frequency=50 \
  wandb.enabled=false \
  rendering.enabled=true
