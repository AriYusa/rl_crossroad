from gym.envs.registration import register

register(
    id='JackalCrossroad-v0',
    entry_point='jackal_crossroad_env.crossroad_task_env:CrossroadEnv',
    max_episode_steps=1000,
)
