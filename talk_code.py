import tensorflow as tf
import energy_py


with tf.Session() as sess:
    env = energy_py.make_env(
        env_id='battery',
        episode_length=288,
        dataset='example'
    )

    agent = energy_py.make_agent(
        sess=sess,
        agent_id='dqn',
        env=env,
        total_steps=1000000
        )

    observation = env.reset()

    done = False
    while not done:
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        training_info = agent.learn()
        observation = next_observation
