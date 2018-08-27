### energy_py

lessons learnt building an energy reinforcement learning library

Adam Green - adam.green@adgefficiency.com

---?image=/assets/humber.jpg&size=auto 100%

---?image=/assets/humber_excel.png&size=auto 100%

---?image=/assets/climate.png&size=auto 80%

---

three pieces of info on climate


---

ml + energy solution to the climate problem

---

price response flexible demand and the lazy taxi driver

---?image=/assets/repo.png&size=auto 100%

---

energy_py = supporting experimentation 

---

---?color=#000000

```bash
$ cd energy_py/experiments

$ python experiment.py example dqn

$ tensorboard --logdir='./results/example/tensorboard'
```

---

```python
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
```

---

```python
observation = env.reset()
done = False

while not done:

    action = agent.act(observation)

    next_observation, reward, done, info = env.step(action)

    training_info = agent.learn()

    observation = next_observation
```

---

details

---

```python
action_space = GlobalSpace('action').from_spaces(
    [ContinuousSpace(0, 100), DiscreteSpace(3)],
    ['acceleration', 'gear']
)

action = action_space.sample()

assert action_space.contains(action)

discrete_spaces = action_space.discretize(20)

action = action_space.sample_discrete()

```
---

```python
#  load a state or observation space from a dataset
state_space = GlobalSpace('state').from_dataset('example')

# we can sample an episode from the state
episode = state_space.sample_episode(start=0, end=100)

# sample from the current episode by calling the space
state = state_space(steps=0)
```

---

performance

---

lessons

---

simplicity

---

three pieces of info on energy and reinforcement learning

---

---?image=/assets/mcts_dqn.png&size=auto 100%

> [Deep Reinforcment Learning Doesn't Work Yet](https://www.alexirpan.com/2018/02/14/rl-hard.html)

---

current performance (show the image of the really smooth learning - don't think this is good (looks like rollout memorization)

---

synthetic data - aka poor mans GANS

---

---

the environment model problem / oppourtunity

---

combining with supervised learning

---

**short term work**

loading memories

test and train expts

early stopping

experiment result analysis

backwards induction

---

**long term work**

wrapping other environments

+

model based methods 

---

thank you

