### energy_py

<br><br>

two years of lessons building an energy reinforcement learning library

<br><br>

adgefficiency.com

---?image=/assets/humber.jpg&size=auto 100%

Note:
- this place was where I learnt energy engineering

---?image=/assets/humber_excel.png&size=auto 100%&color=#ffffff

Note:
- this was where I learn the power of a computer

---?image=/assets/climate.png&size=auto 80%&color=#ffffff

---?image=/assets/repo.png&size=auto 100%&color=#ffffff

---

computation to help solve the climate problem

Note:

1. it's bad - and worse for poor countries
2. technology is not the only solution - business models, policy, public opinion
3. your personal choices matter

- RL = solve the control problem
- digitization challenge in energy


---

single agent - DDQN

energy environments 
- price arbitrage in electric battery storage 
- price responsive flexible electricity demand

wrappers around `gym` envs

tools for experimentation

---

price response flexible demand - the lazy taxi driver

Note:

- taxi system where government ensures that no one ever waits for a taxi
- supply of taxis == peak demand for taxis
- cost = fixed + variable
- the laziest taxi drivers need to recover their entire fixed + var costs through a few trips
- flexibility solves the problem by letting people wait
- responding to price (not signal from system operator)

---?image=/assets/flex_env.png&size=auto 80%&color=#ffffff

---

flexibility as a MDP

- state = current + forecast electricity prices, customer demands
- action = flex up, flex down or no-op (discrete)
- reward = cost to supply electricity

---?image=/assets/repo.png&size=auto 100%&color=#ffffff

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

performance

- show picture of the tensorboard agent graph
- show learning curves (cartpole + battery + flex)


---?color=#000000

---

details

Note:
- space design is fundamental to the library 
- code that interacts both with agents and environments

---?color=#000000

```python
#  create an action space of two dimensions
action_space = GlobalSpace('action').from_spaces(
    [ContinuousSpace(0, 100), DiscreteSpace(3)],
    ['acceleration', 'gear']
)

#  randomly sample an action
action = action_space.sample()

#  check the action is valid
assert action_space.contains(action)

#  discretize the space
discrete_spaces = action_space.discretize(20)

#  randomly sample a discrete action
action = action_space.sample_discrete()

```
---?color=#000000

```python
#  load a state or observation space from a dataset
state_space = GlobalSpace('state').from_dataset('example')

# we can sample an episode from the state
episode = state_space.sample_episode(start=0, end=100)

# sample from the current episode by calling the space
state = state_space(steps=0)
```

---

lessons
- simplicity
- RL is hard and unstable
- small discount rate can help with Bellman blowups
- larger batch sizes

Note:
- two bad implementations don't equal one good one
- solving specific problems here - DQN works because of discretizable action spaces (don't combine high dimensional actions)
- can develop library to take advantage of it
- a master and dev branch
- single inheritance
- use standard library where possible 
- use tensorflow where possible (processors, schedulers etc)

---

three pieces of info on energy and reinforcement learning

---

the environment model problem / oppourtunity

Note:
- modern rl so sample inefficient that you need simualtion
but if you have simulation, then there are other better models such as MCTS

- the work in energy is therefore in building useful simulation models - this unlocks both

- Backwards induction = Allows measuring the quality of forecasts (when the model is wrong)

---?image=/assets/mcts_dqn.png&size=auto 50%&color=#ffffff

@snap[south]
Deep Reinforcement Learning Doesn't Work Yet - Alex Irpan
@snapend

---

synthetic data - aka poor mans GANS

Note:

- possible to generate synthetic rollouts for time series data (ie demands, prices etc)
- these synthetic rollouts allow testing of performance of the agent on rollouts it's never seen
- generating exact customer profiles is hard - generating believable profiles is easier

---

TODO

---

**short term work**

early stopping

+

experiment result analysis

+

backwards induction

---

**long term work**

wrapping other environments

+

model based methods 

+

integrating with `google/dopamine`

---

climate

1.
2.
3.

energy + rl

1. the environment model problem/opportunity
2. synthetic data to test generalization

thank you

