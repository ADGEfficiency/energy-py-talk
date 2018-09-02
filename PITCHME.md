**energy_py**
<br><br>
lessons learnt building an energy reinforcement learning library
<br><br>
www.adgefficiency.com

---

**agenda**
<br><br>
motivation
<br><br>
API + performance
<br><br>
lessons
<br><br>
next steps

---?image=/assets/humber.jpg&size=auto 100%

Note:
- two years ago I made the transition into data
- three years before that I made the transition from uni into energy

- this place was where I learnt energy engineering

---?image=/assets/humber_excel.png&size=auto 100%&color=#ffffff

Note:
- the reason i introduce this place is because of the MILP model I built to optimize this site

- this was where I learn the power of a computer

---

computation to help solve the **control** problem
<br><br>
computation to help solve the **climate** problem

---?image=/assets/climate.png&size=auto 80%&color=#ffffff

Note:

Today control is done using heuristics and rules.  Give example of price of biomass versus gas fired CHP (red duos).

1. it's bad - and worse for poor countries
2. technology is not the only solution - business models, policy, public opinion
3. your personal choices matter

- RL = solve the control problem
- digitization challenge in energy

---

price response flexible demand 
<br><br>
aka the lazy taxi driver

Note:

- taxi system where government ensures that no one ever waits for a taxi
- supply of taxis == peak demand for taxis
- cost = fixed + variable
- the laziest taxi drivers need to recover their entire fixed + var costs through a few trips
- flexibility solves the problem by letting people wait
- responding to price (not signal from system operator)
- need wholesale market

---?image=/assets/mdp.png&size=auto 80%&color=#ffffff

---

---?image=/assets/flex_env.png&size=auto 80%&color=#ffffff

Note:
- state = current + forecast electricity prices, customer demands
- action = flex up, flex down or no-op (discrete)
- reward = cost to supply electricity
- well defined reward signal

---?image=/assets/repo.png&size=auto 100%&color=#ffffff

---

DQN + naive agents
<br><br>
energy environments + wrappers around `gym`
<br><br>
tools for experimentation

Note:
- DQN becuase of discretizable action spaces
- experimentation because this is what needs to be done


---?color=#000000
High level API

```bash
$ cd energy_py/experiments

$ python experiment.py example dqn

$ tensorboard --logdir='./results/example/tensorboard'
```

---?color=#000000
Low level API

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
        total_steps=1000000,
        batch_size=1024,
    )
```

---?image=/assets/tb.png&size=auto 80%&color=#ffffff

---?image=/assets/cartpole.png&size=auto 80%&color=#ffffff&position=bottom

@snap[north]
<font color="black">Performance on Cartpole</font>
@snapend

---?image=/assets/battery.png&size=auto 80%&color=#ffffff&position=bottom

@snap[north]
<font color="black">Battery storage</font>
@snapend

---

details

Note:
- two bad implementations don't equal one good one
- solving specific problems here - DQN works because of discretizable action spaces (don't combine high dimensional actions)
- can develop library to take advantage of it
- a master and dev branch
- single inheritance
- use standard library where possible 
- use tensorflow where possible (processors, schedulers etc)

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

**rl today**
<br><br>
hard
<br><br>
unstable 
<br><br>
sample inefficient
<br><br>
**exciting**

---

the environment model problem / oppourtunity

PUT IN BIT ABOUT WHAT A MODEL IS sa -> r, s'

Note:
- modern rl so sample inefficient that you need simualtion
but if you have simulation, then there are other better models such as MCTS

- the work in energy is therefore in building useful simulation models - this unlocks both

- Backwards induction = Allows measuring the quality of forecasts (when the model is wrong)

---?image=/assets/mcts_dqn.png&size=auto 80%&color=#ffffff

Note:
- UCT = upper confidence bound applied to trees

---?color=#000000

backwards induction

```python

previous_state_payoffs = {state: 0 for state in model.states}

for step in steps[::-1]:
    viable_transitions = [get_viable_transitions(step, next_state, model) for next_state in model.states]
    
    new_state_payoffs = defaultdict(list)
    
    for t in viable_transitions:
        payoff = t.reward + previous_state_payoffs[t.next_state]
        new_state_payoffs[t.state].append(payoff)
        
    for k, v in new_state_payoffs.items():
        new_state_payoffs[k] = np.max(v)
    
    previous_state_payoffs = new_state_payoffs
```
---

synthetic data - aka poor mans GANS

Note:

- possible to generate synthetic rollouts for time series data (ie demands, prices etc)
- these synthetic rollouts allow testing of performance of the agent on rollouts it's never seen
- generating exact customer profiles is hard - generating believable profiles is easier

---?image=/assets/syn.png&size=auto 50%&color=#ffffff

---

**long term work**
<br><br>
a fourth complete rebuild
<br><br>
wrapping other environments
<br><br>
model based methods 
<br><br>
integrating with `google/dopamine`

---

thank you
