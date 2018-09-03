**energy_py**
<br><br>
lessons learnt building an energy reinforcement learning library
<br><br>

---

4 years energy engineer @ **ENGIE**
<br><br>
graduate & 2 years teaching reinforcement learning @ **Data Science Retreat**
<br><br>
data scientist @ **Tempus Energy**
<br><br>
blog at [www.adgefficiency.com](www.adgefficiency.com)

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

---?image=/assets/humber.jpg&size=auto 100%&position=center

---?image=/assets/humber_excel.png&size=auto 100%&color=#ffffff&position=center

---

computation to help solve the **control** problem
<br><br>
computation to help solve the **climate** problem

---?image=/assets/climate.png&size=auto 80%&color=#ffffff&position=center

---

price response flexible demand 
<br><br>
aka the lazy taxi driver


---?image=/assets/mdp.png&size=auto 80%&color=#ffffff&position=center

---?image=/assets/flex_env.png&size=auto 80%&color=#ffffff&position=center

---?image=/assets/repo.png&size=auto 60%&color=#ffffff&position=center

---

DQN + naive agents
<br><br>
energy environments + wrappers around `gym`
<br><br>
tools for experimentation

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

---

fun with cartpole

---?image=/assets/battery.png&size=auto 80%&color=#ffffff&position=bottom

@snap[north]
<font color="black">battery storage</font>
@snapend

---

details - space design

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

the environment model problem / opportunity

---?image=/assets/env_model.png&size=auto 30%&color=#ffffff&position=center

---?image=/assets/mcts_dqn.png&size=auto 70%&color=#ffffff&position=center

Note:
- UCT = upper confidence bound applied to trees

---?color=#000000

backwards induction

```python

previous_state_payoffs = {state: 0 for state in model.states}

for step in steps[::-1]:
    viable_transitions = [get_viable_transitions(step, next_state, model) 
                          for next_state in model.states]
    
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

---?image=/assets/world_models_lit.png&size=auto 30%&color=#ffffff&position=center

---?image=/assets/world_models_env.png&size=auto 70%&color=#ffffff&position=center

---?image=/assets/world_models.png&size=auto 70%&color=#ffffff&position=center

---?image=/assets/syn.png&size=auto 40%&color=#ffffff&position=center

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

---?image=/assets/dopamine.png&size=auto 80%&color=#ffffff&position=center

---

thank you
