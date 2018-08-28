## abstract

This talk reviews two years of work on energy_py - [a reinforcement learning for energy systems(https://github.com/ADGEfficiency/energy_py)].  We will look at lessons learned designing the library, experience using the library with Open AI gym and energy_py environments.  Also covered is the use of synthetic data generation in energy_py environments. 

energy_py = supporting experimentation 

contributions = naive agents, envs, tools for experiments

--- 

Space design is fundamental to the library 

- code that interacts both with agents and environments

--- 

simplicity

single agent

- two bad implementations don't equal one good one
- solving specific problems here - DQN works because of discretizable action spaces (don't combine high dimensional actions)
- can develop library to take advantage of it

a master and dev branch

single inheritance

Use standard library where possible 

Use tensorflow where possible (processors, schedulers etc)

Full docstrings are optional
Defined if needed, otherwise rely on the infomation about a variable type being visible in the code (ie from being used in a function, having a method called on it etc)

---

1. the environment model problem / oppourtunity

modern rl so sample inefficient that you need simualtion
but if you have simulation, then there are other better models such as MCTS

the work in energy is therefore in building useful simulation models - this unlocks both

Backwards induction = Allows measuring the quality of forecasts (i.e. - when the model is wrong)

---

2. synthetic data - aka poor mans gans

key idea - we learn behaviour that will generalize / fighting the generalization problem earlier

this is a subtle point - that you want to hve ways to estimate your error accurately
You don't care about the actual accuracy - getting a high test set error is useful feedback to learn from

rl is careless about the test/train problem

in energy we can test it specifically by keeping time series data separate

generating exact customer profiles is hard.  generating believeable ones is easier

---

3. combination with supervised learning

if you use the input from a supervised model as observation for rl
you should only use the test set press
this limits the data you can train an rl agent with

ie use time series forecasting to reduce dimensionality of the observation
because rl tuning process is longer + runs over mutltiple random seeds -> want to keep the supervised separate
