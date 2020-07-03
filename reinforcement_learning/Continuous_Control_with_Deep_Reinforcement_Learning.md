# Continuous Control with Deep Reinforcement Learning by Lillicrap et al. (2016)

[Link to paper](https://arxiv.org/abs/1509.02971)

## General Notes

* This paper introduced the deep deterministic policy gradient (DDPG), an off-policy algorithm for environments with continuous action spaces. DDPG can be thought of as a deep Q-networks (DQNs) for continuous action spaces because it trains an approximator for the Q function similarly to DQNs. Because the action space is continuous, solving the Q values' maximization over all actions (which is necessary for Q learning) is difficult. DDPG solves this problem by simultaneously learning to estimate both the Q-values of observation-action pairs and the action that maximizes the current Q function;

* The policy of DDPG is trained via gradient ascent on the learned Q value function. The Q-network parameters are treated as constants, and the optimization is done on the policy parameters;

* Since DDPG learns a deterministic policy, noise is added to the action to promote exploration. The authors use time-correlated OU noise, but recent implementations opt for uncorrelated zero-mean Gaussian noise;

## Method

There are two parts of DDPG: Q-Learning and Policy Learning. The Q-Learning part is identical to the DQN algorithm except for the maximization operators' substitution by the policy network. It is important to note that the policy network parameters are treated as constants and are not updated during this step. The authors employ replay buffers and target networks with Polyak averaging (for both the Q-network and the policy network) as in the DQN paper (Mnih et al., 2013) to achieve stable convergence.

The policy learning part of DDPG involves learning a deterministic policy $\mu_\theta(s)$. A deterministic policy outputs only an action, instead of a distribution over the action space. Since the action space is continuous, we can assume that the $Q$-function is differentiable with respect to the action, which means this maximization can be done via gradient ascent:

$$\max_\theta \mathbb{E}_{s \sim \mathcal{D}} \left[ Q_\phi (s, \mu_\theta (s)) \right].$$

Similarly to the Q-learning step, the parameters of the $Q$-function are considered constant, and this maximization procedure does not update them.

## Details

* OpenAI's implementation of DDPG uses a trick to improve exploration at the start of training. For a fixed number of steps at the beginning, the agent takes actions sampled from a uniform random distribution over valid actions. After that, it returns to normal exploration.