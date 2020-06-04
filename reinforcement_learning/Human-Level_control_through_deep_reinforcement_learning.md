# Human-level control through deep reinforcement learning by Mnih et al. (2015)

[Link to the paper](https://www.nature.com/articles/nature14236)

## General Notes

This paper introduced Deep Q-Networks as they are known today. Here are some general notes I made while reading it:

* This is the first paper successfully dealing with high-dimensional input for agents without hand-crafted features;

* This paper is inspired by TD-Gammon and follows the approach of using non-linear function approximators, instead of the linear approximators that were popular due to better convergence guarantees;

* This is a model-free, off-policy method that uses non-linear function approximation. These characteristics usually result in problems in convergence;

## Method

The method is mostly based on Q-Learning, which learns Q values for state-action pairs. The main idea is to use a neural network to approximate the function that maps observation and actions to Q values. More specifically, the neural network minimizes the expected squared error from the predicted values $Q(s, a; \theta)$ to a target

$$\min_{\theta} \mathbb{E} \left[ \left( Q(s, a; \theta) - y\right)^2 \right],$$

where the target $y$ comes from the Bellman Optimality Equation for the state-action values

$$y = r + \gamma \max_{a'} Q(s, a'; \theta),$$

where $r$ is the reward obtained at that timestep and $\gamma$ is the discount factor. The Q values used when computing the target are estimates, which means that the targets are *bootstrapped*.

*Note: the expectation on the first equations samples transitions uniformly from a dataset, instead of depending on the dynamics on the environment and the previous policy. Because this is an off-policy method, it would be very difficult to implement the alternative.*

It has been observed that bootstrapping the targets based on the last parameters resulted in instability and failure to converge. The *Nature* version of this paper mentions that the parameters used when computing the targets are frozen and only updated every $C$ steps.

The experience replay introduced in this paper is a buffer of transitions observed by the agent. When a training step is taken, a batch is sampled uniformly from the buffer. The buffer might contain transitions experienced using an outdated policy, which makes this an off-policy approach. The uniformly sampling from a large buffer helps to break the correlations between transitions, which helps to approach the i.i.d. hypothesis of supervised learning.

## Details

The following points are details I found interesting:

* Formally, the state of the ALE is the whole sequence of frames and actions up to that point. The algorithm partially observes the state, receiving only a stack of the last four frames;

* Frame-skipping (that is, repeating the same action for a set amount of frames) saved them a lot of time while achieving state-of-the-art performance;

* Reward clipping is used, which means that all positive rewards are set to +1 and all negative rewards are set to -1;

* The paper states that the average episode return is a noisy metric that may not capture the progress that is made. They also state that tracking the mean Q value of the chosen actions is a smoother metric to track in training (that is, assuming the Q-network is not overestimating the values);

* Finally, I found it interesting that they cropped the frames to a square area because they were using an implementation of CNNs that only dealt with square inputs.
