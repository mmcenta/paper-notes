# High-Dimensional Continuous Control Using Generalized Advantage Estimation by Schulman et al. (2016)

[Link to paper](https://arxiv.org/abs/1506.02438)

## General Notes

* This paper is known for Generalized Advantage Estimation (GAE), which is a method that allows to balance bias and variance in estimating the advantage funtion through two hyperparameters. This paper doesn't actually introduce the formula for the estimator - which was proposed in earlier work - but it does feature a novel generalized analysis which enables GAE to be used with a more general set of algorithms;

* Aside from GAE, the paper also introduces a trust region optimization procedure for the value function. The policy is updated using trust region policy optimization (TRPO), which was introduced in a previous paper;

* The paper has a clear and concise introduction to policy gradients and techniques used to balance bias and variance when estimating the expected gradient that is used to update the model;

## Method

### Generalized Advantage Estimation (GAE)

GAE estimates the advantage function, which in intuitive terms measures the difference between the expected returns from a taking a specific action and the average over all possible actions. If the advantage of a state-action pair is positive, that action is better than average for that state. Formally, we write the advantage in function of the state-action value and the state value:

$$A^{\pi, \gamma}(s_t, a_t) = Q^{\pi, \gamma}(s_t, a_t) - V^{\pi, \gamma}(s_t),$$

where $\pi$ is the policy, $\gamma$ is the discount factor, $s_t$ and $a_t$ is the state and the chosen action at timestep $t$, respectively. Define $\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$ with $r_t$ the reward at timestep $t$, i.e. the TD-residual of the estimate $V$ of $V^{\pi, \gamma}$. We can then generalize this and estimate the advantage using $k$-step estimates of the returns:

$$\hat{A}_t^{(k)} := \sum_{l=0}^{k-1} \gamma^l \delta^V_{t+l} = -V(s_t) + r_t +\gamma r_{t+1} + \dots + \gamma^{k-1}r_{t+k-1} + \gamma^k V(s_{t+k}) $$

As $k$ increases, the bias introduced by a approximate value function $V$ is decreased, but the variance of the estimate increases significantly. Notice that the discount $\gamma$ also introduces bias in relation to the original undiscounted formulation used in the paper. GAE is simply the exponentially weighted average of the estimators $\hat{A}_t^{(k)}$:

$$\hat{A}_t^{\mathrm{GAE}(\gamma, \pi)} = (1 - \lambda) \left(\hat{A}_t^{(1)} + \lambda \hat{A}_t^{(2)} + \lambda^2 \hat{A}_t^{(2)} + \dots \right)= \sum_{l=0}^\infty (\gamma\lambda)^l \delta^V_{t+l}.$$

Recall that $\delta^V_{t+l}$ contais the hyperparameter $\gamma$ in its formula. This formulations allows to balance the bias/variance trade-off of the different $k$-step estimators using a continuous parameter in the $[0, 1]$ range.

The paper also interprets this new weight parameter $\lambda$ as an extra discount factor applied after performing a reward shaping transformation on the MDP, which I will not reproduce here. They arrive at the interpretation that GAE is equivalent to shaping the reward function to reduce the temporal extent of the response function (which measures the temporal dependencies between actions and rewards) and the introduce a steeper discount to cut off noise arising from long delays.

### Trust-Region Optimization for Value Function Estimation

Instead of treating the estimation of the value function as a simple nonlinear regression problem, the authors opt to use a trust region method in each iteration of a batch optimization procedure. They hope that this helps to avoid overfitting to the most recent batch of data. Let $\hat{V}_t = \sum_{l=0}^\infty \gamma^l r_{t+l}$ be the discounted sum of rewards over a collected trajectory. To formulate the problem, compute $\sigma^2 = \frac{1}{N} \sum_{n=1}^N \|V_{\phi_{\text{old}}}(s_n) - \hat{V}_n\|^2$. Then, solve the following constrained optimization problem:

$$\textrm{minimize}_\phi \sum_{n=1}^N \| V_\phi(s_n) - \hat{V}_n \|^2$$

$$\textrm{subject to} \frac{1}{N} \sum_{n=1}^N \frac{\| V_\phi(s_n) - V_{\phi_\text{old}}(s_n) \|}{2 \sigma^2} \leq \epsilon.$$

This is equivalent to constraining the average KL divergence between the previous value function and the new one to be smaller than $\epsilon$, assuming that the value function parametrizes a Gaussian distribution of mean $V_\phi(s_n)$ and variance $\sigma^2$. A approximate solution is computed to this problem is computed using the conjugate gradient method.

## Policy Optimization

This paper uses the trust region policy optimization algorithm to compute policy updates.

## Results

The authors test the algorithm in a series 3D locomotion tasks in which it achieve state-of-the-art results.