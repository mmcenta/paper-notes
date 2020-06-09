# Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks by Finn et al. (2017)

[Link to paper](https://arxiv.org/abs/1703.03400)

## General Notes

* The authors propose a meta-learning framework that is compatible with any model trained with gradient descent which makes it applicable to a variety of learning tasks;

* The method consists of finding a set of parameters that can be fine-tuned with few gradient steps to produce good results;

* An interesting discovery is that framing the problem this way yields better initial parameters than direct transfer learning;

## Method

### The Algorithm

Let $f_\theta$ be the model, $\tau_i \sim p(\tau)$ a specific task from a distribution over tasks and $\mathcal{L}_i$ the loss associated with this task. The Model-Agnostic Meta-Learning (MAML) approach consists of fine-tuning the prior parameters $\theta$ to obtain the task-specific parameters $\phi_i$ via a optimization procedure - usually a few steps of gradient descent. For example, the fine-tuning for one step of gradient descent can be written as:

$$\phi_i = \theta - \alpha \nabla_\theta \mathcal{L}_i (f_\theta).$$

It is important to note that the loss is evaluated over a few train examples of the task $\tau_i$. The step size $\alpha$ can be either fixed or meta-learned. The prior parameters $\theta$ are then tuned by minimizing the loss using the task-specific parameters $\phi_i$ on a few test examples of the same task. When using batch gradient descent, this can be written as:

$$\theta = \theta - \beta \nabla_\theta \sum_{\tau_k \sim p(\tau)}\mathcal{L}_k (f_{\phi_k}).$$

This procedure is called the meta-gradient update. It involves computing a gradient through a gradient, which is supported by most deep learning libraries but results in additional computational costs. The paper also explores the possibility of using a linear approximation of the meta-gradient.

### Species of MAML

In supervised learning, MAML is applied to $K$-shot learning by fine-tuning the parameters on $K$ input/output pairs for regression or $K$ pairs per class for classification. These pairs for the training set for a specific task at meta-train time. Then, the prior parameters are updated by backpropagating the loss of the task-specific model on held-out examples of the same task *with respect to the prior parameters*. The model is evaluated on held-out *tasks* (as opposed to examples) at what is called meta-test time.

In reinforcement learning, different consist of either different goals in the same environment or entirely different environments (with possibly similar goals). In this case, the model being learned is a policy $\pi_\theta$ that maps observations to a distribution over actions. In $K$-shot reinforcement learning, $K$ rollouts for a task $\tau_i$ are obtained using the policy $\pi_\theta$ (using the prior parameters). The collected rollouts are then used to obtain the task-specific parameters, which are then used to compute the final loss and update the prior parameters.

## Results

The authors first test the proposed framework on the simple regression task of fitting a sine wave. Each task consists of a sine wave with different amplitude and frequency, and the data points are points on the curve. They compare the MAML approach to a baseline pretrained on the entire dataset of tasks and then fine-tuned at test time. The experiments showed that MAML obtained much better results than the baseline and that taking more gradient steps improves the performance, even though the model was trained for maximal performance after a single gradient step.

They apply the method to few-shot classification and achieve state-of-the-art performance on the Omniglot and MiniImagenet datasets.

For reinforcement learning, vanilla policy gradient updates are used in the fine-tuning, and Trust Region Policy Optimization (TRPO) is the meta-optimizer. They achieve much better results with MAML than with policies fine-tuned from pretrained/random weights.