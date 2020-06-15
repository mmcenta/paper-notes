# Fast Context Adaptation via Meta-Learning by Zintgraf et al. (2018)

[Link to paper](https://arxiv.org/abs/1810.03642)

## General Notes

* This paper builds on Model-Agnostic Meta-Learning (MAML) by proposing an architecture that reduces the number of parameters that are fine-tuned to obtain the task-specific parameters (similarly to the Meta-Learning with Latent Embedding Optimization paper). This is done by having a separate set of *context parameters $\phi$*, which are updated in the inner loop and *shared parameters* $\theta$, which are updated in the outer loop. The authors call this method *fast context adaptation via meta-learning* (CAVIA).

* The authors propose CAVIA as more interpretable and less prone to meta-overfitting than MAML. By controlling the set of parameters $\phi$ and $\theta$, one can decide how much network capacity to allocate for learning task-specific representations and for shared representations across all tasks.

## Method

CAVIA builds upon MAML; it consists of separating the parameters that are updated in the inner loop of MAML (context parameters $\phi$) from the ones that are updated on the outer loop (the shared parameters $\theta$). More specifically, $\phi$ is initialized to a constant $\phi_0$ and then learned via gradient updates (which conditions the context parameters on the shared parameters).

The authors opt to condition the network on $\phi$ by inserting the context parameters as input for some of the network layers. For a fully connected layer, a vector is appended to its input, where the vector is the context parameters, and the layer weights and biases are the shared parameters. For convolutional layers, *feature-wise linear modulation* (FiLM) is used to perform an affine transformation on the feature maps in which the transformation is conditioned on a vector of context parameters via a linear model. The context parameter is usually added at the first layer, i.e., the input.

Because of the choice of using $\phi$ as an input to the layers, a natural choice for $\phi_0$ is $\mathbf{0}$ since it does not affect the output of the layer. This doesn't reduce expressivity compared to learning the initialization $\phi$ because such initialization can be interpreted as a part of the bias parameter of the layer.

## Experiments

The experiments were set up to show three things:
Adapting a small number of input parameters is sufficient to yield to match the performance of MAML
CAVIA is robust to task-specific learning rate and scales well without overfitting
An embedding of the task emerges in the context parameters solely via backpropagation

For regression, the authors showed that CAVIA achieves performance that rivals MAML's with just a fraction of parameters being learned in the inner loop. They also observe that CAVIA's performance depends on having enough capacity to encode the task (it performed badly when using a single parameter on the sine wave regression task, with requires at least two parameters to be described). These experiments show that CAVIA is more stable than MAML, displaying a monotonic learning curve and stability to changes in the inner learning rate. In image completion, CAVIA also outperforms MAML with fewer learned parameters.

In classification, the authors benchmarked CAVIA in the Mini-Imagenet dataset. CAVIA bested the MAML model confortably. However, it wasn't able to beat the state-of-the-art results of VERSA and LEO (with the pre-trained deep residual network backend).

Finally, for reinforcement learning, the authors use a similar setup as in the original MAML paper. CAVIA outperforms MAML in the benchmarks. It also exhibits the property observed in the original MAML paper of continued learning for several gradient steps, even though it was trained to maximize return after only one step.

## Details

* The use of feature-wise linear modulation to add vectors as inputs to convolutional layers is a useful trick;

* The visualization of the learned context parameters on the sine wave regression experiment is an interesting way to interpret the embeddings learned by the method, clearly showing that it learned to recognize the phase and amplitude in this instance.
