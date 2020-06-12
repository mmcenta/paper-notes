# Meta-Learning with Latent Embedding Optimization by Rusu et al. (2018)

[Link to paper](https://arxiv.org/abs/1807.05960)

## General Notes

* The main idea of this paper is to learn a low-dimensional latent embedding of the model parameters and perform optimization-based meta-learning on this space;

* In summary, the training examples are fed to an encoder which produces the parameters of a distribution. A latent vector is sampled from this distribution and then decoded to generate the parameters for the model. Once the parameters are produced, MAML is used to fine-tune to parameters optimizing in the latent space, leaving encoder and decoder parameters untouched. Once the task-specific parameters are obtained, both the decoder and encoder are updated in the outer loop of MAML;

## Method

First, training examples are fed to a stochastic encoder that produces a low-dimensional embedding of the model parameters $z$. Then, a decoder maps $z$ to the model parameters $\theta$. Finally, the task-specific parameters $\phi$ are obtained by differentiating the model loss with respect to $z$. The parameters for the encoder and the decoder are only updated in the outer loop of MAML.

For the following summaries, I consider the few-shot classification architecture. However, it is important to note that this framework can be adapted to a variety of settings - including few-shot regression, which is done in the paper.

### Encoder

Aside from encoding each data point, the decoder also features a relation network that considers the pair-wise relationship between the data in the task instance. The relation network outputs a set of parameters for a multivariate Gaussian distribution with diagonal covariance for each class. Samples for each distribution are concatenated to form the latent embedding $z$.

### Decoder

In practice, the parameters $\theta$ that the decoder outputs represent the top layer weights of the classifier instead of all parameters of the model. The authors use a linear softmax classifier for each class in which parameters are decoded from the corresponding class embedding.

## Results

Asides from achieving state-of-the-art results in few-shot classification benchmarks, there is evidence that this method also captures some of the uncertainty of ambiguous problem instances and that it can learn multimodel task distributions. These last characteristics are inferred from the sine wave/line regression task that is featured on several papers in Bayesian meta-learning.