# Few-Shot Learning with Graph Neural Networks by Garcia et al. (2017)

[Link to paper](https://arxiv.org/abs/1711.04043)

## General Notes

* This paper explores the use of Graph Neural Networks (GNNs) to the few-shot classification setting, which matches the state-of-the-art with fewer parameters. They also explore the semi-supervised and active learning variants of this problem. In these formulations, there are unlabeled images in the support set (for which the model can request labels, in the active learning formulation);

* The method proposed encodes all input instances (both labeled and unlabeled) into a fully connected graph. The edge weights at each layer of the GNN are learned similarity measures.

* The graph-based formulation of the problem, which draws from message passing algorithms, allows for **effortless extension to semi-supervised and active learning scenarios**.

* There is a section in this paper that analyses other architectures for few-shot classification (siamese networks, matching networks, and prototypical networks) as special cases of a GNN, determining which edge features are used in each of the methods.

## Method

Because GNNs operate on graphs, it is first necessary to cast the few-shot classification episode into a graph structure. All instances (both labeled and unlabeled) are mapped into nodes of a graph. The node features are encoding of the image concatenated with a representation of the labels. In the mini-Imagenet experiment, a Convolutional Neural Network (CNN) encodes the images into a 128-dimensional embedding, and in the Omniglot experiment, images are flattened. When possible, the label is represented via a one-hot encoding. If there is no label available, then a vector with equal probability for all classes is used.

The graph is fully connected, and the weight between the two nodes is a similarity measure between their features. This measure is not fixed but instead learned using a parametric model. In this paper, the similarity is modeled with a Multi-Layer Perceptron (MLP), which takes the element-wise absolute value on the difference of the node features as input.

Next, a brief primer on GNNs. Let $V$ be the number of nodes and $\mathbf{x}^{(k)} \in \mathbb{R}^{V \times d_k}$ be the node features at layer $k$, then a GNN layer can be described as the following operation:

$$\mathbf{x}^{(k+1)}_l = \rho \left( \sum_{B \in \mathcal{A}} B \mathbf{x}^{(k)} \theta^{(k)}_{B, l} \right), l = 1, \dots, d_{(k+1)}$$

where $\mathcal{A}$ is a set of linear operators on the graph and $\{ \theta^{(k)}_1, \dots, \theta^{(k)}_{|\mathcal{A}|} \}, \theta^{(k)}_B \in \mathbb{R}^{d_k \times d_{k+1}}$ are learned parameters. In this paper, $\mathcal{A}^{(k)} = \{ \mathbf{1}, \hat{A}^{(k)}\}$ and $\hat{A}^{(k)}$ is the adjacency operator based:

$$\left( \hat{A}^{(k)} \mathbf{x} \right)_i = \sum_{i \sim j} w_{i, j} \mathbf{x}_j,$$

where $\rho$ is a element-wise non-linearity, $i \sim j$ if there is an edge between nodes $i$ and $j$ and $w_{i, j}$ is its corresponding weight. The weights determined by the learned similarity measure described previously.

The final architecture consists of three of the blocks described above (with additional batch normalization, etc.) and possibly an encoder for generating image embeddings. The final GNN layer outputs a probability distribution over the labels for each node. For supervised and semi-supervised learning, the model is trained by backpropagating the cross-entropy loss of the distribution of nodes in the test set. For active learning, an additional attention mechanism is added to decide which instances the labels will be requested.

## Experiments

Even though it is a simple approach with fewer parameters, the proposed model outperformed several previous state-of-the-art methods in both the Omniglot and the mini-Imagenet datasets. The authors also added an "Our metric learning + KNN" baseline, which learns the similarity measure as outlined above, which is then used by a K-nearest neighbors classifier. This baseline is surprisingly strong, being competitive with methods such as prototypical networks.

The model was able to reach competitive results in the semi-supervised setting even while having access to only 20% of the labels. The authors also set up a comparison between the model trained on only the labeled examples of the support set versus the semi-supervised model. The latter is able to achieve better accuracies, but the gap narrows as the proportion of labeled examples increases. Finally, the active learning version was able to improve over randomly assigned labels.

## Details

* The authors use a lightweight 4-layer CNN as the encoder for their model, which is trained end-to-end with the model itself, instead of opting for a bigger pre-trained encoder.
