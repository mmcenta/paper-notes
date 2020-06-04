# Few-Shot Autoregressive Density Estimation: Towards Learning to Learn Distributions by Reed et al. (2017)

[Link to paper](https://arxiv.org/abs/1710.10304)

## General Notes

* This paper expands on deep autoregressive models for density estimation by adding 1) neural attention and 2) meta-learning to enable few-shot density estimation;

* The main goal of density estimation is to generate new examples similar to the ones which were observed;

* The tasks the models are trained on are flipping images and few-shot generation of new examples. The datasets used were Omniglot, Stanford Online Product and ImageNet.

## Method

* The paper uses autoregressive networks, which factorize the joint distribution by conditioning the probability of the next element on the previous generated elements (here an element is a pixel, for example). Note that a ordering of elements is imposed when using this method;

* All these methods are based on the PixelCNN architecture proposed in a previous paper, which employs a few tricks which I will not mention here;

* The PixelCNN method consists of encoding the input samples with a standard neural network (or convolutional neural network), which was the method used on the original PixelCNN paper;

* The Attention PixelCNN consists of instead encoding the input samples using an attention mechanism. The key intuition is that different aspects of the support (input) images are relevant depending on which pixel is being generated. The support images are then augmented with positional features ranging from -1 to 1 that encode the position of that pixel on the image. These features are used by the model to query memory (using attention) when generating a new pixel. The memory contents that can be queried are global features from the support images and textures from image patches - I will leave specifics out of these notes;

* Finally, the Meta PixelCNN uses MAML to learn initial parameters that can be fine-tuned on the input samples efficiently. In this case, the fine-tuning consists of a single gradient step. It is important to notice that the conditioning (i.e. the flow of information from the supports to the next pixel) does not introduce new parameters. An interesting choice was that the inner loss (the loss used for fine-tuning) was a learned function instead of the negative log-likelihood. This loss is only used to direct gradient steps during fine-tuning.

## Results

* On the image flipping task, the attention model was able to consistently learn the operation (although imperfectly) and the other models were unable to learn it;

* On the Omniglot few-shot generation task, all architectures produced state-of-the-art likelihood, but the results for the attention model were significantly better;

* On the Stanford Online Products dataset, the attention model did produce slightly better results qualitatively. However, quantitatively there were no significant improvements over the PixelCNN baseline.

## Details

* The paper mentions that autoregressive neural networks were used because they have "tractable likelihoods", among other reasons. I assume this means that the way the join distribution is factorized allows for efficient computation of the likelihood, which is helpful to compare different models;

* According to the authors, cursory experiments with more gradient steps in MAML didn't show promising results and significantly slowed training, which is why only one step is used on the published work;

* The positional encodings on the Attention PixelCNN might explain why it did a much better job flipping images than the other models (which did not have access to these features);

* The learned inner loss is an important element of this paper. It is fed some high level features and learns to compute a function which helps directing the gradients to the right direction in fine-tuning;

* Finally, the authors did explore naive combinations of attention mechanisms and meta-learning but it didn't seem to help in their experience.