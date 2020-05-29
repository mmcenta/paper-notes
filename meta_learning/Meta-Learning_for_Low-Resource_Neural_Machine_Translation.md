# Meta-Learning for Low-Resource Neural Machine Translation by Gu et. al.

[Link to the paper](https://arxiv.org/abs/1808.08437)

## General Notes

* The idea behind this paper is to frame low-resource translation as a meta-learning problem and use the Model-Agnostic Meta-Learning (MAML) framework to learn to adapt to low-resource langueges based on high-resource language tasks;

* Their method outperforms direct transfer learning with only a fraction of the training examples. Their performance is competitive with the previous state-of-the-art for low-resource machine translation which is based on more conventional machine translation;

## Method

* The paper follows the framework of MAML with an explicit isotropic Gaussian prior (that is a $L2$-distance penalty between the initial parameters and the new parameters is added). They also use a linear approximation of the meta-gradient;

* They meta-train the model on 17 high resource languages and meta-test on 4 low-resource languages (which are simulated by subsampling);

* A major challenge with this method, however, is that MAML assumes the input and output spaces are shared across all source and target languages. This poses a problem because for example, when translating from Portuguese and French to English, the model has to map both the Portuguese word embedding for the word "cachorro" and the French word embedding for the word "chien" to the same English word "dog". The authors tackle this problem by proposing a Universal Lexical Representation (ULR). In ULR, words are encoded dynamically, using the monolingual embedding as input to a learned transformation to a universal embedding (more details in paper). This makes the embedding layer trainable.

## Results

* A surprising conclusion of the paper is that the best results were obtained when using meta-learning to initialize the parameters from the embedding and the encoder, letting the decoder be randomly initialized.
