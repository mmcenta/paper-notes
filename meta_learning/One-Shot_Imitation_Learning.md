# One-Shot Imitation Learning by Duan et al. (2017)

[Link to paper](https://arxiv.org/pdf/1703.07326.pdf)

## General Notes

* The paper proposes a way to learn block stacking tasks from a single demonstration using meta-learning. The model processes the demonstration into a context embedding, which then conditions the policy. The whole network is trained end-to-end using behaviour cloning and DAgger;

* Different tasks consist of different block arrangements e.g. the task "ab cd" represents stacking blocks "a" and "c" on top of "b" and "d", respectively. The method generalized to new unseen tasks;

* The proposed architecture has three main components: the Demonstration Network, the Context Network and the Manipulation Network. The first produces embeddings for the demonstration, the second combines the embeddings with the current state to form a context and the third uses the context and the current state to take actions;

* The tasks in theses papers are quite similar (all consist of repeteadly grasping and stacking blocks) and modular (the only difference is which blocks to stack). This may help in achieving good generalization;

* The model allows to visualize to which blocks in the demonstration the agent is paying attention at each timesteps. The results of these visualizations are quite intuitive and seem to indicate the agent pays attention to the blocks that it needs to stack next;

## Method

The proposed architecture consists of three modules:

* **Demonstration Network**: this module receives a demonstration (sequence of observations) and produces an embedding of it. The embedding's size grows linearly with the length of the demonstration and the number of blocks. Dilated Temporal Convolution is applied followed by a Neighborhood Attention operation (a attention mechanism which produces one fixed length output per input, which tends to all other inputs in relation to its corresponding input);

* **Context Network**: this module receives the current observation and the embedding produced by the Demonstration Network. First, it applies temporal attention over the demonstration embedding to reduce it to a single vector with size proportional to the number of blocks. Next, it applies attention over the current state such that the vectors in memory are the posistions of the blocks. The output is then appended to the current obvertation and then fed to the Manipulation Network. A key intuituition is that the number of *relevant* objects is small and usually fixed so that the context does not lose information by not growing in size with the total number of blocks;

* **Manipulation Network**: this module receives the context embedding and the current observation and returns the action that should be taken. Its architecture consists of a simple multi-layer perceptron.

## Results

The authors compare the given architecture trained with both behavior cloning and DAgger. They also ran experiments conditioning the policy to: the full demonstration (with dropout), a sequence of informative snapshots, and only the final frame (which should contain enough information to determine a task). Some results are:

* The agent trained with plain behaviour cloning seems to perform at the same level as the ones trained with DAgger. The authors believe it was due to the noise introduced artificially into the demonstrations which was enough to add sufficient variability to the data;

* Conditioning on the full demonstration seemed to best conditioning on snapshots and the final frame. The authors hypothesize that it may be due to the regularization offect that the dropout over observations introduces.

## Details

* Most (95%) of the demonstration is thrown out before feeding it to the model to reduce computing costs;

* Their use of attention mechanisms is quite clever. Is it necessary?
