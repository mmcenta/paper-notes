# Massively Multitask Networks for Drug Discovery by Ramsundar et al. (2015)

[Link to paper](https://arxiv.org/abs/1502.02072)

## General Notes

* This paper proposes leveraging multi-task learning for hit-finding, which consists of predicting interactions between targets and other molecules;

* Tasks consists of different virtual screening datasets to predict interactions between different types of molecules, such as predicting toxicity. In total, there are 259 datasets containing 37.8M experimental data points for 1.6M compounds;

* Datasets for hit-finding are very imbalanced (1-2% of the tested compounds are active against a given target). The authors use ROC AUC to evaluate the performance of the models;

* A strong point of this paper is their experimental set up: they investigate different hypothesis that explain the performance gain of their method, instead of simply presenting the method and results.

## Method

Their choice of architecture is quite simple: their neural networks consist of multiple common fully-connected layers and a softmax classifier on top for each task. They experiment with a pyramidal network architecture that starts very wide (more neurons than input dimensions) and quickly becomes narrow, which achieved better results than other architectures once a significant dropout (25%) was added.

## Results

This paper uses cross-validation to produce test results. The experiments are set up to answer four questions:

1. Do massively multitask networks improve performance over simple machine learning methods? Yes, in this case at least. The massively multitask neural networks proposed bested every single-task method in their experiments;

2. How does the performance of a multitask network depend on the number of tasks/ammount of data? Performance increases in both but it is clear that the gains start to plateau. Results indicate that training on more tasks helps significantly, with some instances in which doubling the number of tasks (but keeping the number of training examples constant) results in a performance gain comparable to doubling the number of training examples;

3. Do massive multitask networks extract generalizable information about the data? They held out tasks from the training set to evaluate generalization to new tasks. In some datasets, using this initialization resulted in worse performance and most datasets didn't benefit significantly from training on oher tasks;

4. When do datasets benefit from multitask learning? Here the paper delves into specifics to analyze which datasets benefited the most from multitask learning and look for similarities between the tasks. They hypothesize that the extent of the generalizability is determined by the presence of absence of relevant data in the multitask training set.

## Details

* This paper features some creative and informative plots!

* Molecules are featurized using extended connectivity fingerprints (ECFP4), which seems to be the standard for this kind of task.
