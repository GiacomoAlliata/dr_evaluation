# Dimensionality Reduction Evaluation

This research is part of my Doctoral Thesis conducted at the Laboratory for Experimental Museology, within the *Narratives from the Long Tail: Transforming Access to Audiovisual Archives* project. 

The goal of this repository is to compare and evaluate, both qualitatively and quantitatively, four Dimensionality Reduction (DR) algorithms:
* tSNE (t-distributed Stochastic Neighbor Embedding): https://scikit-learn.org/0.16/modules/generated/sklearn.manifold.TSNE.html
* UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction): https://github.com/lmcinnes/umap
* TriMap: https://github.com/eamid/trimap
* PaCMAP (Pairwise Controlled Manifold Approximation): https://github.com/YingfanWang/PaCMAP

The evaluations are run on four cultural collections part of the *Narratives* project that cannot be disclosed for copyright reasons. The code, however, is provided here with the MNIST dataset, commonly used for the evaluation of machine learning algorithms.

## Notebook description
* dr_mapping: run the DR algorithms on the datasets loaded, with the option to also run the evaluation
* dr_eval_analysis: analyse the results of the evaluation for the four DR algorithms on each dataset
* dr_params_study: analyse the effect of the hyper-parameters of the four DR algorithms on each dataset
* dr_computational: evaluate the computational efficiency and the stability of the four DR algorithms

## License

This work is shared under the terms of the MIT license.
