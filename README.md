# Dimensionality Reduction Evaluation

This repository is the supporting material for the submission "Visualising Culture at Scale: A Review of Dimensionality Reduction Algorithms for Cultural Data" for the Journal of Cultural Analytics. 

The goal of this repository is to compare and evaluate, both qualitatively and quantitatively, four Dimensionality Reduction (DR) algorithms:
* tSNE (t-distributed Stochastic Neighbor Embedding): https://scikit-learn.org/0.16/modules/generated/sklearn.manifold.TSNE.html
* UMAP (Uniform Manifold Approximation and Projection for Dimension Reduction): https://github.com/lmcinnes/umap
* TriMap: https://github.com/eamid/trimap
* PaCMAP (Pairwise Controlled Manifold Approximation): https://github.com/YingfanWang/PaCMAP

The evaluations are run on four cultural collections that cannot be disclosed for copyright reasons. For partial reproducibility, 2D embeddings computed with the different DR algorithms are provided.

## Installation

Run the following commands:
conda env create -f environment.yml
conda activate dr_eval
pip install -r requirements-pip.txt

## Notebook description
* dr_mapping: run the DR algorithms on the datasets loaded, with the option to also run the evaluation
* dr_eval_analysis: analyse the results of the evaluation for the four DR algorithms on each dataset
* dr_params_study: analyse the effect of the hyper-parameters of the four DR algorithms on each dataset
* dr_computational: evaluate the computational efficiency and the stability of the four DR algorithms
* dr_classes: specific case study on a subsample of sports videos to visually evaluate global structure preservation

## License

This work is shared under the terms of the MIT license.
