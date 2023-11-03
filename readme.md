# Positive and unlabelled machine learning reveals new fast radio burst repeater candidates

This repository contains the code used to investigate the application of positive and unlabeled (PU) techniques to identify repeater candidates in the [CHIME/FRB](https://www.chime-frb.ca/) 2021 and 2023 catalogs. It is an extension of a previously prepared paper (the code for which is [here](https://github.com/ArjunS07/frb-puml/)), with newer data, expanded analysis, and more extensive comparison with previous results.

## Credits

* For feature extraction, we use a modified version of the code provided by Zhu-Ge et al. for their paper _[Machine learning classification of CHIME fast radio bursts: II. Unsupervised Methods](https://arxiv.org/abs/2210.02471)_. Their original code can be found [here](https://github.com/JiamingZhuge/FRB_ML_unsp/tree/main).
* Classic Elkanoto, Weighted Elkanoto, and Bagging PU are implemented with the [`pulearn` library](https://pulearn.github.io/pulearn/)
* Modified logistic regression, as described by Jasky et al. in _[A modified logistic regression for positive and unlabeled learning](https://ieeexplore.ieee.org/document/9048765)_,  is implemented with a [custom fork](https://github.com/ArjunS07/pu_modified_lr) of the original [repository](https://github.com/kpjaskie/Positive_Unlabeled) provided by the authors.
* PUExtraTrees, as described in [Positive-Unlabeled Learning using Random Forests via Recursive Greedy Risk Minimization](https://arxiv.org/pdf/2210.08461.pdf), is implemented with a [custom fork](https://github.com/ArjunS07/PUExtraTrees) of the original [repository](https://github.com/jonathanwilton/PUExtraTrees) provided by the authors.

## Bug-reporting

Report bugs in the code by opening a new issue in this GitHub repository.
