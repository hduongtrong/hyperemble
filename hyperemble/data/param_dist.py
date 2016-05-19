from __future__ import print_function, absolute_import, division

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

n = 1000
p = 100
n_jobs = 1
seed = 1

classification_models = [
    KNeighborsClassifier,
    LogisticRegression,
    GaussianNB,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    LinearSVC,
    SVC,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
]

default_params = {
    "KNeighborsClassifier": {
        "n_neighbors": 5,
        "weights": 'uniform',
        "n_jobs": 1,
    },
    "LogisticRegression": {
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "penalty": "l2",
        "C": 1.0,
    },
    "GaussianNB": {
    },
    "RandomForestClassifier": {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": 8,
        "bootstrap": True,
        "random_state": seed,
        "verbose": 0,
        "n_jobs": -1,
    },
    "ExtraTreesClassifier": {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": 8,
        "bootstrap": True,
        "random_state": seed,
        "verbose": 0,
        "n_jobs": -1,
    },
    "GradientBoostingClassifier": {
        "loss": "deviance",
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "max_depth": 6,
        "random_state": seed,
        "max_features": 10,
        "verbose": 0,
    },
    "LinearSVC": {
        "penalty": "l2",
        "loss": "hinge",
        "C": 1.0,
        "verbose": 0,
        "random_state": seed,
        "multi_class": "ovr",
    },
    "SVC": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": .01,
        "random_state": seed,
        "verbose": 0,
    },
    "LinearDiscriminantAnalysis": {
        "solver": "lsqr",
        "shrinkage": "auto",
    },
    "QuadraticDiscriminantAnalysis": {
        "reg_param": .1,
    },
}

dist_params = {
    "KNeighborsClassifier": {
        "n_neighbors": np.arange(int(np.sqrt(n))),
        "weights": ['uniform', 'distance'],
        "n_jobs": -1,
        "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        "leaf_size": np.arange(1, 30)
    },
    "LogisticRegression": {
        "solver": "lbfgs",
        "multi_class": "multinomial",
        "penalty": "l2",
        "C": 1.0,
    },
    "GaussianNB": {
    },
    "RandomForestClassifier": {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": 8,
        "bootstrap": True,
        "random_state": seed,
        "verbose": 0,
        "n_jobs": -1,
    },
    "ExtraTreesClassifier": {
        "n_estimators": 500,
        "criterion": "gini",
        "max_depth": 8,
        "bootstrap": True,
        "random_state": seed,
        "verbose": 0,
        "n_jobs": -1,
    },
    "GradientBoostingClassifier": {
        "loss": "deviance",
        "learning_rate": 0.1,
        "n_estimators": 100,
        "subsample": 1.0,
        "max_depth": 6,
        "random_state": seed,
        "max_features": 10,
        "verbose": 0,
    },
    "LinearSVC": {
        "penalty": "l2",
        "loss": "hinge",
        "C": 1.0,
        "verbose": 0,
        "random_state": seed,
        "multi_class": "ovr",
    },
    "SVC": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": .01,
        "random_state": seed,
        "verbose": 0,
    },
    "LinearDiscriminantAnalysis": {
        "solver": "lsqr",
        "shrinkage": "auto",
    },
    "QuadraticDiscriminantAnalysis": {
        "reg_param": .1,
    },
}
