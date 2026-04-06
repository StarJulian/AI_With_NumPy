"""
================================================================================
                        机器学习算法 - NumPy 实现
================================================================================
包含: 线性回归、逻辑回归、决策树、随机森林、SVM、朴素贝叶斯、
      K-Means、马尔可夫模型、HMM、KNN 等
================================================================================
"""

from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTree
from .random_forest import RandomForest
from .svm import SVM
from .naive_bayes import NaiveBayes
from .kmeans import KMeans
from .knn import KNN
from .markov import MarkovChain
from .hidden_markov import HiddenMarkovModel
from .pca import PCA
from .adaboost import AdaBoost
from .training_framework import TrainingFramework, Model, Trainer

__all__ = [
    'LinearRegression',
    'LogisticRegression', 
    'DecisionTree',
    'RandomForest',
    'SVM',
    'NaiveBayes',
    'KMeans',
    'KNN',
    'MarkovChain',
    'HiddenMarkovModel',
    'PCA',
    'AdaBoost',
    'TrainingFramework',
    'Model',
    'Trainer'
]
