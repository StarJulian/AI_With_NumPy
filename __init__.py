"""
================================================================================
                    机器学习算法 - NumPy 从零实现
================================================================================
包含: 线性回归、逻辑回归、决策树、随机森林、SVM、朴素贝叶斯、
      K-Means、马尔可夫模型、HMM、KNN、PCA、AdaBoost 等

每个算法都包含:
    - 训练代码 (fit)
    - 推理代码 (predict/predict_proba)
    - Demo 示例 (if __name__ == "__main__")
================================================================================
"""

from ML.linear_regression import LinearRegression
from ML.logistic_regression import LogisticRegression
from ML.decision_tree import DecisionTree
from ML.random_forest import RandomForest
from ML.svm import SVM
from ML.naive_bayes import NaiveBayes
from ML.kmeans import KMeans
from ML.knn import KNN
from ML.markov import MarkovChain
from ML.hidden_markov import HiddenMarkovModel
from ML.pca import PCA
from ML.adaboost import AdaBoost

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
]
