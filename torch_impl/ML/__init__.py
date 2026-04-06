"""
================================================================================
                    PyTorch 机器学习模块
================================================================================
使用 PyTorch 实现各种机器学习算法
================================================================================
"""

from .linear_regression_torch import LinearRegressionTorch
from .logistic_regression_torch import LogisticRegressionTorch
from .training_framework_torch import MLPTrainer, train_model, evaluate_model

__all__ = [
    'LinearRegressionTorch',
    'LogisticRegressionTorch', 
    'MLPTrainer',
    'train_model',
    'evaluate_model'
]
