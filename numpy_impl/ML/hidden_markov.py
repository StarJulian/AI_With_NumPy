"""
================================================================================
                        隐马尔可夫模型 - NumPy 实现
================================================================================
"""

import numpy as np
from typing import Optional, List, Tuple


class HiddenMarkovModel:
    """
    隐马尔可夫模型 (HMM)
    
    支持前向-后向算法和 Viterbi 算法
    """
    
    def __init__(self, n_states: int, n_observations: int,
                 max_iter: int = 100, tol: float = 1e-4, 
                 random_state: Optional[int] = None):
        self.n_states = n_states
        self.n_observations = n_observations
        self.max_iter = max_iter
        self.tol = tol
        
        if random_state:
            np.random.seed(random_state)
        
        self.initial_probs_: Optional[np.ndarray] = None
        self.transition_matrix_: Optional[np.ndarray] = None
        self.emission_matrix_: Optional[np.ndarray] = None
        self.is_fitted = False
        self.log_likelihood_history: List[float] = []
    
    def _initialize_params(self):
        self.initial_probs_ = np.random.rand(self.n_states)
        self.initial_probs_ /= self.initial_probs_.sum()
        
        self.transition_matrix_ = np.random.rand(self.n_states, self.n_states)
        self.transition_matrix_ /= self.transition_matrix_.sum(axis=1, keepdims=True)
        
        self.emission_matrix_ = np.random.rand(self.n_states, self.n_observations)
        self.emission_matrix_ /= self.emission_matrix_.sum(axis=1, keepdims=True)
    
    def _forward(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        alpha[0] = self.initial_probs_ * self.emission_matrix_[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_matrix_[:, j]) * \
                             self.emission_matrix_[j, observations[t]]
        
        log_prob = np.log(np.sum(alpha[-1]) + 1e-300)
        return alpha, log_prob
    
    def _backward(self, observations: np.ndarray) -> np.ndarray:
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        beta[-1] = 1.0
        
        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix_[i] * 
                    self.emission_matrix_[:, observations[t + 1]] * 
                    beta[t + 1]
                )
        return beta
    
    def fit(self, X: List[np.ndarray]) -> 'HiddenMarkovModel':
        print(f"\n{'='*60}")
        print("训练隐马尔可夫模型")
        print(f"{'='*60}")
        
        self._initialize_params()
        
        for iteration in range(self.max_iter):
            total_log_prob = 0
            sum_init = np.zeros(self.n_states)
            sum_trans = np.zeros((self.n_states, self.n_states))
            sum_emit = np.zeros((self.n_states, self.n_observations))
            
            for observations in X:
                observations = observations.astype(int)
                observations = np.clip(observations, 0, self.n_observations - 1)
                
                T = len(observations)
                alpha, log_prob = self._forward(observations)
                beta = self._backward(observations)
                total_log_prob += log_prob
                
                gamma = alpha * beta
                gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
                
                sum_init += gamma[0]
                for t in range(T - 1):
                    xi_t = np.zeros((self.n_states, self.n_states))
                    for i in range(self.n_states):
                        for j in range(self.n_states):
                            xi_t[i, j] = alpha[t, i] * self.transition_matrix_[i, j] * \
                                        self.emission_matrix_[j, observations[t + 1]] * beta[t + 1, j]
                    xi_sum = np.sum(xi_t)
                    if xi_sum > 0:
                        xi_t /= xi_sum
                    sum_trans += xi_t
                
                for t in range(T):
                    sum_emit[:, observations[t]] += gamma[t]
            
            self.initial_probs_ = sum_init / len(X)
            self.transition_matrix_ = sum_trans / len(X)
            self.emission_matrix_ = sum_emit / len(X)
            
            self.initial_probs_ /= self.initial_probs_.sum() + 1e-300
            self.transition_matrix_ /= self.transition_matrix_.sum(axis=1, keepdims=True) + 1e-300
            self.emission_matrix_ /= self.emission_matrix_.sum(axis=1, keepdims=True) + 1e-300
            
            self.log_likelihood_history.append(total_log_prob)
            
            if (iteration + 1) % 10 == 0:
                print(f"迭代 {iteration + 1}: 对数似然 = {total_log_prob:.4f}")
        
        self.is_fitted = True
        print(f"\n训练完成!")
        return self
    
    def predict_sequence(self, observations: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        observations = observations.astype(int)
        observations = np.clip(observations, 0, self.n_observations - 1)
        
        T = len(observations)
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        viterbi[0] = self.initial_probs_ * self.emission_matrix_[:, observations[0]]
        
        for t in range(1, T):
            for j in range(self.n_states):
                probs = viterbi[t - 1] * self.transition_matrix_[:, j] * \
                       self.emission_matrix_[j, observations[t]]
                viterbi[t, j] = np.max(probs)
                backpointer[t, j] = np.argmax(probs)
        
        best_path = np.zeros(T, dtype=int)
        best_path[-1] = np.argmax(viterbi[-1])
        
        for t in range(T - 2, -1, -1):
            best_path[t] = backpointer[t + 1, best_path[t + 1]]
        
        return best_path


if __name__ == "__main__":
    print("="*60)
    print("HMM 隐马尔可夫模型示例")
    print("="*60)
    
    np.random.seed(42)
    sequences = []
    
    # 生成模拟数据
    for _ in range(30):
        seq = []
        current_state = 0
        for _ in range(10):
            if current_state == 0:
                obs = np.random.randint(0, 2)
            else:
                obs = np.random.choice([0, 1], p=[0.2, 0.8])
            seq.append(obs)
            if np.random.rand() < 0.3:
                current_state = 1 - current_state
        sequences.append(np.array(seq))
    
    hmm = HiddenMarkovModel(n_states=2, n_observations=2, max_iter=30, random_state=42)
    hmm.fit(sequences)
    
    predicted = hmm.predict_sequence(sequences[0])
    print(f"\n观测序列: {sequences[0]}")
    print(f"预测状态: {predicted}")
    
    print("\n" + "="*60)
