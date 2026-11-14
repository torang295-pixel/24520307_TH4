import numpy as np
from sklearn.svm import LinearSVC


class SVM:
    def __init__(self, C: float = 0.1, max_iter: int = 1000):
        self.model = LinearSVC(C=C, max_iter=max_iter)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray):
        return self.model.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray):
        return self.model.score(X, y)

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        y_pred = np.where(self.predict(X) >= 0, 1, -1)
        A = np.mean(y_pred == y)
        tp = np.sum((y_pred == 1) & (y == 1))
        P = tp / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
        R = tp / np.sum(y == 1) if np.sum(y == 1) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        return A, P, R, F1