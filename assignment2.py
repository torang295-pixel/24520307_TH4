import numpy as np


class SVM:
    def __init__(self, C: float = 0.1, epochs: int = 100, tol: float = 1e-3, eps: float = 1e-8):
        self.W = None
        self.b = None
        self.C = C
        self.epochs = epochs
        self.tol = tol
        self.eps = eps
        self.alpha = None
        self.n_iter = 0

    def _violates_kkt(self, alpha_i, y_i_error):
        if alpha_i <= self.eps:
            return y_i_error < -self.tol
        if alpha_i >= self.C - self.eps:
            return y_i_error > self.tol
        return abs(y_i_error) > self.tol

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y, dtype=float)
        n, dim = X.shape
        self.W = np.zeros(dim, dtype=float)
        self.b = 0.0
        self.alpha = np.zeros(n, dtype=float)
        self.n_iter = 0

        for epoch in range(self.epochs):
            changed = 0
            for i in range(n):
                Ei = float(np.dot(X[i], self.W) + self.b - y[i])
                yiEi = y[i] * Ei
                if not self._violates_kkt(self.alpha[i], yiEi):
                    continue

                j = (i + epoch + 1) % n
                if j == i:
                    continue
                Ej = float(np.dot(X[j], self.W) + self.b - y[j])

                old_ai = self.alpha[i]
                old_aj = self.alpha[j]
                if y[i] != y[j]:
                    lower = max(0.0, old_aj - old_ai)
                    upper = min(self.C, self.C + old_aj - old_ai)
                else:
                    lower = max(0.0, old_ai + old_aj - self.C)
                    upper = min(self.C, old_ai + old_aj)
                if lower >= upper - self.eps:
                    continue

                eta = float(np.dot(X[i] - X[j], X[i] - X[j]))
                if eta <= self.eps:
                    continue

                new_aj = old_aj + y[j] * (Ei - Ej) / eta
                new_aj = np.clip(new_aj, lower, upper)
                if abs(new_aj - old_aj) <= self.eps:
                    continue

                new_ai = old_ai + y[i] * y[j] * (old_aj - new_aj)
                delta_ai = new_ai - old_ai
                delta_aj = new_aj - old_aj

                kii = np.dot(X[i], X[i])
                kij = np.dot(X[i], X[j])
                kjj = np.dot(X[j], X[j])
                b1 = (self.b - Ei - y[i] * delta_ai * kii
                      - y[j] * delta_aj * kij)
                b2 = (self.b - Ej - y[i] * delta_ai * kij
                      - y[j] * delta_aj * kjj)

                self.alpha[i] = new_ai
                self.alpha[j] = new_aj
                self.W += y[i] * delta_ai * X[i] + y[j] * delta_aj * X[j]

                if self.eps < new_ai < self.C - self.eps:
                    self.b = b1
                elif self.eps < new_aj < self.C - self.eps:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2.0
                changed += 1

            self.n_iter = epoch + 1
            if changed == 0:
                break

        self.support_vectors_ = np.flatnonzero(self.alpha > self.eps)

    def loss_fn(self, y_hat: np.ndarray, y: np.ndarray):
        if self.W is None:
            return 0
        y_hat = np.asarray(y_hat)
        y = np.asarray(y)
        return np.mean(np.maximum(0, 1 - y * y_hat)) + 0.5 * np.sum(self.W ** 2)

    def predict(self, X: np.ndarray):
        return np.dot(X, self.W) + self.b if self.W is not None else 0

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        y = np.asarray(y)
        y_pred = self.predict(X)
        y_pred_labels = np.where(y_pred >= 0, 1, -1)
        A = np.mean(y_pred_labels == y)
        tp = np.sum((y_pred_labels == 1) & (y == 1))
        P = tp / np.sum(y_pred_labels == 1) if np.sum(y_pred_labels == 1) > 0 else 0
        R = tp / np.sum(y == 1) if np.sum(y == 1) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        return A, P, R, F1
