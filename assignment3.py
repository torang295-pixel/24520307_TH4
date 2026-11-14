import numpy as np

class SVM:
    def __init__ (self, C: float=0.1, epochs: int=100):
        self.W = None
        self.b = None
        self.C = C
        self.epochs = epochs
        self.learning_rate = 0.001

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, dim = X.shape
        self.W = np.zeros(dim)
        self.b = 0

        for epoch in range(self.epochs):
            for i in range(n):
                x = X[i]
                a = y[i]
                y_pred = self.predict(x)    

                dW = None
                db = None
                if a * y_pred < 1:
                    dW = self.W - self.C * a * x
                    db = -self.C * a                  
                else:
                    dW = self.W
                    db = 0

            self.W -= dW * self.learning_rate
            self.b -= db * self.learning_rate

    def loss_fn(self, y_hat: np.ndarray, y: np.ndarray):
       return np.mean(np.maximum(0, 1 - y * y_hat)) + (self.C / 2) * np.sum(self.W ** 2) if self.W is not None else 0

    def predict(self, X: np.ndarray):
        y_pred = np.dot(X, self.W) + self.b if self.W is not None else 0
        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)
        y_pred_labels = np.where(y_pred >= 0, 1, -1)
        A = np.mean(y_pred_labels == y)
        tp = np.sum((y_pred_labels == 1) & (y == 1))
        P = tp / np.sum(y_pred_labels == 1) if np.sum(y_pred_labels == 1) > 0 else 0
        R = tp / np.sum(y == 1) if np.sum(y == 1) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        return A, P, R, F1