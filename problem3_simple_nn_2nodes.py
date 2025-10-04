from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

df = pd.read_csv("Titanic.csv")
df = df.drop(columns=["Passenger"])
for col in ["Class", "Sex", "Age"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
df['Survived'] = df['Survived'].map({'No':0, 'Yes':1})

X = df.drop(columns=['Survived']).values
y = df['Survived'].values

# Train and test partitions
X_train, X_test , Y_train, Y_test = \
    train_test_split(X, y,test_size=0.30, random_state=2023, stratify=y)

class TwoNodeNN:
    def __init__(self, learning_rate, n_features):
        # Hidden layer: 2 neurons
        self.W1 = np.random.randn(2, n_features) * 0.01   # (2, d)
        self.b1 = np.zeros(2)                              # (2,)
        # Output layer: 1 neuron (binary classification)
        self.W2 = np.random.randn(1, 2) * 0.01             # (1, 2)
        self.b2 = 0.0                                      # scalar
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_deriv_from_sigmoid(self, s):
        # given s = sigmoid(z)
        return s * (1.0 - s)

    # -------- forward pass --------
    def _forward_single(self, x):
        # x: (d,)
        z1 = self.W1 @ x + self.b1            # (2,)
        a1 = self._sigmoid(z1)                # (2,)
        z2 = self.W2 @ a1 + self.b2           # (1,)
        yhat = self._sigmoid(z2)[0]           # scalar
        return z1, a1, z2.item(), yhat

    # Vectorized predict_proba for grid/batches
    def predict_proba(self, X):
        if hasattr(X, "values"):              # DataFrame/Series
            X = X.values
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            _, a1, _, yhat = self._forward_single(X)
            return yhat
        else:
            # Vectorized forward
            # Z1 = X @ W1.T + b1 -> (n,2)
            Z1 = X @ self.W1.T + self.b1
            A1 = self._sigmoid(Z1)
            Z2 = A1 @ self.W2.T + self.b2     # (n,1)
            Yhat = self._sigmoid(Z2).reshape(-1)
            return Yhat

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # -------- backprop (MSE) for one sample --------
    def _compute_gradients(self, x, y):
        # Forward
        z1, a1, z2, yhat = self._forward_single(x)

        # Loss L = (yhat - y)^2
        dL_dyhat = 2.0 * (yhat - y)                            # scalar
        dyhat_dz2 = self._sigmoid_deriv_from_sigmoid(1/(1+np.exp(-z2)))
        # More stable: use yhat directly:
        dyhat_dz2 = yhat * (1.0 - yhat)
        delta2 = dL_dyhat * dyhat_dz2                          # scalar

        # Output layer grads
        dL_dW2 = delta2 * a1.reshape(1, -1)                    # (1,2)
        dL_db2 = delta2                                        # scalar

        # Backprop to hidden
        # delta1 = (W2^T * delta2) ⊙ sigmoid'(z1)
        sigp_z1 = self._sigmoid_deriv_from_sigmoid(a1)         # (2,)
        delta1 = (self.W2.T.reshape(2,) * delta2) * sigp_z1    # (2,)

        # Hidden layer grads
        dL_dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)      # (2,d)
        dL_db1 = delta1                                         # (2,)

        return dL_dW1, dL_db1, dL_dW2, dL_db2

    def _update(self, dW1, db1, dW2, db2):
        lr = self.learning_rate
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def train(self, X, y, iterations=20000, report_every=200):
        errs = []
        n = len(y)
        for it in range(iterations):
            i = np.random.randint(n)
            dW1, db1, dW2, db2 = self._compute_gradients(X[i], y[i])
            self._update(dW1, db1, dW2, db2)
            if it % report_every == 0:
                yhat_all = self.predict_proba(X)
                errs.append(np.mean((yhat_all - y) ** 2))
        return errs

# Plot the results

learning_rate = 0.1
neural_network = TwoNodeNN(learning_rate, n_features=X_train.shape[1])
training_error = neural_network.train(X_train, Y_train, iterations=1000, report_every=50)
test_error = neural_network.train(X_test, Y_test, iterations=1000, report_every=50)

Y_train_predict = neural_network.predict(X_train)
train_accuracy = accuracy_score(Y_train, Y_train_predict)
train_confusion = confusion_matrix(Y_train, Y_train_predict)

Y_test_predict = neural_network.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_predict)
test_confusion = confusion_matrix(Y_test, Y_test_predict)

print(f"Training Accuracy: {train_accuracy:.4f}")
print("Training Confusion Matrix:")
print(train_confusion)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Confusion Matrix:")
print(test_confusion)

# Plot training error curve
plt.plot(training_error)
plt.plot(test_error)
plt.xlabel("Iteration (per 50)")
plt.ylabel("MSE Error")
plt.show()

class TwoLayerNN:
    def __init__(self, learning_rate, n_features):
        # Hidden layer 1: 2 neurons
        self.W1 = np.random.randn(2, n_features) * 0.01   # (2, d)
        self.b1 = np.zeros(2)                              # (2,)
        # Hidden layer 2: 2 neurons
        self.W2 = np.random.randn(2, 2) * 0.01             # (2, 2)
        self.b2 = np.zeros(2)                              # (2,)
        # Output layer: 1 neuron (binary classification)
        self.W3 = np.random.randn(1, 2) * 0.01             # (1, 2)
        self.b3 = 0.0                                      # scalar
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        x = np.clip(x, -50, 50)
        return 1.0 / (1.0 + np.exp(-x))

    def _sigmoid_deriv_from_sigmoid(self, s):
        return s * (1.0 - s)

    # -------- forward pass for single sample --------
    def _forward_single(self, x):
        # Layer 1
        z1 = self.W1 @ x + self.b1             # (2,)
        a1 = self._sigmoid(z1)                 # (2,)
        # Layer 2
        z2 = self.W2 @ a1 + self.b2            # (2,)
        a2 = self._sigmoid(z2)                 # (2,)
        # Output layer
        z3 = self.W3 @ a2 + self.b3            # (1,)
        yhat = self._sigmoid(z3)[0]            # scalar
        return (z1, a1, z2, a2, z3.item(), yhat)

    def predict_proba(self, X):
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            _, a1, _, a2, _, yhat = self._forward_single(X)
            return yhat
        else:
            Z1 = X @ self.W1.T + self.b1       # (n, 2)
            A1 = self._sigmoid(Z1)             # (n, 2)
            Z2 = A1 @ self.W2.T + self.b2      # (n, 2)
            A2 = self._sigmoid(Z2)             # (n, 2)
            Z3 = A2 @ self.W3.T + self.b3      # (n, 1)
            Yhat = self._sigmoid(Z3).reshape(-1)
            return Yhat

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def _compute_gradients(self, x, y):
        z1, a1, z2, a2, z3, yhat = self._forward_single(x)

        # Loss = (yhat - y)^2
        dL_dyhat = 2.0 * (yhat - y)                            # scalar
        dyhat_dz3 = yhat * (1.0 - yhat)                        # scalar
        delta3 = dL_dyhat * dyhat_dz3                          # scalar

        # Output layer grads
        dL_dW3 = delta3 * a2.reshape(1, -1)                    # (1, 2)
        dL_db3 = delta3                                        # scalar

        # Backprop to hidden layer 2
        sigp_z2 = self._sigmoid_deriv_from_sigmoid(a2)         # (2,)
        delta2 = (self.W3.T.reshape(2,) * delta3) * sigp_z2    # (2,)

        dL_dW2 = delta2.reshape(-1, 1) @ a1.reshape(1, -1)     # (2, 2)
        dL_db2 = delta2                                         # (2,)

        # Backprop to hidden layer 1
        sigp_z1 = self._sigmoid_deriv_from_sigmoid(a1)         # (2,)
        delta1 = (self.W2.T @ delta2) * sigp_z1                # (2,)

        dL_dW1 = delta1.reshape(-1, 1) @ x.reshape(1, -1)      # (2, d)
        dL_db1 = delta1                                         # (2,)

        return dL_dW1, dL_db1, dL_dW2, dL_db2, dL_dW3, dL_db3

    def _update(self, dW1, db1, dW2, db2, dW3, db3):
        lr = self.learning_rate
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3

    def train(self, X, y, iterations=20000, report_every=200):
        errs = []
        n = len(y)
        for it in range(iterations):
            i = np.random.randint(n)
            grads = self._compute_gradients(X[i], y[i])
            self._update(*grads)
            if it % report_every == 0:
                yhat_all = self.predict_proba(X)
                errs.append(np.mean((yhat_all - y) ** 2))
        return errs

# Plot the results

learning_rate = 0.1
neural_network = TwoLayerNN(learning_rate, n_features=X_train.shape[1])
training_error = neural_network.train(X_train, Y_train, iterations=1000, report_every=50)
test_error = neural_network.train(X_test, Y_test, iterations=1000, report_every=50)

Y_train_predict = neural_network.predict(X_train)
train_accuracy = accuracy_score(Y_train, Y_train_predict)
train_confusion = confusion_matrix(Y_train, Y_train_predict)

Y_test_predict = neural_network.predict(X_test)
test_accuracy = accuracy_score(Y_test, Y_test_predict)
test_confusion = confusion_matrix(Y_test, Y_test_predict)

print(f"Training Accuracy: {train_accuracy:.4f}")
print("Training Confusion Matrix:")
print(train_confusion)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Test Confusion Matrix:")
print(test_confusion)

# Plot training error curve
plt.plot(training_error)
plt.plot(test_error)
plt.xlabel("Iteration (per 50)")
plt.ylabel("MSE Error")
plt.show()