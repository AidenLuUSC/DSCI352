# Aiden Lu
# DSCI Fall 2025
# HW4

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1.1
np.random.seed(42)

points = 400
theta = np.linspace(0, 2 * np.pi, points)

r1 = 2 * theta + np.pi
x1 = r1 * np.cos(theta) + np.random.normal(0, 1, points)
y1 = r1 * np.sin(theta) + np.random.normal(0, 1, points)
r2 = -2 * theta - np.pi
x2 = r2 * np.cos(theta) + np.random.normal(0, 1, points)
y2 = r2 * np.sin(theta) + np.random.normal(0, 1, points)

X = np.vstack([np.column_stack((x1, y1)), np.column_stack((x2, y2))])
y = np.hstack([np.zeros(points), np.ones(points)])

# print(X)
# print(y)

# 1.2
plt.scatter(x1, y1, c='blue', label='0', alpha=0.5)
plt.scatter(x2, y2, c='red', label='1', alpha=0.5)
plt.legend()
# plt.show()

# 1.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y,random_state=2023)

# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

# 1.4
mlp = MLPClassifier(hidden_layer_sizes=(60, 60), learning_rate_init=0.01, max_iter=100, random_state=2023)
mlp.fit(X_train, y_train)

# 1.5
plt.close()
plt.plot(mlp.loss_curve_)
# plt.show()

# 1.6
y_predict = mlp.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
# print("Test Accuracy: ", accuracy)

# 1.7
plt.close()
confuse_mat = confusion_matrix(y_test, y_predict)
disp = ConfusionMatrixDisplay(confusion_matrix=confuse_mat, display_labels=[0, 1])
# disp.plot()
# plt.show()

# 1.8
x_range, y_range = np.meshgrid(np.linspace(-20, 20, 400), np.linspace(-20, 20, 400))
predict_grid = mlp.predict(np.c_[x_range.ravel(), y_range.ravel()])
predict_grid = predict_grid.reshape(x_range.shape)
plt.imshow(predict_grid, extent=(-20, 20, -20, 20), origin='lower', alpha=0.5)
plt.scatter(x1, y1, color='blue', label='Class 0', alpha=0.5)
plt.scatter(x2, y2, color='red', label='Class 1', alpha=0.5)
plt.legend()
# plt.show()