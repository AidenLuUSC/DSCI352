# Credit: https://github.com/christianversloot/machine-learning-articles/blob/main/linking-maths-and-intuition-rosenblatts-perceptron-in-python.md?plain=1


# Sepal and petal length
import numpy as np
import pandas as pd

df = pd.read_csv("Iris.csv")
df = df[df["Species"].isin(["Iris-setosa", "Iris-versicolor"])]
df["Label"] = df["Species"].map({"Iris-setosa": 0, "Iris-versicolor": 1})

X = df[["SepalLengthCm", "PetalLengthCm"]].values
T = df["Label"].values

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plt.scatter(df["SepalLengthCm"], df["PetalLengthCm"], c=T, cmap='bwr')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

class RBPerceptron():

  # Constructor object, self is the instance of the object NeuralNetwork itself
  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate

  # Train perceptron
  def train(self, X, T):
    # Initialize weights vector with zeroes
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    # Perform the epochs
#    diff = []
#    err = []
    for i in range(self.number_of_epochs):
#      err.append(sum(diff))
#      print(err)
      # For every combination of (X_i, T_i), zip creates the tuples
      for sample, desired_outcome in zip(X, T):
        # Generate prediction and compare with desired outcome
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
#        diff.append(difference)
        # Compute weight update via Perceptron Learning Rule
        self.w[1:]    += self.learning_rate * difference * sample
        self.w[0]     += self.learning_rate * difference * 1
    return self

  # Generate prediction
  def predict(self, sample):
    # dot product:
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    # Activation function:
    return np.where(outcome > 0, 1, 0)

# ks = [100,200,300,400,500,600,700, 800, 900, 1000]
ks = [100, 200, 300, 500]
colors = ['blue','limegreen','gray','cyan','red','red','red']

for k in ks:
  # print(k)
  rbp = RBPerceptron(k, 0.1)
  trained_model = rbp.train(X, T)

  plot_decision_regions(X, T.astype(int), clf=trained_model, legend=0)

plt.title('Perceptron')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


# Sepal and petal width
df = pd.read_csv("Iris.csv")
df = df[df["Species"].isin(["Iris-setosa", "Iris-versicolor"])]
df["Label"] = df["Species"].map({"Iris-setosa": 0, "Iris-versicolor": 1})

X = df[["SepalWidthCm", "PetalWidthCm"]].values
T = df["Label"].values

from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
plt.scatter(df["SepalWidthCm"], df["PetalWidthCm"], c=T, cmap='bwr')
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Petal Width (cm)")
plt.show()

# Rosenblatt Perceptron

# Basic Rosenblatt Perceptron implementation
class RBPerceptron():

  # Constructor object, self is the instance of the object NeuralNetwork itself
  def __init__(self, number_of_epochs = 100, learning_rate = 0.1):
    self.number_of_epochs = number_of_epochs
    self.learning_rate = learning_rate

  # Train perceptron
  def train(self, X, T):
    # Initialize weights vector with zeroes
    num_features = X.shape[1]
    self.w = np.zeros(num_features + 1)
    # Perform the epochs
#    diff = []
#    err = []
    for i in range(self.number_of_epochs):
#      err.append(sum(diff))
#      print(err)
      # For every combination of (X_i, T_i), zip creates the tuples
      for sample, desired_outcome in zip(X, T):
        # Generate prediction and compare with desired outcome
        prediction    = self.predict(sample)
        difference    = (desired_outcome - prediction)
#        diff.append(difference)
        # Compute weight update via Perceptron Learning Rule
        self.w[1:]    += self.learning_rate * difference * sample
        self.w[0]     += self.learning_rate * difference * 1
    return self

  # Generate prediction
  def predict(self, sample):
    # dot product:
    outcome = np.dot(sample, self.w[1:]) + self.w[0]
    # Activation function:
    return np.where(outcome > 0, 1, 0)

# ks = [100,200,300,400,500,600,700, 800, 900, 1000]
ks = [100, 200, 300, 500]
colors = ['blue','limegreen','gray','cyan','red','red','red']

for k in ks:
  # print(k)
  rbp = RBPerceptron(k, 0.1)
  trained_model = rbp.train(X, T)

  plot_decision_regions(X, T.astype(int), clf=trained_model, legend=0)

plt.title('Perceptron')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

# Not sure what the 'bigger problems' are supposed to be? This worked fine for me?