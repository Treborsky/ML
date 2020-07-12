#!/usr/bin/python
# -*- coding: utf-8 -*-
# we got it boys it works actually nice man
# Generate Toy Dataset
import matplotlib.pyplot as plt
import numpy

x = numpy.linspace(-1, 1, 100)
signal = 2 + x + 2 * x * x
noise = numpy.random.normal(0, 0.1, 100)
y = signal + noise
plt.plot(signal, 'b')
plt.plot(y, 'g')
plt.plot(noise, 'r')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(["Without Noise", "With Noise", "Noise"], loc=2)
x_train = x[0:80]
y_train = y[0:80]


# Model with degree 1
plt.figure()
degree = 2
X_train = numpy.column_stack([numpy.power(x_train,i) for i in range(0, degree)])
model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
plt.plot(x, y, 'g')
plt.xlabel("x")
plt.ylabel("y")
predicted = numpy.dot(model, [numpy.power(x,i) for i in range(0, degree)])
plt.plot(x, predicted, 'r')
plt.legend(["Actual", "Predicted"], loc=2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
test_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
print("Train RMSE (Degree = 1)", train_rmse1)
print("Test RMSE (Degree = 1)", test_rmse1)


# Model with degree 2
plt.figure()
degree = 3
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, degree)])
model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
plt.plot(x, y, 'g')
plt.xlabel("x")
plt.ylabel("y")
predicted = numpy.dot(model, [numpy.power(x, i) for i in range(0, degree)])
plt.plot(x, predicted, 'r')
plt.legend(["Actual", "Predicted"], loc=2)
train_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
test_rmse1 = numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
print("Train RMSE (Degree = 2)", train_rmse1)
print("Test RMSE (Degree = 2)", test_rmse1)

# Model with degree 8
plt.figure()
degree = 9
X_train = numpy.column_stack([numpy.power(x_train, i) for i in range(0, degree)])
model = numpy.dot(numpy.dot(numpy.linalg.inv(numpy.dot(X_train.transpose(), X_train)), X_train.transpose()), y_train)
plt.plot(x, y, 'g')
plt.xlabel("x")
plt.ylabel("y")
predicted = numpy.dot(model, [numpy.power(x, i) for i in range(0, degree)])
plt.plot(x, predicted, 'r')
plt.legend(["Actual", "Predicted"], loc=3)
train_rmse2 = numpy.sqrt(numpy.sum(numpy.dot(y[0:80] - predicted[0:80], y_train - predicted[0:80])))
test_rmse2 = numpy.sqrt(numpy.sum(numpy.dot(y[80:] - predicted[80:], y[80:] - predicted[80:])))
print("Train RMSE (Degree = 8)", train_rmse2)
print("Test RMSE (Degree = 8)", test_rmse2)
