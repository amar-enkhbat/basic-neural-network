## Import modules
import numpy as np 
import matplotlib.pyplot as plt 
rgen = np.random.RandomState(3)

## Sigmoid function
def sigmoid(Z):
    return 1/(1 - np.exp(-Z))

## Load data
X = np.loadtxt('input.csv', delimiter= ',')
y = np.loadtxt('label.csv', delimiter= ',')

weight_1 = np.loadtxt('theta1.csv', delimiter = ',')
weight_2 = np.loadtxt('theta2.csv', delimiter = ',')

## Plot samples
# Take 10 random samples
sample_number = rgen.randint(X.shape[0], size = 10)
sample_data = X[sample_number]

# Plot 10 random samples
# for i in range(10):
#     plt.subplot(2, 5, i+1)
#     plt.imshow(np.flip(np.rot90(sample_data[i].reshape(20, 20), 1), 0))
#     plt.xticks([])
#     plt.yticks([])
# plt.show()

def predict(X, y, weight_1, weight_2):
    X = np.hstack((np.ones(1), X))
    Z_2 = X.dot(weight_1.T)
    a_2 = sigmoid(Z_2)
    a_2 = np.hstack((np.ones(1), a_2))
    Z_3 = a_2.dot(weight_2.T)
    a_3 = sigmoid(Z_3)
    return np.argmax(a_3)

print(sample_data[4].shape)

for i in sample_data:
    print(predict(i, y, weight_1, weight_2))

for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(sample_data[i].reshape(20, 20))
    plt.xticks([])
    plt.yticks([])
plt.show()



