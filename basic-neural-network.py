## Import modules
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import os, struct

## Random seed
rgen = np.random.RandomState(1)

## Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

## Derivitave of the Sigmoid function
def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

## Load data
iris = datasets.load_iris()
X = iris.data[:, [0, 3]]
y = iris.target

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, 
                               '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, 
                               '%s-images.idx3-ubyte' % kind)
        
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', 
                                 lbpath.read(8))
        labels = np.fromfile(lbpath, 
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", 
                                               imgpath.read(16))
        images = np.fromfile(imgpath, 
                             dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.) - .5) * 2
 
    return images, labels

X_train, y_train = load_mnist('./mnist/', kind = 'train')
X, y = load_mnist('./mnist/', kind = 't10k')
## Data standardization
stdsc = StandardScaler()
X = stdsc.fit_transform(X)

## Neural network architecture
#   3 layers: input layer, hidden layer, output layer
#       Input layer: 2 nodes
#       Hidden layer: 5 nodes
#       Output layer: 3 nodes
input_nodes = 784
hidden_nodes = 25
output_nodes = 10

# Number of classes
K = np.unique(y).astype(int)
# Number of labels
class_labels = len(K)

## Weight initialization
weight_1 = rgen.normal(scale = 0.1, size = (X.shape[1] + 1, hidden_nodes))
weight_2 = rgen.normal(scale = 0.1, size = (weight_1.shape[1] + 1, output_nodes))

# Data shuffle
train_test_split = np.c_[X, y]
rgen.shuffle(train_test_split)
X = train_test_split[:, :X.shape[1]]
y = train_test_split[:, X.shape[1]]

## Forward propagation function
def predict(X, weight_1, weight_2):
    if X.ndim > 1:
        a_1 = np.hstack((np.ones((X.shape[0], 1)), X))
        Z_2 = a_1.dot(weight_1)
        a_2 = sigmoid(Z_2)
        a_2 = np.hstack((np.ones((a_2.shape[0], 1)), a_2))
        Z_3 = a_2.dot(weight_2)
        a_3 = sigmoid(Z_3)
        return a_3, a_2, a_1, np.argmax(a_3, axis = 1)
    elif X.ndim == 1:
        a_1 = np.hstack((1, X))
        Z_2 = a_1.dot(weight_1)
        a_2 = sigmoid(Z_2)
        a_2 = np.hstack((1, a_2))
        Z_3 = a_2.dot(weight_2)
        a_3 = sigmoid(Z_3)
        return a_3, a_2, a_1, np.argmax(a_3)
    
## One hot encoder function
def one_hot_encoder(y):
    a = np.zeros((len(y), class_labels))
    for idx, i in enumerate(y):
        a[idx, int(i)] = 1
    return a

y_coded = one_hot_encoder(y)
# Learning rate
eta = 0.1

# Number of epochs
epoch = 500

# Cost array
cost_array = []

# Sample size
m = X.shape[0]

# Weight decay
cost_lambda = 1

## Training
for _ in range(epoch):
    # Forward propagation
    a_3, a_2, a_1, y_pred = predict(X, weight_1, weight_2)
    # Weight gradient initialization
    weight_1_grad = np.zeros(weight_1.shape)
    weight_2_grad = np.zeros(weight_2.shape)
    
    # Cost calculation
    cost = 0
    for k in K:
        cost = cost + np.sum(-y_coded[:, k]*np.log(a_3[:, k]) - (1 - y_coded[:, k]) * np.log(1 - a_3[:, k]))/m
    cost_array.append(cost)
    
    regularization = (np.sum(np.sum(weight_1[:, 1:]**2)) + np.sum(np.sum(weight_2[:, 1:]**2)))*cost_lambda/2/m
    cost += regularization
    
    cap_delta_1 = np.zeros(weight_1.shape)
    cap_delta_2 = np.zeros(weight_2.shape)  
    
    for t in range(m):
        # Forward propagation
        a_3, a_2, a_1, y_pred = predict(X[t], weight_1, weight_2)
        delta_3 = a_3 - y_coded[t]
        z_2 = a_1.dot(weight_1)
        delta_2 = weight_2.dot(delta_3)
        delta_2 = delta_2[1:]
        delta_2 = delta_2 * sigmoidGradient(z_2)
        
        cap_delta_1 += a_1.reshape(-1, 1).dot(delta_2.reshape(1, -1))
        cap_delta_2 += a_2.reshape(-1, 1).dot(delta_3.reshape(1, -1))
        
    weight_1_grad = cap_delta_1 / m
    weight_2_grad = cap_delta_2 / m
    
    weight_1 -= eta * weight_1_grad
    weight_2 -= eta * weight_2_grad
#    
#    if _ > 2:
#        if cost_array[-2] - cost_array[-1] < 10**-4:
#            break
plt.plot(range(len(cost_array)), cost_array)
plt.show()

# Data plot function
# from matplotlib.colors import ListedColormap
# def plot_decision_regions(X, y, weight_1, weight_2, resolution = 0.01):
#     markers = ('s', 'x', 'o', '^', 'v')
#     colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
#     cmap = ListedColormap(colors[:len(np.unique(y))])
    
#     # Draw the plane
#     x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    
#     # Predict using the meshgrid
#     a_3, a_2, a_1, Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T, weight_1, weight_2)
#     Z = Z.reshape(xx1.shape)
    
#     # Draw the binary decision range
#     plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
#     plt.xlim(xx1.min(), xx1.max())
#     plt.ylim(xx2.min(), xx2.max())

#     # Plot the data
#     for idx, cl in enumerate(np.unique(y)):
#         plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')

# Calculate accuracy and confusion matrix
from sklearn.metrics import confusion_matrix

a_3, a_2, a_1, Z = predict(X, weight_1, weight_2)
accuracy = np.mean((Z == y)*100)
print("Accuracy =", str(accuracy))
print(confusion_matrix(y, Z))

# Data plot
# plot_decision_regions(X, y, weight_1, weight_2)
# plt.show()