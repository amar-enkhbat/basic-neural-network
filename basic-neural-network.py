## Import modules
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

rgen = np.random.RandomState(1)

## Sigmoid function
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def sigmoidGradient(z):
    return sigmoid(z) * (1 - sigmoid(z))
## Load data
iris = datasets.load_iris()
X = iris.data[:, [0, 3]]
y = iris.target
stdsc = StandardScaler()
X = stdsc.fit_transform(X)
## Neural network architecture
#   3 layers: input layer, hidden layer, output layer
#       Input layer: 2 nodes
#       Hidden layer: 5 nodes
#       Output layer: 3 nodes
input_nodes = 2
hidden_nodes = 5
output_nodes = 3
class_labels = 3
## Initializing weight
weight_1 = rgen.normal(scale = 0.1, size = (X.shape[1] + 1, hidden_nodes))
weight_2 = rgen.normal(scale = 0.1, size = (weight_1.shape[1] + 1, output_nodes))

def predict(X, weight_1, weight_2):
    a_1 = np.hstack((np.ones((X.shape[0], 1)), X))
    Z_2 = a_1.dot(weight_1)
    a_2 = sigmoid(Z_2)
    a_2 = np.hstack((np.ones((a_2.shape[0], 1)), a_2))
    Z_3 = a_2.dot(weight_2)
    a_3 = sigmoid(Z_3)
    return a_3, a_2, a_1, np.argmax(a_3, axis = 1)


def one_hot_encoder(y):
    a = np.zeros((len(y), class_labels))
    for idx, i in enumerate(y):
        a[idx, int(i)] = 1
    return a

eta = 0.1
cost_array = []
y_coded = one_hot_encoder(y)
epoch = 5000
m = X.shape[0]
cost_lambda = 1

for _ in range(epoch):
    
#    a_3, a_2, a_1, y_pred = predict(X, weight_1, weight_2)
    a_1 = np.hstack((np.ones((X.shape[0], 1)), X))
    Z_2 = a_1.dot(weight_1)
    a_2 = sigmoid(Z_2)
    a_2 = np.hstack((np.ones((a_2.shape[0], 1)), a_2))
    Z_3 = a_2.dot(weight_2)
    a_3 = sigmoid(Z_3)
    
    
    cost = 0
    weight_1_grad = np.zeros(weight_1.shape)
    weight_2_grad = np.zeros(weight_2.shape)
    K = np.unique(y)
    
    
    for k in K:
        cost = cost + np.sum(-y_coded[:, k]*np.log(a_3[:, k]) - (1 - y_coded[:, k]) * np.log(1 - a_3[:, k]))/m
    
    cost_array.append(cost)
#    regularization = (np.sum(np.sum(weight_1[:, 1:]**2)) + np.sum(np.sum(weight_2[:, 1:]**2)))*cost_lambda/2/m
#    cost += regularization
    
    cap_delta_1 = np.zeros(weight_1.shape)
    cap_delta_2 = np.zeros(weight_2.shape)    
    for t in range(m):
        a_1 = X[t]
        a_1 = np.hstack((1, a_1))
#        print("a_1:")
#        print(a_1)
        z_2 = a_1.dot(weight_1)
#        print("z_2:")
#        print(z_2)
        a_2 = sigmoid(z_2)
#        print("a_2:")
#        print(a_2)
        a_2 = np.hstack((1, a_2))
        z_3 = a_2.dot(weight_2)
#        print("z_3:")
#        print(z_3)
        a_3 = sigmoid(z_3)
#        print("a_3:")
#        print(a_3)

        delta_3 = a_3 - y_coded[t]
        
        delta_2 = weight_2.dot(delta_3)
        delta_2 = delta_2[1:]
        delta_2 = delta_2 * sigmoidGradient(z_2)
        cap_delta_1 += a_1.reshape(-1, 1).dot(delta_2.reshape(1, -1))
        cap_delta_2 += a_2.reshape(-1, 1).dot(delta_3.reshape(1, -1))
#        print()
    weight_1_grad = cap_delta_1 / m
    weight_2_grad = cap_delta_2 / m
    
    weight_1 -= eta * weight_1_grad
    weight_2 -= eta * weight_2_grad
plt.plot(range(len(cost_array)), cost_array)
plt.show()

# Data plot
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, weight_1, weight_2, resolution = 0.01):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    a_3, a_2, a_1, Z = predict(np.array([xx1.ravel(), xx2.ravel()]).T, weight_1, weight_2)
    
    Z = Z.reshape(xx1.shape)
    print(np.unique(Z))
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = colors[idx], marker = markers[idx], label = cl, edgecolor = 'black')

plot_decision_regions(X, y, weight_1, weight_2)
a_3, a_2, a_1, Z = predict(X, weight_1, weight_2)
print(Z)
accuracy = np.mean((Z == y)*100)
print("accuracy = ", str(accuracy))
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, Z))
#plt.show()
