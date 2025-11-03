# This is a demo code which shows the neural network implementation with SKLearn.
# Please Note: Don't use SKLearn implementation of Neural Network as it is not optimized.............

# Using Neural Network on IRIS Dataset..........

from sklearn import model_selection
from sklearn import datasets

iris=datasets.load_iris()                       # Loading the IRIS Dataset.
x=iris.data
y=iris.target

# Splitting the data into training and testing.......
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,random_state=0)

# Importing SKLearn Multi Layer Perceptron Classifier........
from sklearn.neural_network import MLPClassifier

# Parameters of this function are:
# 1. hidden_layer_sizes: It takes a tuple which holds the values of the no of nodes each layer will have.
# e.g.1:- (100,):- One Layer with 100 nodes. Default Value.
# e.g.2:- (100,200):- Two Layers with 100 and 200 nodes respectively.
# e.g.3:- (100,20,300):- Three Layers with 100, 20 and 300  nodes respectively.
# 2. activation: The activation function to use. Default Value: relu function
# 3. alpha:- Regularization factor. Default value: 0.0001
# 4. batch_size:- The size of the batches that we have. Default Value:auto
#               For example while using Stochastic Gradient Descent training it in batches. Changing the weights value after some training point batch.
# 4. max_iter:- No. of iterations in Gradient Descent. Default Value: 200
# Note:-
# Maximum iterations(200) reached and the optimization hasn't converged yet.
# This means that even after runnning 200 iterations of Gradient Descent we have not reached the optimal/least cost value.

clf=MLPClassifier(hidden_layer_sizes=(20,),max_iter=3000)

clf.fit(x_train,y_train)                    # Fitting the Data.....

# To get the all the weights of the neural network use this(don't include the biases):
print('All the Weights of Neural Network: ')
# print(len(clf.coefs_))                # No. of layers in Neural Network
# print((clf.coefs_[0]).shape)          # The weights between input layer and first layer
print(clf.coefs_)
print('All the Biases of Neural Network: ')
print(clf.intercepts_)

print('Score of SKlearn Neural Network is:',clf.score(x_test,y_test))



