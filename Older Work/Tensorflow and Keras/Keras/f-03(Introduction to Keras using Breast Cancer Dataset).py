# A Basic Implementation of Keras.......................

# This code of Keras creates a basic neural network, and predicts on the breast cancer dataset.
# Keras is a framework/layer that is implemented over the tensorflow.
# It automates the forward and backward propagation, creating the layers and nodes, creating the optimizer and loss function, specifying batch sizes and metrics etc.

# Steps for creating a neural network using keras:
# Step1:- Creating a model. Models are either Sequential or Functional.
# Step2:- Defining the architecture of the model, that means defining the no of layers, no of units in a layer, the activation function of that layer etc.
# Step3:- Compiling the model, that includes specifying the loss function and the error function, specifyng the no. of epochs/iterations, checking metrics on validation data etc.
# Step4:- Training the model on the Dataset. It includes passing the model the x_train, y_train values, the batch size of training data
#         for learning that is doing the forward and backward propagation.
# Step5:- Predicting the output on testing data. Checking the metrics etc.


# Importing the basic libraries for creating a neural network..........
from keras.models import Sequential
from keras.layers import Dense

# Creating a Model.......
# A model is a basic structure that is created to hold the entire neural network(a box around the neural network). It holds the layers that are added to it.
# There are two types of models, the first is the Sequential model and the next is the FunctionalAPI model.
# Sequential Model:- It is the basic sense of model used everywhere, where output of previous layer goes as input to next layer. Doesn't require any parameter.
# Functional API:- Used to create complex models. e.g Input from two layers goes into next layer.

# Step1: Creating the Model........................
model = Sequential()

# Layers in Keras.........
# The Layers in Keras are the layers of neural network. So before adding layers we need to define the architecture of the neural network.
# Every layer has some common parameters that it needs (except if it is a specific type of layer):
# 1. Activation Function: Function applied on the input to node. If no activation function is applied the Identity function/No function is applied.
# 2. No. of Units: The no. of nodes in a layer. No.of Units always needed to be provided.
# 3. use_bias: A boolean value that is provided to specify whether bias is needed or not.
# 4. Regularization factor: The regularization factor with the prevents overfitting and is added to the cost function. Regulaization factor can be set for weights or  biases or both.
# 5. Initialization factor: Initial initializing of wights and biases. e.g. whether it is to random or constant.
# 6. Constraints: whether any constarint on the biases. e.g:- weights can't be negative etc.

# There are two layers common to every neural network:
# 1. Input Layer:- This is the first layer of the model. It is a simple layer/Dense Layer that accepts the features of training data as input.
#                  Its shape has to be defined always by the user and is equal to the number of features as the dataset.
#                  In this layer we don't have to provide no. of units but rather input shape.

# 2. Output Layer:- This is the last layer of the model. It is a simple layer/Dense Layer that gives the value/ predicts the output from the given input.
#                   Its shape need not be defined by user, but is equal to the no. of classes present in the dataset if classes>=3 and 1 if classes<3
# There are other layers that are present in the model:
# 1. Dense Layer: Default neural network layer is called Dense Layer. It is a Basic/Simple/Dense Layer that is used to perform computations
# 2. Convolution Layer: Layer with convolution function
# 3. Recurrent Layer etc.....

# Step2:- Defining the Architecture of the Model.......................................
# Adding layers to model........
# The first layer is a Dense and input layer.
layer1 = Dense(units=32, activation = 'relu', input_dim = 30)                   # Creating a Dense Layer
model.add(layer1)                                                               # Adding layer to model.
model.add(Dense(units=16, activation = 'relu'))
model.add(Dense(units=1, activation = 'sigmoid'))

# Step3:- Compiling the Model.......
# These 3 Parameters need to be specified to model using compilation:
# 1. Optimizer: e.g Adam Optimizer, Gradient Descent etc....
# 2. Loss Function: The Cost Function.g Binary_crossentropy / multiclass_cross_entropy etc.
# 3. Metrics: Accuracy for Classification problem, Score for Regression problem etc.

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Importing the dataset and splitting the data into training and testing.............................
from sklearn import datasets
cancer = datasets.load_breast_cancer()
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.2, random_state = 0)

# Scaling the Breast Cancer Dataset.....................
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Step4:- Fitting the Dataset onto the model...................
# The fit function requires some parameters:
# 1. The x_train and the y_train values: Training Dataset
# 2. Epochs:- The no. of times the gradient descent is run/no. of iterations. Default size is 1
# 3. Batch Size:- The batch size of training data
# 4. Validation data:- During training to view the data metrics,that is the accuracy and loss we use this data. It doesn't use this to train the model/.

model.fit(x_train, y_train, epochs=20, batch_size = 50, validation_data=(x_test, y_test))

# Step5:- Predicting the values using the model.........
# The predict function predicts the values for the given x values.
predictions = model.predict(x_test)
# The evaluate function presents the same metrics as that used during compilation. e.g:- Here it is loss and accuracy.
score = model.evaluate(x_test, y_test)
print(score)