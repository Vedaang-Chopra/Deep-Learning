# This Code is an python implementation of Neural Networks, with no hidden layers and 1 hidden layer (2 nodes).
# It explains how forward and backward propagation work in such neural network.
# The implemented function is AND operation, with neural network.

# Forward Propagation:-
#       The mathematical operations that we do( e.g:- function(Multiplication of weights and inputs and adding of biases)) to reach the output is Forward Propagation.
# Backward Propagation:-
#       The change in the weights by subtracting it with derivative that results in the least cost.
import numpy as np

# Implementing the
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0,0,0,1]]).T
# print(y.shape)

def sig(z):
    return 1/(1+np.exp(-z))

def derivative_sig(z):
    return sig(z)*(1-sig(z))



# This is an implementation of neural network with no hidden layer. That means at input layer it has 2 input nodes 1 bias node and 1 node at output layer.
# This is simple implementation of Logistic Regression on AND operation.
def neural_network_zero_hidden_layer(x,y):
    # The np.random.random returns weights between 0 and 1. We multiply the range with 2 and subtract it with 1 to make the range between -1 and 1.
    # The function takes a tuple that is the shape of resultant array, as input that is the no. of random numbers to be generated.
    # The shape of array is: (no of nodes in current layer, on of nodes in next layer)

    # Change this to get better result.
    learning_rate=0.1
    # For this : no. of nodes in current layer=2 (input layer), no, of nodes in next layer=1 (output layer)
    weights_input_layer = 2 * np.random.random((2, 1)) - 1

    # For this : no. of bias node in current layer=1 (input layer), no, of nodes in next layer=1 (output layer)
    weights_bias_node_input = 2 * np.random.random((1)) - 1

    # print('Random Weights of Bias Node:')
    # print(weights_bias_node_input)
    # print('Random Weights of Input Nodes:')
    # print(weights_input_layer)


    for i in range(0,10000):
        # Forward propagation (Calculated once using Random Weights):
        input_values = x
        # Here we use the dot product to calculate w1x1+w2x2+b for the output node
        output = sig(np.dot(input_values, weights_input_layer) + weights_bias_node_input)
        # print('First Forward Propagation Result:')
        # print(output)
        # Backward Propagation
        # (This is done for fixed no. of iterations) similar to gradient descent:
        # The derivatives has 3 parts. We will calculate them separately.
        first_part=output-y                                     # y_pred-y_actual
        # This value is just w1x1+w2x2+b.
        input_last_for_last_layer=np.dot(input_values,weights_input_layer)+weights_bias_node_input
        second_part = derivative_sig(input_last_for_last_layer)  # sig(z)*[1-sig(z)]
        # print('Shape of Derivative First Part:',first_part.shape)
        # print('Derivative First Part Value:')
        # print(first_part)
        # print('Shape of Derivative Second Part:',second_part)
        # print('Derivative Second Part Value:')
        # print(second_part)

        # Still incomplete........
        final_derivative_value=first_part*second_part
        # print('Shape of Product of first and second part is:',final_derivative_value.shape)

        # This will hold the value of weights that will have to be subtracted from the original random weights to get to best result.
        # They need to have the shape same as the weights declared earlier.
        weights_cost_function=np.array([[0.0],[0.0]])
        # print('Changing weights array shape',weights_cost_function.shape)
        # Here 2 and 4 are dependent on input x.


        # for i in range(0,len(input_values[0])):
        #     for j in range(0,len(x)):
        #         weights_cost_function[i][0]+=final_derivative_value[j][0]*input_values[j][i]

        # This for loop can be replaced by the following code:
        weights_cost_function=np.dot(input_values.T,final_derivative_value)

        # This will hold the value of bias weights that will have to be subtracted from the original random bias weights to get to best result.
        # They need to have the shape same as the bias weights declared earlier.
        bias_cost_function = np.array([0.0])

        # for j in range(0,len(input_values)):
        #     bias_cost_function[0]+=final_derivative_value[j][0]*1

        bias_cost_function =np.sum(final_derivative_value)

        # print('Decreased Weights Value/Derivative Final Value for weights:')
        # print(weights_cost_function)
        # print('Decreased Bias value/Derivative Final value for bias:')
        # print(bias_cost_function)

        # Change in the weights value
        weights_input_layer=weights_input_layer-learning_rate*weights_cost_function
        weights_bias_node_input=weights_bias_node_input-learning_rate*bias_cost_function

    # print('Final weights:')
    # print(weights_input_layer)
    # print('Final Biases:')
    # print(weights_bias_node_input)
    # output=sig(np.dot(x,weights_input_layer)+weights_bias_node_input)
    # print('Final Output/ Y_predicted:')
    # print(output)
    return output

# The output will not be correct all the time because of low training points
output=neural_network_zero_hidden_layer(x,y)
for i in output:
    if i<0.5:
        print(0)
    else:
        print(1)
# c1,c2=0,0
# for i in range(0,100):
#     output=neural_network_zero_hidden_layer(x,y)
#     if output[3]>0.5:
#         c1+=1
#     else:
#         c2+=1
# print(c1,c2)
# # for i in output:
# #     if i<0.5:
# #         print(0,end=',')
# #     else:
# #         print(1,end=',')
