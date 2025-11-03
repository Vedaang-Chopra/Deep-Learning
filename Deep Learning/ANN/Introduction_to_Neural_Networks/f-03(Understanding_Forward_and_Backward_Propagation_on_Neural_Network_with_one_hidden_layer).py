# This Code is an python implementation of Neural Networks, with  1 hidden layer (2 nodes).
# It explains how forward and backward propagation work in such neural network.
# The implemented function is XOR operation, with neural network.

# Forward Propagation:-
#       The mathematical operations that we do( e.g:- function(Multiplication of weights and inputs and adding of biases)) to reach the output is Forward Propagation.

import numpy as np

# Implementing the neural network
x=np.array([[0,0],[0,1],[1,0],[1,1]])
y=np.array([[0,1,1,0]]).T
# print(y.shape)

def sig(z):
    return 1/(1+np.exp(-z))

def derivative_sig(z):
    return sig(z)*(1-sig(z))

# This is an implementation of neural network with 1 hidden layer. It is an implementation of neural network on XOR operation.
# That means at: The Input layer it has 2 input nodes 1 bias node.
# The Hidden layer has 2 nodes and 1 bias node.
# The output layer has 1 node at output layer.

def neural_network_one_hidden_layer(x,y):
    # The np.random.random returns weights between 0 and 1. We multiply the range with 2 and subtract it with 1 to make the range between -1 and 1.
    # The function takes a tuple that is the shape of resultant array, as input that is the no. of random numbers to be generated.
    # The shape of array is: (no of nodes in current layer, no of nodes in next layer)
    learning_rate=0.25
    # For this : no. of nodes in current layer=2 (input layer), no, of nodes in next layer=2 (hidden layer)
    # weights_hidden_layer = 2*np.random.random((2,2))-1
    weights_hidden_layer=np.array([[0.6,-0.3],[-0.1,0.4]])

    # For this : no. of bias node in current layer=1 (input layer), no, of nodes in next layer=2 (hidden layer)
    # weights_bias_node_hidden_layer = 2 * np.random.random((1,2)) - 1
    weights_bias_node_hidden_layer=np.array([[0.3,0.5]])
    # For this : no. of nodes in current layer=2 (hidden layer), no, of nodes in next layer=1 (output layer)
    # weights_output_layer = 2 * np.random.random((2,1)) - 1
    weights_output_layer=np.array([[0.4],[0.1]])
    # For this : no. of bias node in current layer=1 (hidden layer), no, of nodes in next layer=1 (output layer)
    # weights_bias_node_output_layer = 2 * np.random.random((1,1)) - 1
    weights_bias_node_output_layer =np.array([-0.2])
    # print('Random Weights of Bias Node of Input Layer(connecting to  hidden layer):')
    # print(weights_bias_node_hidden_layer)
    # print('Random Weights of Input Nodes(connecting to  hidden layer):')
    # print(weights_hidden_layer)
    # print('Random Weights of Bias Node of Hidden Layer(connecting to output layer):')
    # print(weights_bias_node_output_layer)
    # print('Random Weights of Input Nodes of Hidden Layer(connecting to  output layer):')
    # print(weights_output_layer)

    for i in range(0, 1):

            # Forward propagation:
            input_values=x
            # Performing Dot Product of inputs with weights and adding biases to get hidden layer output .........
            input_hidden_layer =np.dot(input_values,weights_hidden_layer)+weights_bias_node_hidden_layer
            output_hidden_layer=sig(input_hidden_layer)
            # print('Output of hidden layer:')
            # print(output_hidden_layer)
            input_output_layer=np.dot(output_hidden_layer,weights_output_layer)+weights_bias_node_output_layer
            output_output_layer=sig(input_output_layer)
            # print('First Forward Propagation Result:')
            # print(output_layer_result)

            # Backward Propagation
            # (This is done for fixed no. of iterations) similar to gradient descent:
            # The derivatives has 3 parts. We will calculate them separately.
            first_part_output_layer = output_output_layer - y  #            y_pred-y_actual
            # This value is just w1x1+w2x2+b.
            input_last_for_last_layer = np.dot(output_hidden_layer,weights_output_layer)+weights_bias_node_output_layer
            second_part_for_output_layer = derivative_sig(input_last_for_last_layer)  # sig(z)*[1-sig(z)]

            # print('Shape of Derivative First Part:',first_part.shape)
            # print('Derivative First Part Value:')
            # print(first_part)
            # print('Shape of Derivative Second Part:',second_part)
            # print('Derivative Second Part Value:')
            # print(second_part)

            # Still incomplete........
            final_derivative_value_first_two_part_output_layer = first_part_output_layer * second_part_for_output_layer
            # print('Shape of Product of first and second part is:',final_derivative_value.shape)
            first_part_hidden_layer=np.dot(final_derivative_value_first_two_part_output_layer,weights_output_layer.T)
            # print(first_part_hidden_layer)
            second_part_for_hidden_layer=derivative_sig(input_hidden_layer)

            final_derivative_value_first_two_part_hidden_layer=first_part_hidden_layer*second_part_for_hidden_layer

            # This will hold the value of weights that will have to be subtracted from the original random weights to get to best result.
             # They need to have the shape same as the weights declared earlier.
            weights_cost_function_output_layer = np.dot(output_hidden_layer.T, final_derivative_value_first_two_part_output_layer)

            # This will hold the value of bias weights that will have to be subtracted from the original random bias weights to get to best result.
            # They need to have the shape same as the bias weights declared earlier.
            # keepdims: Keeps the dimension of array, because np.sum returns a number
            # axis:- Keeps the axis = row

            bias_cost_function_output_layer = np.sum(final_derivative_value_first_two_part_output_layer,keepdims=True,axis=0)
            print('Decreased Weights Value/Derivative Final Value for weights:')
            print(weights_cost_function_output_layer)
            print('Decreased Bias value/Derivative Final value for bias:')
            print(bias_cost_function_output_layer)

            weights_cost_function_hidden_layer = np.dot(input_values.T, final_derivative_value_first_two_part_hidden_layer)

            # This will hold the value of bias weights that will have to be subtracted from the original random bias weights to get to best result.
            # They need to have the shape same as the bias weights declared earlier.

            bias_cost_function_hidden_layer = np.sum(final_derivative_value_first_two_part_hidden_layer, keepdims=True, axis=0)
            print('Decreased Weights Value/Derivative Final Value for weights:')
            print(weights_cost_function_hidden_layer)
            print('Decreased Bias value/Derivative Final value for bias:')
            print(bias_cost_function_hidden_layer)

            weights_output_layer = weights_output_layer - learning_rate * weights_cost_function_output_layer
            weights_bias_node_output_layer = weights_bias_node_output_layer - learning_rate * bias_cost_function_output_layer

            weights_hidden_layer = weights_hidden_layer - learning_rate * weights_cost_function_hidden_layer
            weights_bias_node_hidden_layer = weights_bias_node_hidden_layer - learning_rate * bias_cost_function_hidden_layer
            # Change in the weights value
            print('Final weights Hidden Layer:')
            print(weights_hidden_layer)
            print('Final Biases Hidden Layer:')
            print(weights_bias_node_hidden_layer)
            output_hidden_layer=sig(np.dot(x,weights_hidden_layer)+weights_bias_node_hidden_layer)
            print('Final weights Output Layer:')
            print(weights_output_layer)
            print('Final Biases Output Layer:')
            print(weights_bias_node_output_layer)
            output_output_layer = sig(np.dot(x, weights_output_layer) + weights_bias_node_output_layer)
            print('Final Output/ Y_predicted:')
            print(output_output_layer)
    output_output_layer=sig(input_output_layer)
    return output_output_layer

final_output=neural_network_one_hidden_layer(x,y)
for i in final_output:
    if i<0.5:
        print(0)
    else:
        print(1)
