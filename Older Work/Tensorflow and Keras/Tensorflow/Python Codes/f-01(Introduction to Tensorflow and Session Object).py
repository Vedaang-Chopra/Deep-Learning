# TensorFlow:- It is an optimized library that is used to create a neural networks. Sklearn's implementation of neural networks is not optimized.
# Open Source Library developed by Google for any kind of machine learning algorithm or deep learning algorithms.
# It is written in C and C++ and provides implementation in python.
# In this we write a bunch of code, which stores in the ememory and  is executed once we call run function on it.
# Tensor-flow has to be installed to conda interpreter,then needs to be imported.

import tensorflow as tf

# Creating Constant in Tensor flow........
a=tf.constant(2)
b=tf.constant(3)
# a is tensor object. It has parameters:
# Const: The no. of constant ever created by the user using tensor flow
# shape: The shape of constant created
# dtype: Data type of Constant
print(a)
# By adding two tensor objects the va;ue are not added, but only the add function is called
print(a+b)
# To add the values of two tensor constants a and b we have to create a Tesnorflow session.
# Then we have to execute this session.
sess=tf.Session()                      # Creating a session
print(sess.run(a+b))                   # Executing the session, which actually does all the computations.

a1=tf.constant([[3,3]])                 # 2-D array of shape 1*2
a2=tf.constant([[3],[3]])               # 2-D array of shape 2*1
print(tf.matmul(a1,a2))                             # Returns the Tensor Object as matrix multiplication object
print(sess.run((tf.matmul(a1,a2))))              # Performing the Matrix Multiplication on actual values.


# Understanding the Session object............................
# We can run call session.run() all the computations that we have or we can use it the other way.

print('Using the Eval Function.............')
# To use the eval function, we need to create a Session Block using with which has a default sesssion.
with tf.Session() as abc:
    # To add two tensor objects we could use the add function. Then using the eval function we could evaluate those values.
    # This is done as a replacement method to session.run()
    print(tf.add(a,b).eval())

