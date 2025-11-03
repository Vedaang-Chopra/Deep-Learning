# Understanding tensorflow language.........................

import tensorflow as tf

# To create Variables :

a=tf.Variable(100)              # The value needs to be passed as parameter
b=tf.Variable(200)
print(a)
# Note:- 1. To evaluate the value of variables, we cant' do that until and unless we explicitly initialize the values of variables.
#        2. These are global variables. So that is why a global_initializer is called. To create local variables use variable scope.
#           Pass this scope to Variable function. Use it similar to namespace concept of C++. The scope name is just a string.
sess=tf.Session()
# This is done so as to initialize the values of all variable that are passed as parameter.
sess.run(tf.global_variables_initializer())
print(sess.run(a+b))

# We can assign/change the values of variable, using the assign function.
# This doesn't change the value just calls the assign function. We would have to run this command using session.run, or run the tensor object returned from this command.
a.assign(1234)
sess.run(a.assign(1234))
assigned_obj=b.assign(2323)
sess.run(assigned_obj)
print(sess.run(a+b))