import torch

# Pytorch works primarily on tensors, 
# Numpy- Arrays and vectors

#  Tensors can be 1-D, 2-D, 3-D, ....


x=torch.empty(1)       #  Creating an empty tensor
print(x)

x=torch.empty(1,2,3)    # Creating a multi-dimentsional tensor
print(x)


# Tensor with random values
x=torch.rand(2,2)


# Tensor with zeros or ones
x=torch.zeros(2,2)
x=torch.ones(1,1)


# Checking datatype, default data type is float 
print(x.dtype)

# Creating tensor with specific dtype 
x=torch.ones(1,1, dtype=torch.int32)


# Check size of Tensor
x.size()

# Constrcut tensor from list
x=torch.tensor([0,1,2,3,4])
print(x)


# Basic Operations on Tensors
x1=torch.rand(2,2)
x2=torch.rand(2,2)
x3=x1+x2
x3=torch.add(x1,x2)
print('x1+x2=x3'+x1+"+"+x2+"="+x3)

# Add value to x, an change value(_ after function makes changes in variable)
x2.add_(x1)

# Subtraction
x4=x2-x1
x4=torch.sub(x1,x2)


# Multiplication
x5=x2*x1
x5=torch.mul(x1,x2)

# Division
x6=x2/x1
x6=torch.div(x1,x2)