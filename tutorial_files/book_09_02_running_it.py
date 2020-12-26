"""
Building on book_09_01 file to actually show how this is run
"""

############# Full forward pass for one node ##################

# values, weights, biases 
x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

# values * weights
xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# add it together with the bias to get output
z = (xw0 + xw1 + xw2 + b)

# ReLU activation
y = max(z, 0)
print(f"First pass value: {z}")

###################### Calculating the derivatives #######################

# Assume that the neuron output receives '1'
dvalue = 1

# Use way simplified version as explained above
drelu_dx0 = dvalue * (1. if z>0 else 0.) * w[0]
drelu_dx1 = dvalue * (1. if z>0 else 0.) * w[1]
drelu_dx2 = dvalue * (1. if z>0 else 0.) * w[2]
drelu_dw0 = dvalue * (1. if z>0 else 0.) * x[0]
drelu_dw1 = dvalue * (1. if z>0 else 0.) * x[1]
drelu_dw2 = dvalue * (1. if z>0 else 0.) * x[2]
drelu_db = 1 # always one - no valariables to deal with

# New we have the gradients for each input
# Yes, this is technically 4 dimensional multi variable calculus
dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db= drelu_db

############################### Example time! ########################

# Input values
print("Input1 values:", w,b)

w[0] += -0.001*dw[0]
w[1] += -0.001*dw[1]
w[2] += -0.001*dw[2]
b += -0.001*db

print("Input2 values:", w,b)

xw0 = x[0] * w[0]
xw1 = x[1] * w[1]
xw2 = x[2] * w[2]

# add it together with the bias to get output
z = (xw0 + xw1 + xw2 + b)

# ReLU activation
y = max(z, 0)
print(f"Second pass value: {z}") # Wowza!


"""
Okay, why did this work?

The derivatives shows how the input affects the final output value. That is, at
a given point the weight will affect the output a given amount. In the case of
a full network, the final output is the loss function which we are trying to
minimize. Consider a 2D (x,y) function. y=5x. d/dx=5. So, if we want
to minimize the output we can do x+=-0.001*(5) And this would reduce the output
value by a small amount. We are doing the same thing, but with many, many times
more values that can affect the output in different ways
"""
