"""
See book_09_00 file for full explaination - this is just simplifying that to be
better
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

############## Now time to run it backwards #######################

"""
Calculating the final derivative and simplifying
originally: drelu_dx0 = drelu_dxw0 * dmul_dx0
where:
dmul_dx0 = w[0]
drelu_dx0 = drelu_dz * dsum_dxw0 
dsum_dxw0 = 1
drelu_dz = dvalue*(1. if z > 0 else 0)
Substitute everything ==> drelu_dx0 = dvalue * (1. if z>0 else 0.) * w[0]

Which means we can do everything in one quick and easy equation instead doing
each step individually
"""

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
