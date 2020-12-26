"""
Using derivatives to run the process backwards for a single neuron
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
But first, let's take a look at what we are doing. For this example, we want to
see what affect the w[0] value has on the final output. First, we need to get
the whole equation

y = ReLU(sum(mul(x0, w0), mul(x1, w1), mul(x2,w2)))

To get change in y for a change in w0, we need to do [d/dsum() ReLU]*[d/dmul()
sum()] * [d/dx0 mul(x0, w0)]. Okay - I know how ugly that looks, but stick with
me. You just take the derivative of the activation function (ReLU) * derivative
of the sum function * derivative of the multiplication.

Okay, but why? In the full network we will calculate the derivative of the
loss, then use this as starting point to run backwards through the network.
This will give us gradient values for all of the weights and biases that we can
use as a starting point for adjustments

Finally, all nerons are chained together in a similar fashion on the individual
neron. In the example below we are assuming that the value received is the
already calculated derivative so we can just keep on multiplying the value
through
"""

# Assume that the neuron output receives '1'
dvalue = 1

# Derivative of the activation function * received value
drelu_dz = dvalue * (1. if z > 0 else 0.0)

print(drelu_dz)

"""
Now we need to differentiate the sum function
Derivative of a sum is always one - that was easy. This is because adding the
value together has no affect on the slope of the final output. Similar to how
the activation function either does or does not work - it doesn't change the
actual slope of the final output
"""

dsum_dxw0 = 1 # derivative of sum w.r.t [0] inputs / weights
drelu_dxw0 = drelu_dz * dsum_dxw0 # working backwards via chain rule

# Do the same thing for rest of inputs and the bias
dsum_dxw1 = 1 
drelu_dxw1 = drelu_dz * dsum_dxw1 

dsum_dxw2 = 1 
drelu_dxw2 = drelu_dz * dsum_dxw2 

dsum_db = 1
drelu_db = drelu_dz * dsum_db


"""
Finally, we can do the derivative of the multiplication functions
Okay, technically we are getting into multivaraible calculuse territory here,
but don't panic and have college flashbacks just yet. We are multiplying two
values, which means their derivatives are:
d/dx[x*y] = y
d/dy[x*y] = x
You know, set all other variables constant, do derivative? Easy, peasy.
"""

# d/dx[w0*x0] = w0 ==> so just use w0
dmul_dx0 = w[0]
drelu_dx0 = drelu_dxw0 * dmul_dx0

print(drelu_dx0)

# Okay, now calculate derivative for the rest of the inputs - x and w

dmul_dx1 = w[1]
dmul_dx2 = w[2]
dmul_dw0 = x[0]
dmul_dw1 = x[1]
dmul_dw2 = x[2]

drelu_dx1 = drelu_dxw1 * dmul_dx1
drelu_dx2 = drelu_dxw2 * dmul_dx2

drelu_dw0 = drelu_dxw0 * dmul_dw0
drelu_dw1 = drelu_dxw1 * dmul_dw1
drelu_dw2 = drelu_dxw2 * dmul_dw2

print(drelu_dx0, drelu_dx1, drelu_dx2, drelu_dw0, drelu_dw1, drelu_dw2)
