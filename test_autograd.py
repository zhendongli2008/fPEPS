import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad    # The only autograd function you may ever need

def lg(x):                 # Define a function
   y = x*x
   return np.log(y)

grad_tanh = grad(lg)       # Obtain its gradient function
print grad_tanh(1.0)               # Evaluate the gradient at x = 1.0
