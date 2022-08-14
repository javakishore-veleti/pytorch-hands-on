import torch
torch.__version__

# Create a Liner Regression Equation Y = a + Xb
# Assume a Real Observation Value of Y 3
# Now we need to build a model and determine the lost
# i.e. graident i.e. how much predicted value is differing
# from the observation value 3. 
# For the above we will be using MeanSquaredError (MSE)
X = torch.rand(1, requires_grad=True)
Y = X + 0.1
XCOPY = torch.clone(X)
YCOPY = torch.clone(Y)
print(f"XCOPY {XCOPY}")
print(f"YCOPY {YCOPY}")

def mse(Y):
    diff = 3 - Y
    return (diff * diff).sum() / 2 # This is Mean Squared Error

loss = mse(Y)
loss.backward()
print(X.grad)

# Applying the learning rate
learning_rate = 1e-3

# Running through the various epochs
for an_epcoh in (0,1000):
    Y = X + 1.0
    loss = mse(Y)
    loss.backward()
    
    with torch.no_grad():
        X -= learning_rate * X.grad
        X.grad.zero_()
        
print(X)
print(Y) 

