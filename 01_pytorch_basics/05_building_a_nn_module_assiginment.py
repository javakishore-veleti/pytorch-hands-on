import torch
torch.__version__

# This Python file is as assignment to 04_building_a_nn_module.py
# In this assignment we do not hardcode 256 in Linear(64,256)
# Above 256 is called no of parameters
# Instead of hardcoding we use Python's range function and iterate
# through by reducing minus one (-1) from 256 until we see a 
# Model lost of 0.0001

# Below is a greycolor image with 1 batch, 1 input and 64,64 size
inputs = torch.rand(1,1, 64,64)
outputs = torch.rand(1,2)
learning_steps = []

for no_of_parameters in range(256,1, -1):
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer_one = torch.nn.Linear(64, no_of_parameters)
            self.activation_one = torch.nn.ReLU()
            self.layer_two = torch.nn.Linear(no_of_parameters, no_of_parameters)
            self.activation_two = torch.nn.ReLU()
            self.shape_outputs = torch.nn.Linear(no_of_parameters * 64, 2)
        
        def forward(self, inputs):
            buffer = self.layer_one(inputs)
            buffer = self.activation_one(buffer)
            buffer = self.layer_two(buffer)
            buffer = self.activation_two(buffer)
            buffer = buffer.flatten(start_dim=1)
            return self.shape_outputs(buffer)

    model = Model()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for an_epoch in range(10000):
        optimizer.zero_grad()
        intermeddate_results = model(inputs)
        loss = loss_function(intermeddate_results, outputs)
        loss.backward()
        optimizer.step()
        gradients = 0.0
        for parameter in model.parameters():
            gradients += parameter.grad.data.sum()
        if abs(gradients) <= 0.0001:
            learning_steps.append((no_of_parameters, an_epoch, intermeddate_results ))
            break
        
print(f"learning_steps {learning_steps}")