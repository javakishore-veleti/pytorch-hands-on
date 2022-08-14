import torch
torch.__version__

# 1,1,64,64 means -> 1 batch, GrayScale image (64,64)
inputs = torch.rand(1,1,64,64)
# Binary Outputs
outputs = torch.rand(1,2)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_one = torch.nn.Linear(64,256)
        self.activation_one = torch.nn.ReLU()
        self.layer_two = torch.nn.Linear(256, 256)
        self.activation_two = torch.nn.ReLU()
        self.shape_outputs = torch.nn.Linear(16384,2)
        
    def forward(self, inputs):
        buffer = self.layer_one(inputs)
        buffer = self.activation_one(buffer)
        buffer = self.layer_two(buffer)
        buffer = self.activation_two(buffer)
        buffer = buffer.flatten(start_dim=1)
        return self.shape_outputs(buffer)
    
model = Model()    
results = model(inputs)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for i in range(10000):
    optimizer.zero_grad()
    results = model(inputs)
    loss = loss_function(results, outputs)
    loss.backward()
    optimizer.step()
    # now look for vanishing graidents
    gradients = 0.0
    for parameter in model.parameters():
        gradients += parameter.grad.data.sum()
    if abs(gradients) <= 0.00001:
        print(f"GRADIENTS {gradients}")
        print("Graidents vanished at iterator {0}".format(i))
        break
        
