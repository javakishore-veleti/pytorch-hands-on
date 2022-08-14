import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
torch.__version__

# This is an image with "batch" of 1, with 1 color channel, 
# Assuming a grayscale with 64 * 64 pixels
inputs = torch.rand(1,1,64,64)
INPUTS_CLONE = torch.clone(inputs)
print(f"INPUTS_CLONE {INPUTS_CLONE} {INPUTS_CLONE.size()}")

# Binary classification, so 2 outputs possibilities
outputs = torch.rand(1,2)
OUTPUTS_CLONE = torch.clone(outputs)
print(f"OUTPUTS_CLONE {OUTPUTS_CLONE} {OUTPUTS_CLONE.size()}")

model = torch.nn.Sequential(
    torch.nn.Linear(64, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 2)
)
results = model(inputs)
print(f"results {results} {results.shape}")
loss = torch.nn.MSELoss()(results,outputs)
print(f"loss {loss} {loss.shape}")

model.zero_grad()
loss.backward()

learning_rate = 0.01
for parameter in model.parameters():
    parameter.data -= parameter.grad.data * learning_rate

outputs_after_learning = model(inputs)
loss_after_learning = torch.nn.MSELoss()(outputs_after_learning, outputs)
print(f"loss_after_learning {loss_after_learning} {loss_after_learning.shape}")
