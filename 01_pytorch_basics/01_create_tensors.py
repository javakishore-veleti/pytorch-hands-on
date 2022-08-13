import torch
torch.__version__

# 01. Create Empty Tensors
empty_tensors = torch.empty(2,2)
print(empty_tensors)

# 02. Create Tensors with random numbers
random_num_tensors = torch.rand(2,2)
print(random_num_tensors)

# 03. Create Tensors with zeroes
zeros_tensors = torch.zeros(2,2)
print(zeros_tensors)

specific_constant_tensors = torch.full([2,2], 3)
print(specific_constant_tensors)

tensors_from_arrays = torch.tensor([[2,2],[5,5]])
print(tensors_from_arrays)