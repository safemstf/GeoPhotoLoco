import torch

print(torch.cuda.is_available())
print(torch.__version__)

# Check if CUDA is available
if torch.cuda.is_available():
    # Create two random tensors
    a = torch.rand(10000, 10000, device='cuda')
    b = torch.rand(10000, 10000, device='cuda')

    # Perform a matrix multiplication
    c = torch.matmul(a, b)

    # Print result to ensure the operation was completed
    print("Matrix multiplication result:", c)
    print("CUDA is working!")
else:
    print("CUDA is not available.")


