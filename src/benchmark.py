import torch
from main import triton_gemm, M, K, N
import time

def benchmark_gemm(A, B, func):
    # Ensure all operations are finished before starting the timer
    torch.cuda.synchronize()
    start = time.time()
    func(A, B)
    torch.cuda.synchronize()
    return time.time() - start

# Initialize random input matrices
A = torch.randn((M, K), device="cuda", dtype=torch.float32)
B = torch.randn((K, N), device="cuda", dtype=torch.float32)

# Measure Triton GEMM time
triton_time = benchmark_gemm(A, B, triton_gemm)
print(f"Triton GEMM time: {triton_time:.6f} seconds")

# Measure PyTorch CUDA GEMM time
torch_time = benchmark_gemm(A, B, torch.matmul)
print(f"PyTorch GEMM time: {torch_time:.6f} seconds")

# Compare the results
print(f"Speedup: {torch_time / triton_time:.2f}x" if triton_time < torch_time else "No speedup")
