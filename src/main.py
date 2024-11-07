import triton
import triton.language as tl
import torch
import time

@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr
):
    # Block indices
    pid = tl.program_id(axis=0)
    rm = pid // (N // BLOCK_SIZE)
    rn = pid % (N // BLOCK_SIZE)

    # Set up pointers for A and B based on block size
    a_ptr = A_ptr + rm * BLOCK_SIZE * stride_am + tl.arange(0, BLOCK_SIZE)[:, None] * stride_ak
    b_ptr = B_ptr + rn * BLOCK_SIZE * stride_bn + tl.arange(0, BLOCK_SIZE)[None, :] * stride_bk
    
    # Initialize accumulator for the block product
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Manually compute the dot product without `tl.dot`, looping over the K dimension
    for k in range(K):
        # Load one element from each row of A and one from each column of B, then multiply
        a = tl.load(a_ptr + k * stride_ak, mask=tl.arange(0, BLOCK_SIZE)[:, None] < M, other=0.0)
        b = tl.load(b_ptr + k * stride_bk, mask=tl.arange(0, BLOCK_SIZE)[None, :] < N, other=0.0)
        acc += a * b

    # Define the output pointer for storing the block
    c_ptr = C_ptr + rm * BLOCK_SIZE * stride_cm + rn * BLOCK_SIZE * stride_cn
    # Use `tl.store` to store the block `acc` to `c_ptr`
    tl.store(c_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * stride_cm + tl.arange(0, BLOCK_SIZE)[None, :] * stride_cn, acc)

def pad_matrix(matrix, multiple_of=16):
    # Calculate new dimensions, rounding up to the nearest multiple of `multiple_of`
    new_shape = [
        ((dim + multiple_of - 1) // multiple_of) * multiple_of for dim in matrix.shape
    ]
    # Create a padded matrix and copy the original data
    padded_matrix = torch.zeros(new_shape, device=matrix.device, dtype=matrix.dtype)
    padded_matrix[:matrix.shape[0], :matrix.shape[1]] = matrix
    return padded_matrix

def triton_gemm(A, B, BLOCK_SIZE=16):
    # Ensure input tensors are on the GPU
    A, B = A.cuda(), B.cuda()
    
    # Pad matrices A and B if needed
    A_padded = pad_matrix(A, multiple_of=BLOCK_SIZE)
    B_padded = pad_matrix(B, multiple_of=BLOCK_SIZE)
    
    # Get the new dimensions of the padded matrices
    M, K = A_padded.shape
    K, N = B_padded.shape
    
    # Prepare the output tensor with the padded dimensions
    C_padded = torch.empty((M, N), device="cuda", dtype=A.dtype)
    
    # Launch the Triton kernel
    gemm_kernel[(M * N // BLOCK_SIZE,)](
        A_padded, B_padded, C_padded,
        M, N, K,
        A_padded.stride(0), A_padded.stride(1),
        B_padded.stride(0), B_padded.stride(1),
        C_padded.stride(0), C_padded.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Extract the original-sized result
    C = C_padded[:A.shape[0], :B.shape[1]]
    return C

def benchmark_gemm(A, B, func):
    # Ensure all operations are finished before starting the timer
    torch.cuda.synchronize()
    start = time.time()
    func(A, B)
    torch.cuda.synchronize()
    return time.time() - start

### MAIN ---------------------------------------------------------

torch.cuda.empty_cache()

M, K, N = 256, 256, 256
A = torch.randn((M, K), device="cuda", dtype=torch.float32)
B = torch.randn((K, N), device="cuda", dtype=torch.float32)

# Compute with Triton GEMM
C_triton = triton_gemm(A, B)

# Compute with torch.matmul (PyTorch's built-in CUDA GEMM)
C_torch = torch.matmul(A, B)

# Check if the results are close enough
assert torch.allclose(C_triton, C_torch, atol=1e-5), "Triton GEMM does not match torch.matmul"
print("Validation passed: Triton GEMM output matches torch.matmul output.")

### BENCHMARKING ---------------------------------------------------------

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
