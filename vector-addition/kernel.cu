// main.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// vector_add.cu
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}