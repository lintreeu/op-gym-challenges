#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

__global__ void rms_norm_forward(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    float* __restrict__ inv_rms,
    int N,
    float eps
) {
    int row = blockIdx.x;
    const float* x_row = x + row * N;
    float* y_row = y + row * N;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    __shared__ float mean_sq;
    float thread_sum = sum_sq;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    if (threadIdx.x == 0) mean_sq = thread_sum / N;
    __syncthreads();

    float rrms = rsqrtf(mean_sq + eps);
    if (threadIdx.x == 0) inv_rms[row] = rrms;

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        y_row[i] = x_row[i] * rrms * w[i];
    }
}