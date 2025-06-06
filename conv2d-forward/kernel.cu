#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>

__global__ void conv2d_forward(
    const float* __restrict__ input,      // [N, C, H, W]
    const float* __restrict__ weight,     // [F, C, KH, KW]
    const float* __restrict__ bias,       // [F]
    float* __restrict__ output,           // [N, F, OH, OW]
    int N, int C, int H, int W,
    int F, int KH, int KW,
    int OH, int OW,
    int stride, int padding
) {
    int n = blockIdx.x;
    int f = blockIdx.y;
    int oh = threadIdx.y;
    int ow = threadIdx.x;

    if (oh >= OH || ow >= OW) return;

    float sum = bias ? bias[f] : 0.0f;
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
            for (int kw = 0; kw < KW; ++kw) {
                int ih = oh * stride - padding + kh;
                int iw = ow * stride - padding + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int input_idx = ((n * C + c) * H + ih) * W + iw;
                    int weight_idx = ((f * C + c) * KH + kh) * KW + kw;
                    sum += input[input_idx] * weight[weight_idx];
                }
            }
        }
    }

    int output_idx = ((n * F + f) * OH + oh) * OW + ow;
    output[output_idx] = sum;
}
