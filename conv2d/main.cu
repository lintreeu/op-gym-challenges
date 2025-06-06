int main() {
    const int N = 1, C = 1, H = 5, W = 5;
    const int F = 1, KH = 3, KW = 3;
    const int stride = 1, padding = 1;
    const int OH = (H + 2 * padding - KH) / stride + 1;
    const int OW = (W + 2 * padding - KW) / stride + 1;

    std::vector<float> h_input(N * C * H * W, 1.0f);
    std::vector<float> h_weight(F * C * KH * KW, 1.0f);
    std::vector<float> h_bias(F, 0.0f);
    std::vector<float> h_output(N * F * OH * OW);

    float *d_input, *d_weight, *d_bias, *d_output;
    cudaMalloc(&d_input, h_input.size() * sizeof(float));
    cudaMalloc(&d_weight, h_weight.size() * sizeof(float));
    cudaMalloc(&d_bias, h_bias.size() * sizeof(float));
    cudaMalloc(&d_output, h_output.size() * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(N, F);
    dim3 blockDim(OW, OH);

    conv2d_forward<<<gridDim, blockDim>>>(
        d_input, d_weight, d_bias, d_output,
        N, C, H, W, F, KH, KW, OH, OW,
        stride, padding
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output:\n";
    for (int i = 0; i < OH * OW; ++i) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % OW == 0) std::cout << "\n";
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_output);
    return 0;
}

