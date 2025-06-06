int main() {
    const int M = 2;
    const int N = 4;
    const float eps = 1e-5f;

    std::vector<float> h_x = {1, 2, 3, 4,  4, 3, 2, 1};   // 2x4
    std::vector<float> h_w = {1, 1, 1, 1};
    std::vector<float> h_y(M * N);
    std::vector<float> h_inv_rms(M);

    float *d_x, *d_w, *d_y, *d_inv_rms;
    cudaMalloc(&d_x, M * N * sizeof(float));
    cudaMalloc(&d_w, N * sizeof(float));
    cudaMalloc(&d_y, M * N * sizeof(float));
    cudaMalloc(&d_inv_rms, M * sizeof(float));

    cudaMemcpy(d_x, h_x.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    rms_norm_forward<<<M, 32>>>(d_x, d_w, d_y, d_inv_rms, N, eps);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_inv_rms.data(), d_inv_rms, M * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output Y:\n";
    for (int i = 0; i < M * N; ++i) {
        std::cout << h_y[i] << " ";
        if ((i + 1) % N == 0) std::cout << std::endl;
    }

    std::cout << "Inverse RMS per row:\n";
    for (int i = 0; i < M; ++i)
        std::cout << h_inv_rms[i] << " ";
    std::cout << std::endl;

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);
    cudaFree(d_inv_rms);
    return 0;
}