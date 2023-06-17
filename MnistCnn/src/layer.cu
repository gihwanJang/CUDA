#include "cnn/layer.h"

// layer 생성자
Layer::Layer(int M, int N, int O)
{
    this->M = M;
    this->N = N;
    this->O = O;

    float h_bias[N];
    float h_weight[N][M];

    output = NULL;
    preact = NULL;
    bias = NULL;
    weight = NULL;

    for (int i = 0; i < N; ++i)
    {
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

        for (int j = 0; j < M; ++j)
            h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
    }

    cudaMalloc(&output, sizeof(float) * O);
    cudaMalloc(&preact, sizeof(float) * O);
    cudaMalloc(&bias, sizeof(float) * N);
    cudaMalloc(&weight, sizeof(float) * M * N);

    cudaMalloc(&d_output, sizeof(float) * O);
    cudaMalloc(&d_preact, sizeof(float) * O);
    cudaMalloc(&d_weight, sizeof(float) * M * N);

    cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
}

// layer 소멸자
Layer::~Layer()
{
    cudaFree(output);
    cudaFree(preact);
    cudaFree(bias);
    cudaFree(weight);

    cudaFree(d_output);
    cudaFree(d_preact);
    cudaFree(d_weight);
}

// host -> device 메모리 복사
void Layer::setOutput(float *data)
{
    cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// Reset GPU memory
void Layer::clear()
{
    cudaMemset(output, 0, sizeof(float) * O);
    cudaMemset(preact, 0, sizeof(float) * O);
}

void Layer::bp_clear()
{
    cudaMemset(d_output, 0, sizeof(float) * O);
    cudaMemset(d_preact, 0, sizeof(float) * O);
    cudaMemset(d_weight, 0, sizeof(float) * M * N);
}

