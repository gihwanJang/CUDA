#include "cnn/cudaLayer.h"

// layer 생성자
CudaLayer::CudaLayer(int M, int N, int O)
{
    this->M = M; // 입력
    this->N = N; // 출력
    this->O = O; // 뉴런

    float h_bias[N]; // host 편향
    float h_weight[N][M]; //host 가중치

    output = NULL;
    preact = NULL;
    bias = NULL;
    weight = NULL;

    for (int i = 0; i < N; ++i)
    {
        // 편향 값을 -0.5 ~ 0.5 난수 초기화
        h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

        // 가중치 값을 -0.5 ~ 0.5 난수 초기화
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
CudaLayer::~CudaLayer()
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
void CudaLayer::setOutput(float *data)
{
    cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
}

// 출력, 활성화 입력 초기화
void CudaLayer::clear()
{
    cudaMemset(output, 0, sizeof(float) * O);
    cudaMemset(preact, 0, sizeof(float) * O);
}

// 기울기 값 초기화
void CudaLayer::bp_clear()
{
    cudaMemset(d_output, 0, sizeof(float) * O);
    cudaMemset(d_preact, 0, sizeof(float) * O);
    cudaMemset(d_weight, 0, sizeof(float) * M * N);
}

