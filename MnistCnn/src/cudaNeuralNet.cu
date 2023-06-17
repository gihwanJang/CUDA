#include "cnn/cudaNeuralNet.h"

// 시그모이드 함수
__device__ float cu_activ_func(float v)
{
    return 1 / (1 + exp(-v));
}

// 입력 값에 대하여 활성화 함수 적용
__global__ void cu_apply_activ_func(float *input, float *output, int N)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    for (int i = N * t_id / size; i < N * (t_id + 1) / size; ++i)
        output[i] = cu_activ_func(input[i]);
}

// 오차 계산
__global__ void cu_update_error(float *err, float *output, unsigned int Y, int N) 
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    for (int i = N * t_id / size; i < N * (t_id + 1) / size; ++i)
        err[i] = ((Y == i ? 1.0f : 0.0f) - output[i]);
}

// 기울기 계산
__global__ void cu_update_grad(float *output, float *grad, int N)
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    for (int i = N * t_id / size; i < N * (t_id + 1) / size; ++i)
        output[i] += dt * grad[i];
}

// convolution layer에 대한 forward propagation, 해당 레이어의 기울기 -> 입력 레이어의 output에 대한 5 x 5 가중치 곱셈
__global__ void cu_fp_preact_conv(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 5 * 5 * 6 * 24 * 24;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 5);
        int i2 = ((idx /= 5) % 5);
        int i3 = ((idx /= 5) % 6);
        int i4 = ((idx /= 6) % 24);
        int i5 = ((idx /= 24) % 24);

        atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
    }
}

// convolution layer에 대한 forward propagation, 해당 레이어에 바이어스 값 합산
__global__ void cu_fp_bias_conv(float preact[6][24][24], float bias[6])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 6 * 24 * 24;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 6);
        int i2 = ((idx /= 6) % 24);
        int i3 = ((idx /= 24) % 24);

        preact[i1][i2][i3] += bias[i1];
    }
}

// polling layer에 대한 forward propagation, 합성곱 레이어dml output에 대한 4 x 4 가중치 곱셈
__global__ void cu_fp_preact_poll(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 4 * 4 * 6 * 6 * 6;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 4);
        int i2 = ((idx /= 4) % 4);
        int i3 = ((idx /= 4) % 6);
        int i4 = ((idx /= 6) % 6);
        int i5 = ((idx /= 6) % 6);

        atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
    }
}

// polling layer에 대한 forward propagation, 해당 레이어에 바이어스 값 합산
__global__ void cu_fp_bias_poll(float preact[6][6][6], float bias[1])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 6 * 6 * 6;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 6);
        int i2 = ((idx /= 6) % 6);
        int i3 = ((idx /= 6) % 6);

        preact[i1][i2][i3] += bias[0];
    }
}

// fully connected layer에 대한 forward propagation, polling 레이어에 대한 가중치 곱셈
__global__ void cu_fp_preact_full(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 10 * 6 * 6 * 6;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 10);
        int i2 = ((idx /= 10) % 6);
        int i3 = ((idx /= 6) % 6);
        int i4 = ((idx /= 6) % 6);

        atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
    }
}

// fully connected layer에 대한 forward propagation, 해당 레이어에 바이어스 값 합산
__global__ void cu_fp_bias_full(float preact[10], float bias[10])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 10;

    for (int i = N * t_id / size; i < N * (t_id + 1) / size; ++i)
        preact[i] += bias[i];
}

// fully connected layer에 대한 back propagation, 가중치를 계산하기 위해 활성화 함수 기울기와 해당 위치의 출력 활성화 값을 곱셈
__global__ void cu_bp_weight_full(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 10 * 6 * 6 * 6;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 10);
        int i2 = ((idx /= 10) % 6);
        int i3 = ((idx /= 6) % 6);
        int i4 = ((idx /= 6) % 6);

        d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
    }
}

// fully connected layer에 대한 바이어스 값 갱신
__global__ void cu_bp_bias_full(float bias[10], float d_preact[10])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 10;

    for (int idx = N * t_id / size; idx < N * (t_id + 1) / size; ++idx)
        bias[idx] += dt * d_preact[idx];
}

// polling layer에 대한 back propagation, 해당 레이어의 출력 값에 대한 flully connected의 가중치와 기울기의 곱의 합산
__global__ void cu_bp_output_poll(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 10 * 6 * 6 * 6;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 10);
        int i2 = ((idx /= 10) % 6);
        int i3 = ((idx /= 6) % 6);
        int i4 = ((idx /= 6) % 6);

        atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
    }
}

// polling layer에 대한 back propagation, 해당 레이어의 기울기 값에 대하여 해당 레이어의 출력과 활성화 함수를 거친 기울기 값의 곱셈
__global__ void cu_bp_preact_poll(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 6 * 6 * 6;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 6);
        int i2 = ((idx /= 6) % 6);
        int i3 = ((idx /= 6) % 6);

        float o = cu_activ_func(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

// polling layer에 대한 back propagation, 해당 레이어의 가중치에 대하여 해당 레이어의 기울기와 출력가의 곱의 합산
__global__ void cu_bp_weight_poll(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 1 * 4 * 4 * 6 * 6 * 6;
    float d = pow(6.0f, 3.0f);

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 1);
        int i2 = ((idx /= 1) % 4);
        int i3 = ((idx /= 4) % 4);
        int i4 = ((idx /= 4) % 6);
        int i5 = ((idx /= 6) % 6);
        int i6 = ((idx /= 6) % 6);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
    }
}

// polling layer에 대한 back propagation, 해당 레이어의 바이어스 값을 업데이트
__global__ void cu_bp_bias_poll(float bias[1], float d_preact[6][6][6])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 6 * 6 * 6;
    float d = pow(6.0f, 3.0f);

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 6);
        int i2 = ((idx /= 6) % 6);
        int i3 = ((idx /= 6) % 6);

        atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
    }
}

// convolution layer에 대한 back propagation, 해당 레이어의 출력에 대하여 poling layer의 가중치와 기울기 곱의 합
__global__ void cu_bp_output_conv(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 1 * 4 * 4 * 6 * 6 * 6;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 1);
        int i2 = ((idx /= 1) % 4);
        int i3 = ((idx /= 4) % 4);
        int i4 = ((idx /= 4) % 6);
        int i5 = ((idx /= 6) % 6);
        int i6 = ((idx /= 6) % 6);

        atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
    }
}

// convolution layer에 대한 back propagation, 해당 레이어의 기울기에 대하여 해당 레이어의 출력과 기울기의 곱으로 갱신
__global__ void cu_bp_preact_conv(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 6 * 24 * 24;

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 6);
        int i2 = ((idx /= 6) % 24);
        int i3 = ((idx /= 24) % 24);

        float o = cu_activ_func(preact[i1][i2][i3]);

        d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
    }
}

// convolution layer에 대한 back propagation, 해당 레이어의 가중치에 대하여 기울기와 input layer의 출력 값으 곱의 합
__global__ void cu_bp_weight_conv(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 6 * 5 * 5 * 24 * 24;
    float d = pow(24.0f, 2.0f);

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 6);
        int i2 = ((idx /= 6) % 5);
        int i3 = ((idx /= 5) % 5);
        int i4 = ((idx /= 5) % 24);
        int i5 = ((idx /= 24) % 24);

        atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
    }
}

// convolution layer에 대한 back propagation, 해당 레이어의 바이어스 업데이트
__global__ void cu_bp_bias_conv(float bias[6], float d_preact[6][24][24])
{
    int t_id = blockIdx.x * blockDim.x + threadIdx.x;
    int size = blockDim.x * gridDim.x;

    int N = 6 * 24 * 24;
    float d = pow(24.0f, 2.0f);

    for (int n = N * t_id / size; n < N * (t_id + 1) / size; ++n)
    {
        int idx = n;
        int i1 = ((idx /= 1) % 6);
        int i2 = ((idx /= 6) % 24);
        int i3 = ((idx /= 24) % 24);

        atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
    }
}
