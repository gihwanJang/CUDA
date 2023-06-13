#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUM_STREAMS 2

#define NUM_T_IN_B 1024
#define NUM_BLOCK (128 * 1024)
#define ARRAY_SIZE (NUM_T_IN_B * NUM_BLOCK)

__global__ void myKernel(int*_in, int*_out)
{
    int tId = blockDim.x * blockIdx.x + threadIdx.x;
    int temp = 0;
    
    for(int i = 0; i < 250; ++i)
        temp = (temp + _in[tId] * 5) % 10;
    
    _out[tId] = temp;
}

int main(int argc, char const *argv[])
{
    DS_timer timer(5);
    timer.setTimerName(0, "Single stream");
    timer.setTimerName(1, "Host -> Device");
    timer.setTimerName(2, "Kernel execution");
    timer.setTimerName(3, "Device -> Host");
    timer.setTimerName(4, "Multiple streams");

    int *in = NULL, *out = NULL, *out2 = NULL;

    cudaMallocHost(&in, sizeof(int) * ARRAY_SIZE);
    cudaMemset(in, 0, sizeof(int) * ARRAY_SIZE);

    cudaMallocHost(&out, sizeof(int) * ARRAY_SIZE);
    cudaMemset(out, 0, sizeof(int) * ARRAY_SIZE);

    cudaMallocHost(&out2, sizeof(int) * ARRAY_SIZE);
    cudaMemset(out2, 0, sizeof(int) * ARRAY_SIZE);

    int *dIn, *dOut;

    cudaMalloc(&dIn, sizeof(int) * ARRAY_SIZE);
    cudaMalloc(&dOut, sizeof(int) * ARRAY_SIZE);

    for(int i = 0; i < ARRAY_SIZE; ++i)
        in[i] = rand() % 10;

    timer.onTimer(0);

    timer.onTimer(1);
    cudaMemcpy(dIn, in, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);
    timer.offTimer(1);

    timer.onTimer(2);
    myKernel<<<NUM_BLOCK, NUM_T_IN_B>>>(dIn, dOut);
    cudaDeviceSynchronize();
    timer.offTimer(2);

    timer.onTimer(3);
    cudaMemcpy(out, dOut, sizeof(int) * ARRAY_SIZE, cudaMemcpyDeviceToHost);
    timer.offTimer(3);

    timer.offTimer(0);

    cudaStream_t stream[NUM_STREAMS];

    for(int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&stream[i]);

    timer.onTimer(4);
    int chunkSize = ARRAY_SIZE / NUM_STREAMS;

    for(int i = 0; i < NUM_STREAMS; ++i)
    {
        int offset = chunkSize * i;
        cudaMemcpyAsync(dIn + offset, in + offset, sizeof(int) * chunkSize, cudaMemcpyHostToDevice, stream[i]);
        myKernel<<<NUM_BLOCK / NUM_STREAMS, NUM_T_IN_B, 0, stream[i]>>>(dIn + offset, dOut + offset);
        cudaMemcpyAsync(out2 + offset, dOut + offset, sizeof(int) * chunkSize, cudaMemcpyDeviceToHost, stream[i]);
    }

    cudaDeviceSynchronize();
    timer.offTimer(4);

    for(int i = 0; i < ARRAY_SIZE; ++ i)
        if(out[i] != out2[i])
            printf("!\n");

    for(int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamDestroy(stream[i]);

    timer.printTimer();

	cudaFree(dIn);
	cudaFree(dOut);

	cudaFreeHost(in);
	cudaFreeHost(out);
	cudaFreeHost(out2);
    return 0;
}
