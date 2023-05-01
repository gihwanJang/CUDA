#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello(void){
    printf("hello CUDA %d!\n", threadIdx.x);
}

int main(void){
    hello<<<1,8>>>();
	cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }
    return 0;
}