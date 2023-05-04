#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer/DS_definitions.h"
#include "timer/DS_timer.h"

#define DATA_SIZE 1024 * 1024 * 256

__global__ void vectorSum(int *_a, int *_b, int *_res, long _size){
    long tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID < _size)
        _res[tID] = _a[tID] + _b[tID];
}

void settingTimmer(DS_timer&timer, std::string*names){
    for(size_t i = 0; i < names->size(); ++i)
        timer.setTimerName(i, names[i]);  
      
	timer.initTimers();
}

void hostVectorSum(int*operand_a, int*operand_b, int*res){
    for (size_t i = 0; i < DATA_SIZE; ++i)
		res[i] = operand_a[i] + operand_b[i];
}

void checkResult(int*host_res, int*device_res){
    bool result = true;
	for (size_t i = 0; i < DATA_SIZE; ++i)
		if (host_res[i] != device_res[i]) {
			printf("[%zu] The resutls is not matched! (%d, %d)\n", i, host_res[i], device_res[i]);
			result = false;
		}

	if (result)
		printf("GPU works well!\n");
}

int main(){
    std::string *names = new std::string[5] {
        "total", 
        "host vectorSum", 
        "device vectorSum", 
        "host -> device", 
        "device -> host"
        };
    DS_timer timer(5);
	settingTimmer(timer, names);
    
    dim3 dimGrid(DATA_SIZE / 256, 1, 1);
    dim3 dimBlock(256, 1, 1);

    long memSize = sizeof(int) * DATA_SIZE;

    int *a = new int[DATA_SIZE];
    int *b = new int[DATA_SIZE];
    int *res = new int[DATA_SIZE]; 
    int *h_res = new int[DATA_SIZE];
    
    int *d_a, *d_b, *d_res;
    cudaMalloc(&d_a, memSize);
	cudaMalloc(&d_b, memSize);
	cudaMalloc(&d_res, memSize);

    for (size_t i = 0; i < DATA_SIZE; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

    timer.onTimer(1);
    hostVectorSum(a, b, h_res);
    timer.offTimer(1);

    timer.onTimer(0);

    timer.onTimer(3);
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
    timer.offTimer(3);

    timer.onTimer(2);
	vectorSum<<<dimGrid, dimBlock>>>(d_a, d_b, d_res, DATA_SIZE);
	cudaDeviceSynchronize();
	timer.offTimer(2);

    timer.onTimer(4);
    cudaMemcpy(res, d_res, memSize, cudaMemcpyDeviceToHost);
    timer.offTimer(4);
    
    timer.offTimer(0);
    timer.printTimer();

    checkResult(h_res, res);

    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_res);
	
	delete[] a; 
    delete[] b; 
    delete[] res;
    delete[] h_res;

    return 0;
}