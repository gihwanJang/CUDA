#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#include "timer/DS_definitions.h"
#include "timer/DS_timer.h"

#define ROW_SIZE (8192)
#define COL_SIZE (8192)
#define MAT_SIZE (ROW_SIZE * COL_SIZE)

__global__ void MatSum2GirdWith2Block(int *d_a, int *d_b, int *d_res, int size){
    long block_loc = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y;
    long thread_loc = threadIdx.y * blockDim.x + threadIdx.x;
    long idx = block_loc + thread_loc;

    if(idx < size)
		d_res[idx] = d_a[idx] + d_b[idx];
}

__global__ void MatSum1GirdWith1Block(int *d_a, int *d_b, int *d_res, int size){
    long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < size)
		d_res[idx] = d_a[idx] + d_b[idx];
}

__global__ void MatSum2GirdWith1Block(int *d_a, int *d_b, int *d_res, int size){
    long block_loc = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x;
    long idx = block_loc + threadIdx.x;

    if (idx < size)
		d_res[idx] = d_a[idx] + d_b[idx];
}

void hostMatSUm(int *MatA, int *MatB, int*res){
    for(int r = 0; r < ROW_SIZE; ++r)
        for(int c = 0; c < COL_SIZE; ++c){
            int idx = r * COL_SIZE + c;
            res[idx] = MatA[idx] + MatB[idx];
        }
}

void settingTimmer(DS_timer&timer, std::string*names){
    for(size_t i = 0; i < names->size(); ++i)
        timer.setTimerName(i, names[i]);  
      
	timer.initTimers();
}

void checkResult(int*host_res, int*device_res){
    bool result = true;

	for (size_t i = 0; i < MAT_SIZE; ++i) {
		if (host_res[i] != device_res[i]) {
			result = false;
            break;
		}
    }

	if (result)
		printf("GPU works well!\n");
    else
        printf("GPU fail to make correct result(s)..\n");
}

int main(){
    std::string *names = new std::string[5] {
        "total", 
        "host Compute", 
        "device Compute", 
        "host -> device", 
        "device -> host"
        };
    DS_timer timer(5);
	settingTimmer(timer, names);

    //dim3 blockDim(16, 8);
	//dim3 gridDim(ROW_SIZE / blockDim.x, COL_SIZE / blockDim.y);
    //dim3 blockDim(128);
	//dim3 gridDim(ROW_SIZE * COL_SIZE / blockDim.x);
    dim3 blockDim(256);
	dim3 gridDim(COL_SIZE / blockDim.x, ROW_SIZE);
    
    long mem_size = sizeof(int) * MAT_SIZE;

    int *a, *b, *res, *h_res;
    int *d_a, *d_b, *d_res;

    a = new int[MAT_SIZE]; 
    memset(a, 0, mem_size);
    b = new int[MAT_SIZE]; 
    memset(b, 0, mem_size);
    res = new int[MAT_SIZE]; 
    memset(res, 0, mem_size);
    h_res = new int[MAT_SIZE]; 
    memset(h_res, 0, mem_size);

    cudaMalloc(&d_a, mem_size); 
    cudaMemset(d_a, 0, mem_size);
	cudaMalloc(&d_b, mem_size); 
    cudaMemset(d_b, 0, mem_size);
	cudaMalloc(&d_res, mem_size); 
    cudaMemset(d_res, 0, mem_size);

    for(int i = 0; i < MAT_SIZE; i++) {
	    a[i] = rand() % 100;
	    b[i] = rand() % 100;
    }

    timer.onTimer(1);
    hostMatSUm(a, b, h_res);
    timer.offTimer(1);

    timer.onTimer(0);

    timer.onTimer(3);
    cudaMemcpy(d_a, a, mem_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, mem_size, cudaMemcpyHostToDevice);
    timer.offTimer(3);

    timer.onTimer(2);
    //MatSum2GirdWith2Block<<<gridDim, blockDim>>>(d_a, d_b, d_res, MAT_SIZE); 
    //MatSum1GirdWith1Block<<<gridDim, blockDim>>>(d_a, d_b, d_res, MAT_SIZE); 
    MatSum2GirdWith1Block<<<gridDim, blockDim>>>(d_a, d_b, d_res, MAT_SIZE); 
    cudaDeviceSynchronize();
    timer.offTimer(2);

    timer.onTimer(4);
    cudaMemcpy(res, d_res, mem_size, cudaMemcpyDeviceToHost);
    timer.offTimer(4);

    timer.offTimer(0);
    timer.printTimer();

    checkResult(h_res, res);
    return 0;
}