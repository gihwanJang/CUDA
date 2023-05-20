#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 16

#define MAT_A_ROW_SIZE 1024 // m 
#define MAT_A_COL_SIZE 512 // k
#define MAT_A_SIZE (MAT_A_ROW_SIZE * MAT_A_COL_SIZE)

#define MAT_B_ROW_SIZE (MAT_A_COL_SIZE)
#define MAT_B_COL_SIZE 1024 // n
#define MAT_B_SIZE (MAT_B_ROW_SIZE * MAT_B_COL_SIZE)

#define MAT_RES_SIZE (MAT_A_ROW_SIZE * MAT_B_COL_SIZE)

/*
__global__ void cudaMatrixMultiplication(
    float *matA, float *matB, float *matC, long m, long n, long k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < m && col < n){
        float _c = 0;

	    for(int i = 0 ; i < k; ++i)
		    _c += matA[row * k + i] * matB[i * n + col];

        matC[row * n + col] = _c;
    }
}
*/

__global__ void cudaMatrixMultiplication(float* matA, float* matB, float* matC, long m, long n, long k){
    __shared__ float sharedMatA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedMatB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    float result = 0;

    for (int t = 0; t < ceil((float)k / BLOCK_SIZE); ++t) {
        int row_condition = ((row < m) && (t * BLOCK_SIZE + threadIdx.y) < k);
        int col_condition = ((col < n) && (t * BLOCK_SIZE + threadIdx.x) < k);

        sharedMatA[threadIdx.x][threadIdx.y] = matA[row * k + t * BLOCK_SIZE + threadIdx.y] * row_condition;
        sharedMatB[threadIdx.x][threadIdx.y] = matB[(t * BLOCK_SIZE + threadIdx.x) * n + col] * col_condition;

        __syncthreads();

        for (int i = 0; i < BLOCK_SIZE; ++i)
            result += sharedMatA[threadIdx.x][i] * sharedMatB[i][threadIdx.y];

        __syncthreads();
    }

    if ((row < m) && (col < n))
        matC[row * n + col] = result;
}

void ompMatrixMultiplication(float*a, float*b, float*res){
    #pragma omp parallel for num_threads(8)
    for(int m = 0; m < MAT_A_ROW_SIZE; ++m)
        for(int n = 0; n < MAT_B_COL_SIZE; ++n)
            for(int k = 0; k < MAT_A_COL_SIZE; ++k)
                res[m * MAT_B_COL_SIZE + n] += a[m * MAT_A_COL_SIZE + k] * b[k * MAT_B_COL_SIZE + n];
}

void serialMatrixMultiplication(float*a, float*b, float*res){
    for(int m = 0; m < MAT_A_ROW_SIZE; ++m)
        for(int n = 0; n < MAT_B_COL_SIZE; ++n)
            for(int k = 0; k < MAT_A_COL_SIZE; ++k)
                res[m * MAT_B_COL_SIZE + n] += a[m * MAT_A_COL_SIZE + k] * b[k * MAT_B_COL_SIZE + n];
}

void checkResult(float*serial_res, float*omp_res, float*cuda_res){
    bool omp_result = true;
    bool cuda_result = true;

    for(int i = 0; i < MAT_RES_SIZE; ++i)
        if(serial_res[i] != omp_res[i]){
            printf("serial : %f , opm :%f\n", serial_res[i], omp_res[i]);
            omp_result = false;
            break;
        }

    for(int i = 0; i < MAT_RES_SIZE; ++i)
        if(serial_res[i] != cuda_res[i]){
            printf("serial : %f , cuda :%f\n", serial_res[i], cuda_res[i]);
            cuda_result = false;
            break;
        }

    if(omp_result && cuda_result) 
        printf("CUDA and OpenMP works well!\n");
    else if(omp_result)
        printf("OpenMP works well but CUDA fail to make correct result(s)..\n");
    else if(cuda_result)
        printf("CUDA works well but OpenMP fail to make correct result(s)..\n");
    else
        printf("CUDA & OpenMP fail to make correct result(s)..\n");
}

void dataGenerate(float*mat, int matSize){
    for(int i = 0; i < matSize; ++i)
        mat[i] = double(rand() % 100000) / 10000;
}

int main(){
    DS_timer timer(6);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(ceil((float)MAT_A_ROW_SIZE / BLOCK_SIZE), ceil((float)MAT_B_COL_SIZE / BLOCK_SIZE));
    
    int aMemSize = sizeof(float) * MAT_A_SIZE;
    int bMemSize = sizeof(float) * MAT_B_SIZE;
    int resMemSize = sizeof(float) * MAT_RES_SIZE;

    float *a, *b, *s_res, *omp_res, *cuda_res;
    float *d_a, *d_b, *d_res;

    // timer setting
    timer.setTimerName(0, (char*)"sirial Compute");
    timer.setTimerName(1, (char*)"openmp Compute");
    timer.setTimerName(2, (char*)"cuda Compute total");
    timer.setTimerName(3, (char*)"cuda Compute");
    timer.setTimerName(4, (char*)"host -> device");
    timer.setTimerName(5, (char*)"device -> host");

    // host memory malloc
    a = new float[MAT_A_SIZE];
    memset(a, 0, aMemSize);
    b = new float[MAT_B_SIZE];
    memset(b, 0, bMemSize);
    s_res = new float[MAT_RES_SIZE];
    memset(s_res, 0, resMemSize);
    omp_res = new float[MAT_RES_SIZE];
    memset(omp_res, 0, resMemSize);
    cuda_res = new float[MAT_RES_SIZE];
    memset(cuda_res, 0, resMemSize);

    // device memory malloc
    cudaMalloc(&d_a, aMemSize);
    cudaMemset(d_a, 0, aMemSize);
    cudaMalloc(&d_b, bMemSize);
    cudaMemset(d_b, 0, bMemSize);
    cudaMalloc(&d_res, resMemSize);
    cudaMemset(d_res, 0, resMemSize);

    // data generate
    dataGenerate(a, MAT_A_SIZE);
    dataGenerate(b, MAT_B_SIZE);

    // show matrix size
    printf("Mat A data size : %d x %d\n", MAT_A_ROW_SIZE, MAT_A_COL_SIZE);
    printf("Mat A memory size : %d\n", aMemSize);
    printf("Mat B data size : %d x %d\n",MAT_B_ROW_SIZE, MAT_B_COL_SIZE);
    printf("Mat B memory size : %d\n", bMemSize);

    // serial matrix multiplication
    timer.onTimer(0);
    serialMatrixMultiplication(a, b, s_res);
    timer.offTimer(0);

    // OpenMP matrix multiplication
    timer.onTimer(1);
    ompMatrixMultiplication(a, b, omp_res);
    timer.offTimer(1); 

    timer.onTimer(2);
    
    // memory copy host to device
    timer.onTimer(4);
    cudaMemcpy(d_a, a, aMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bMemSize, cudaMemcpyHostToDevice);
    timer.offTimer(4);

    // cuda matrix multiplication
    timer.onTimer(3);
    cudaMatrixMultiplication<<<gridDim, blockDim>>>(d_a, d_b, d_res, MAT_A_ROW_SIZE, MAT_B_COL_SIZE, MAT_A_COL_SIZE);
    cudaDeviceSynchronize();
    timer.offTimer(3);

    // memory copy device to host
    timer.onTimer(5);
    cudaMemcpy(cuda_res, d_res, resMemSize, cudaMemcpyDeviceToHost);
    timer.offTimer(5);

    timer.offTimer(2);
    timer.printTimer();

    // check correct
    checkResult(s_res, omp_res, cuda_res);

    // host memory free
    delete a;
    delete b;
    delete s_res;
    delete omp_res;
    delete cuda_res;

    // device memory free
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);
    return 0;
}