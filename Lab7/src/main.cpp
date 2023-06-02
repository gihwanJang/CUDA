#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "timer/DS_timer.h"
#include "timer/DS_definitions.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "kernel/kernelCall.h"

#define BLOCK_SIZE 16

#define MAT_A_ROW_SIZE 2048 // m
#define MAT_A_COL_SIZE 2048  // k
#define MAT_A_SIZE (MAT_A_ROW_SIZE * MAT_A_COL_SIZE)

#define MAT_B_ROW_SIZE (MAT_A_COL_SIZE)
#define MAT_B_COL_SIZE 2048 // n
#define MAT_B_SIZE (MAT_B_ROW_SIZE * MAT_B_COL_SIZE)

#define MAT_RES_SIZE (MAT_A_ROW_SIZE * MAT_B_COL_SIZE)

#define CUDA_NOMAL 0
#define CUDA_USING_SHARED_MEMORY 1
#define CUDA_USING_SHARED_MEMORY_OPTIMIZED 2

void ompMatrixMultiplication(float *a, float *b, float *res)
{
#pragma omp parallel for num_threads(8)
    for (int m = 0; m < MAT_A_ROW_SIZE; ++m)
        for (int n = 0; n < MAT_B_COL_SIZE; ++n)
            for (int k = 0; k < MAT_A_COL_SIZE; ++k)
                res[m * MAT_B_COL_SIZE + n] += a[m * MAT_A_COL_SIZE + k] * b[k * MAT_B_COL_SIZE + n];
}

void serialMatrixMultiplication(float *a, float *b, float *res)
{
    for (int m = 0; m < MAT_A_ROW_SIZE; ++m)
        for (int n = 0; n < MAT_B_COL_SIZE; ++n)
            for (int k = 0; k < MAT_A_COL_SIZE; ++k)
                res[m * MAT_B_COL_SIZE + n] += a[m * MAT_A_COL_SIZE + k] * b[k * MAT_B_COL_SIZE + n];
}

void checkResult(float *serial_res, float *res, const char *s)
{
    bool result = true;

    for (int i = 0; i < MAT_RES_SIZE; ++i)
        if (serial_res[i] != res[i])
        {
            printf("serial : %f , %s :%f\n", serial_res[i], s, res[i]);
            result = false;
            break;
        }

    if (result)
        printf("%s works well!\n", s);
    else
        printf("%s fail to make correct result(s)..\n", s);
}

void dataGenerate(float *mat, int matSize)
{
    for (int i = 0; i < matSize; ++i)
        mat[i] = double(rand() % 100000) / 10000;
}

int main()
{
    DS_timer timer(7);

    int aMemSize = sizeof(float) * MAT_A_SIZE;
    int bMemSize = sizeof(float) * MAT_B_SIZE;
    int resMemSize = sizeof(float) * MAT_RES_SIZE;

    float *a, *b, *s_res, *omp_res, *cuda_res;
    float *d_a, *d_b, *d_res;

    // timer setting
    timer.setTimerName(0, (char *)"sirial Compute");
    timer.setTimerName(1, (char *)"openmp Compute");
    timer.setTimerName(2, (char *)"cuda Nomal Compute");
    timer.setTimerName(3, (char *)"cuda using shared memory Compute");
    timer.setTimerName(4, (char *)"cuda using shared memory and optimized Compute");
    timer.setTimerName(5, (char *)"host -> device");
    timer.setTimerName(6, (char *)"device -> host");

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
    printf("Mat B data size : %d x %d\n", MAT_B_ROW_SIZE, MAT_B_COL_SIZE);
    printf("Mat B memory size : %d\n\n", bMemSize);

    // serial matrix multiplication
    timer.onTimer(0);
    serialMatrixMultiplication(a, b, s_res);
    timer.offTimer(0);

    // OpenMP matrix multiplication
    timer.onTimer(1);
    ompMatrixMultiplication(a, b, omp_res);
    timer.offTimer(1);

    // check correct
    checkResult(s_res, omp_res, "openMP");

    // memory copy host to device
    timer.onTimer(5);
    cudaMemcpy(d_a, a, aMemSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bMemSize, cudaMemcpyHostToDevice);
    timer.offTimer(5);


    // cuda matrix multiplication nomal part
    {
        // cuda matrix multiplication
        timer.onTimer(2);
        kernelCall(d_a, d_b, d_res, MAT_A_ROW_SIZE, MAT_B_COL_SIZE, MAT_A_COL_SIZE, BLOCK_SIZE, CUDA_NOMAL);
        cudaDeviceSynchronize();
        timer.offTimer(2);

        // memory copy device to host
        timer.onTimer(6);
        cudaMemcpy(cuda_res, d_res, resMemSize, cudaMemcpyDeviceToHost);
        timer.offTimer(6);

        // check correct
        checkResult(s_res, cuda_res, "CUDA nomal");
        memset(cuda_res, 0, resMemSize);
    }


    // cuda matrix multiplication using shared memory part
    {
        // cuda matrix multiplication using shared memory
        timer.onTimer(3);
        kernelCall(d_a, d_b, d_res, MAT_A_ROW_SIZE, MAT_B_COL_SIZE, MAT_A_COL_SIZE, BLOCK_SIZE, CUDA_USING_SHARED_MEMORY);
        cudaDeviceSynchronize();
        timer.offTimer(3);

        // memory copy device to host
        cudaMemcpy(cuda_res, d_res, resMemSize, cudaMemcpyDeviceToHost);

        // check correct
        checkResult(s_res, cuda_res, "CUDA using shared memory");
        memset(cuda_res, 0, resMemSize);
    }


    // cuda matrix multiplication using shared memory optimized part
    {
        // cuda matrix multiplication using shared memory optimized
        timer.onTimer(4);
        kernelCall(d_a, d_b, d_res, MAT_A_ROW_SIZE, MAT_B_COL_SIZE, MAT_A_COL_SIZE, BLOCK_SIZE, CUDA_USING_SHARED_MEMORY_OPTIMIZED);
        cudaDeviceSynchronize();
        timer.offTimer(4);

        // memory copy device to host
        cudaMemcpy(cuda_res, d_res, resMemSize, cudaMemcpyDeviceToHost);

        // check correct
        checkResult(s_res, cuda_res, "CUDA using shared memory and optimized");
        memset(cuda_res, 0, resMemSize);
    }


    timer.printTimer();

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