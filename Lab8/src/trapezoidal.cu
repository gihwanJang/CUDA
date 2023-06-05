#include "trapezoidal/trapezoidalGPU.h"
#include "trapezoidal/trapezoidalCPU.h"

#include "parameter/parameter.h"

float func(float x)
{
    return (x * x);
}

float Trapezoidal(float i, float h)
{
    float a = START + i * h;
    float b = a + h;
    return (func(a) + func(b)) * h / 2;
}

void serialTrapezoidal(float &serial_res, float h)
{
    for (int i = 0; i < SECTION; ++i)
        serial_res += Trapezoidal(i, h);
}

void ompTrapezoidal(float &omp_res, float h)
{
    #pragma omp parallel num_threads(THREAD_NUM) reduction(+ : omp_res)
    {
        #pragma omp for
        for (int i = 0; i < SECTION; ++i)
            omp_res += Trapezoidal(i, h);
    }
}

__global__ void trapezoidal()
{
}

__global__ void trapezoidalOptimizing()
{
}

void kernelCall(int mode)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(ceil((float)SECTION / BLOCK_SIZE));

    switch (mode)
    {
    case CUDA_BASIC:
        trapezoidal<<<grid, block>>>();
        break;
    case CUDA_OPIMIZING:
        trapezoidalOptimizing<<<grid, block>>>();
        break;
    default:
        break;
    }
}