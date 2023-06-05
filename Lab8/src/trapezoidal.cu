#include "trapezoidal/trapezoidal_gpu.h"
#include "trapezoidal/trapezoidal_cpu.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__host__ __device__  double func(double x)
{
    return (x * x);
}

__host__ __device__ double Trapezoidal(long i, double h)
{
    double a = START + i * h;
    double b = a + h;
    return (func(a) + func(b)) * h / 2;
}

void serialTrapezoidal(double &serial_res, double h, DS_timer&timer, int mode)
{
    timer.onTimer(mode);

    for (long i = 0; i < SECTION; ++i)
        serial_res += Trapezoidal(i, h);

    timer.offTimer(mode);
}

void ompTrapezoidal(double &omp_res, double h, DS_timer&timer, int mode)
{
    timer.onTimer(mode);

    #pragma omp parallel num_threads(THREAD_NUM) reduction(+ : omp_res)
    {
        #pragma omp for
        for (long i = 0; i < SECTION; ++i)
            omp_res += Trapezoidal(i, h);
    }

    timer.offTimer(mode);
}

__global__ void trapezoidal(double h, double*d_res)
{
    long idx = long(blockIdx.x) * long(blockDim.x) + threadIdx.x;

	if(idx < SECTION){
        double a = START + idx * h;
        double b = a + h;
        double sum = (a*a + b*b) * h / 2;
	    atomicAdd(d_res, sum);
    }
}

__global__ void trapezoidalOptimizing(double h, double*d_res)
{
    long tid = long(blockIdx.x) * long(blockDim.x) + threadIdx.x;

	__shared__ double localVal[BLOCK_SIZE];
	localVal[threadIdx.x] = 0;

	if (tid >= SECTION)
		return;

	localVal[threadIdx.x] = Trapezoidal(tid, h);

	__syncthreads();

	// reduction here
	long offset = BLOCK_SIZE / 2;

	while (offset > 0) {
		if (threadIdx.x < offset) {
			localVal[threadIdx.x] += localVal[threadIdx.x + offset];
		}
		offset /= 2;

		__syncthreads();
	}

	if (threadIdx.x == 0) {
		atomicAdd(d_res, localVal[0]);
	}
}

void kernelCall(double&cuda_res, double h, DS_timer&timer, int mode){
    dim3 dimGrid(ceil(SECTION / (float)BLOCK_SIZE));
    double*d_res;

    timer.onTimer(mode+2);
    cudaMalloc(&d_res, sizeof(double));
    cudaMemset(d_res, 0, sizeof(double));
    timer.offTimer(mode+2);

    timer.onTimer(mode);

    switch(mode)
    {
    case CUDA_BASIC:
	    trapezoidal<<<dimGrid, BLOCK_SIZE>>> (h, d_res);
	    break;
    case CUDA_OPTIMIZING:
        trapezoidalOptimizing<<<dimGrid, BLOCK_SIZE>>> (h, d_res);
        break;
    default:
        break;
    }

    cudaDeviceSynchronize();
    
    timer.offTimer(mode);

    timer.onTimer(mode+2);
    cudaMemcpy(&cuda_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    timer.offTimer(mode+2);
}