#include "kernel/kernelCall.h"

#define CUDA_NOMAL 0
#define CUDA_USING_SHARED_MEMORY 1
#define CUDA_USING_SHARED_MEMORY_OPTIMIZED 2

__global__ void cudaMatrixMultiplication(float *matA, float *matB, float *matC, int m, int n, int k)
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

__global__ void cudaMatrixMultiplicationUsingSharedMemory(float* matA, float* matB, float* matC, int m, int n, int k, int block_size){
    extern __shared__ float sharedMat[];
    float* sharedMatA = &sharedMat[0];
    float* sharedMatB = &sharedMat[block_size * block_size];

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    float result = 0;

    for (int t = 0; t < ceil((float)k / block_size); ++t) {
        bool row_condition = ((row < m) && (t * block_size + threadIdx.y) < k);
        bool col_condition = ((col < n) && (t * block_size + threadIdx.x) < k);
        
        sharedMatA[threadIdx.x * block_size + threadIdx.y] = row_condition ? matA[row * k + t * block_size + threadIdx.y] : 0;
        sharedMatB[threadIdx.x * block_size + threadIdx.y] = col_condition ? matB[(t * block_size + threadIdx.x) * n + col] : 0;

        __syncthreads();

        for (int i = 0; i < block_size; ++i)
            result += sharedMatA[threadIdx.x * block_size + i] * sharedMatB[i * block_size + threadIdx.y];

        __syncthreads();
    }

    if ((row < m) && (col < n))
        matC[row * n + col] = result;
}

__global__ void cudaMatrixMultiplicationUsingSharedMemoryOptimized(float* matA, float* matB, float* matC, int m, int n, int k, int block_size){
    extern __shared__ float sharedMat[];

    float* sharedMatA = &sharedMat[0];
    float* sharedMatB = &sharedMat[block_size * block_size];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float result = 0;

    for (int t = 0; t < ceil((float)k / block_size); ++t) {
        // Load tiles into shared memory
        int aRow = row;
        int aCol = t * block_size + threadIdx.x;
        int bRow = t * block_size + threadIdx.y;
        int bCol = col;

        if (aRow < m && aCol < k)
            sharedMatA[threadIdx.y * block_size + threadIdx.x] = matA[aRow * k + aCol];
        else
            sharedMatA[threadIdx.y * block_size + threadIdx.x] = 0;

        if (bRow < k && bCol < n)
            sharedMatB[threadIdx.y * block_size + threadIdx.x] = matB[bRow * n + bCol];
        else
            sharedMatB[threadIdx.y * block_size + threadIdx.x] = 0;

        __syncthreads();

        // Compute partial dot product
        for (int i = 0; i < block_size; ++i)
            result += sharedMatA[threadIdx.y * block_size + i] * sharedMatB[i * block_size + threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        matC[row * n + col] = result;
}

void kernelCall(float* a, float* b, float* c, int m, int n, int k, int block_size, int mode) {
    dim3 block(block_size, block_size);
    dim3 grid(ceil((float)m / block_size), ceil((float)n / block_size));

    switch (mode) {
        case CUDA_NOMAL:
            cudaMatrixMultiplication<<<grid, block>>>(a, b, c, m, n, k);
            break;
        case CUDA_USING_SHARED_MEMORY:
            cudaMatrixMultiplicationUsingSharedMemory<<<grid, block, sizeof(float) * block_size * block_size * 2>>>(a, b, c, m, n, k, block_size);
            break;
        case CUDA_USING_SHARED_MEMORY_OPTIMIZED:
            cudaMatrixMultiplicationUsingSharedMemoryOptimized<<<grid, block, sizeof(float) * block_size * block_size * 2>>>(a, b, c, m, n, k, block_size);
            break;
        default:
            break;
    }
}