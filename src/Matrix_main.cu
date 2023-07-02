#include <iostream>
#include <cuda_runtime.h>

__global__
void matrixMul()
{
    extern __shared__ int s[];
    __syncthreads();
}