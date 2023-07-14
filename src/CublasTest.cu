#include "cublas_v2.h"
#include <cuda_runtime.h>

#define M 512
#define N 512
#define K 512

#define RUNTIME_CUDA_ERROR(errorInfo) CHECK_CUDA(errorInfo, cudaSuccess)

#define CUDA_FREE(d_ptr){\
    if (d_ptr != nullptr) RUNTIME_CUDA_ERROR(cudaFree(d_ptr)); d_ptr = nullptr;\
}\

int main(int argc, char* argv[])
{
    // 组织数据
    int A = M * N;


    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasDestroy(handle);

    return 0;
}