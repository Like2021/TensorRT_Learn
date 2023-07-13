#include "cublas_v2.h"

int main(int argc, char* argv[])
{
    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasDestroy(handle);

    return 0;
}