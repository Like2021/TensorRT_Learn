#include "cublas_v2.h"
#include <cuda_runtime.h>
#include <iostream>

#define M 512
#define N 512
#define K 512

// #define RUNTIME_CUDA_ERROR(errorInfo) CHECK_CUDA(errorInfo, cudaSuccess)

// #define CUDA_FREE(d_ptr){\
//     if (d_ptr != nullptr) RUNTIME_CUDA_ERROR(cudaFree(d_ptr)); d_ptr = nullptr;\
// }\

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(rand() % 10 + 1);
	}
}

int main(int argc, char* argv[])
{
    // 组织数据
    // 定义矩阵元素个数
    int numsA = M * K;
    int numsB = K * N;
    int numsC = M * N;

    // define pointer of host and device 
    float *A, *B, *C, *deviceA, *deviceB, *deviceC;
    // 矩阵内存大小
    size_t sizeA = numsA * sizeof(float);
    size_t sizeB = numsB * sizeof(float);
    size_t sizeC = numsC * sizeof(float);
    // 根据size在host端开辟内存
    A = (float*)malloc(sizeA);
    B = (float*)malloc(sizeB);
    C = (float*)malloc(sizeC);
    // 初始化矩阵
    initial(A, numsA);
    initial(B, numsB);
    // 根据size在设备端开辟内存
    cudaMalloc((void**) &deviceA, sizeA);
    cudaMalloc((void**) &deviceB, sizeB);
    cudaMalloc((void**) &deviceC, sizeC);
    // 将AB转移到设备端
    cudaMemcpy(deviceA, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, B, sizeB, cudaMemcpyHostToDevice);

    // 创建句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 创建Event，计算耗时
    float elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // 标记流
    cudaEventRecord(start, 0);

    // alpha beta
    float a = 1, b = 0;
    cublasSgemm(
        handle,  // 句柄
        CUBLAS_OP_T,  // A是否转置
        CUBLAS_OP_T,  // B是否转置
        M,  // A的行
        N,  // B的列
        K,  // A的列和B的行
        &a,  // alpha
        deviceA,  // 设备端A的指针
        K,  // leading dimension，转置，故输入A的列数
        deviceB,  // 设备端B的指针
        N,  // leading dimension，转置，故输入B的列数
        &b,  // beta
        deviceC,  // 设备端C的指针
        M  // C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
    );
    // 将计算结果拷贝回host
    cudaMemcpy(C, deviceC, sizeC, cudaMemcpyDeviceToHost);
    // 等待同步，然后标记结束，并计算时间，销毁Event
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("GPU运行时间为: %fs\n", elapsedTime / 1000);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    free(A);
    free(B);
    free(C);

    // cudaDeviceReset();
    cublasDestroy(handle);

    return 0;
}