#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

__global__
void vecAddKernel(float* A_d, float* B_d, float* C_d, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C_d[i] = A_d[i] + B_d[i];
}

int main(int agrc, char* agrv[])
{
    // 初始化输入
    int n = atoi(agrv[1]);
    std::cout << n << std::endl;

    // 申请主机内存
    size_t size = n * sizeof(float);
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);

    // 组织随机向量数据
    for (int i = 0; i < n; i++)
    {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    // 创建设备指针
    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;

    // 申请GPU内存
    cudaMalloc((void**)& da, size);
    cudaMalloc((void**)& db, size);
    cudaMalloc((void**)& dc, size);

    // 将数据从主机移动到设备上
    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, size, cudaMemcpyHostToDevice);

    struct timeval t1, t2;

    // 线程块所含线程的数量和网格所含线程筷的数量
    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1) / threadPerBlock;

    // 定义时间
    gettimeofday(&t1, nullptr);

    // 核函数计算
    vecAddKernel <<< blockPerGrid, threadPerBlock >>> (da, db, dc, n);

    gettimeofday(&t2, nullptr);

    // 将计算结果从设备转移到主机
    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    // 计算时间
    double timeUse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec) / 1000000.0;
    std::cout << timeUse << std::endl;

    // 释放指针
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);

    return 0;
}