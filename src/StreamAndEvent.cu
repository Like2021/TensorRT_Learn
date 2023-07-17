#include <cuda_runtime.h>
#include <iostream>

#define STREAMNUMS 10

__global__ void vecAddKernel(float* A_d, float* B_d, float* C_d, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) C_d[i] = A_d[i] + B_d[i];
}

int main(int argc, char* argv[])
{
    // 初始化输入
    int n = atoi(argv[1]);
    std::cout << "计算向量的级数: " << n << std::endl;

    // 申请主机内存
    size_t size = n * sizeof(float);
    float* a = (float*)malloc(size);
    float* b = (float*)malloc(size);
    float* c = (float*)malloc(size);

    for (int i = 0; i < n; i++)
    {
        float af = rand() / double(RAND_MAX);
        float bf = rand() / double(RAND_MAX);
        a[i] = af;
        b[i] = bf;
    }

    // 申请设备指针
    float* a_d = nullptr;
    float* b_d = nullptr;
    float* c_d = nullptr;

    // 线程块所含线程的数量和网格所含线程筷的数量
    int threadPerBlock = 256;
    int blockPerGrid = (n + threadPerBlock - 1) / threadPerBlock;

    // 创建流
    cudaStream_t streams[STREAMNUMS];
    for (int i = 0; i < STREAMNUMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // 定义偏移量
    int offset = 0;
    const int dataBlock = n / STREAMNUMS;
    // 循环启动stream
    for (int i = 0; i < STREAMNUMS; i++)
    {
        std::cout << "接下来开始进行流操作" << i << std::endl;
        offset = dataBlock * i;
        cudaMemcpyAsync(a_d + offset, a + offset, dataBlock, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(b_d + offset, b + offset, dataBlock, cudaMemcpyHostToDevice, streams[i]);
        vecAddKernel<<<threadPerBlock, blockPerGrid>>>(a_d, b_d, c_d, dataBlock);
        cudaMemcpyAsync(c + offset, c_d + offset, dataBlock, cudaMemcpyHostToDevice, streams[i]);
    }

    // cudaDeviceSynchronize();
    // 同步流
    for (int i = 0; i < STREAMNUMS; i++)
    {
        cudaStreamSynchronize(streams[i]);
    }
    for (int i = 0; i < STREAMNUMS; i++)
    {
        cudaStreamDestroy(streams[i]);
    }

    cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

    std::cout << "正常运行" << std::endl;

	free(a);
	free(b);
	free(c);

    return 0;
}