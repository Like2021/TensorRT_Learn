#include <cudnn.h>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(){\
    if(cudaPeekAtLastError() != cudaSuccess){\
        printf("CUDA error in line %d of file %s:%s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));\
        exit(-1);\
    }\
}\

#define CUDA_CALL(cudaFunc){\
    if(cudaFunc != cudaSuccess){\
        printf("CUDA error in line %d of file %s:%s\n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()));\
        exit(-1);\
    }\
}\

int main(int agrc, char* argv[])
{
    // 初始化cudnn
    cudnnStatus_t status_cudnn;
    cudnnHandle_t haddle_cudnn;
    cudnnCreate(&haddle_cudnn);
    // 定义输入输出张量
    cudnnTensorDescriptor_t ts_in, ts_out;
    // 创建输入张量
    status_cudnn = cudnnCreateTensorDescriptor(&ts_in);
    if (CUDNN_STATUS_SUCCESS == status_cudnn) {std::cout << "创建成功" << std::endl;}
    // 设置输入张量数据
    status_cudnn = cudnnSetTensor4dDescriptor(
        ts_in,
        CUDNN_TENSOR_NHWC,
        CUDNN_DATA_FLOAT,
        1,
        3,
        1080,
        1920
    );
    // 设置输出张量
    cudnnCreateTensorDescriptor(&ts_out);
    if (CUDNN_STATUS_SUCCESS == status_cudnn) std::cout << "创建输入张量成功" << std::endl;
    // 设置输出张量数据
    status_cudnn = cudnnSetTensor4dDescriptor(
        ts_out,
        CUDNN_TENSOR_NHWC, 
        CUDNN_DATA_FLOAT,
        1,
        3,
        1080,
        1920
    );
    // 定义和创建卷积核
    cudnnFilterDescriptor_t kernel;
    cudnnCreateFilterDescriptor(&kernel);
    // 设置卷积核
    status_cudnn = cudnnSetFilter4dDescriptor(
        kernel,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NHWC, 3, 3, 3, 3
    );
    // 定义和创建卷积
    cudnnConvolutionDescriptor_t conv;
    status_cudnn = cudnnCreateConvolutionDescriptor(&conv);
    status_cudnn = cudnnSetConvolution2dDescriptor(
        conv, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT
    );
    // 
}
