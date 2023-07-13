#include <cudnn.h>
#include <iostream>


void cudnn_conv()
{

}

int main(int agrc, char* argv[])
{
    // 1.初始化cudnn
    cudnnStatus_t cudnn_re;
    cudnnHandle_t h_cudnn;
    cudnn_re = cudnnCreate(&h_cudnn);
    if (CUDNN_STATUS_SUCCESS == cudnn_re) {std::cout << "创建成功" << std::endl;}

    
}
