#include <iostream>
#include <cuda_runtime.h>

#define M 512
#define K 512
#define N 512

#define BLOCK_SIZE 32 

void initial(float *array, int size)
{
	for (int i = 0; i < size; i++)
	{
		array[i] = (float)(rand() % 10 + 1);
	}
}

void printMatrix(float *array, int row, int col)
{
	float *p = array;
	for (int y = 0; y < row; y++)
	{
		for (int x = 0; x < col; x++)
		{
			printf("%10lf", p[x]);
		}
		p = p + col;
		printf("\n");
	}
	return;
}

void multiplicateMatrixOnHost(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p)
{
    for (int i = 0; i < M_p; i++)
    {
        for (int j = 0; j < N_p; j++)
        {
            float sum = 0;
            for (int k = 0; k < K_p; k++)
            {
                sum += array_A[i*K_p + k] * array_B[k*N_p + j];
            }
            array_C[i*N_p + j] = sum;
        }
    }
}

__global__ void multiplicateMatrixOnDevice(float* array_A, float* array_B, float* array_C, int M_p, int K_p, int N_p)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;

    if (ix < N_p && iy < M_p)
    {
        float sum = 0;
        for (int k = 0; k < K_p; k++)
        {
            sum += array_A[iy*K_p + k] * array_B[ix*N_p + ix];
        }
        array_C[iy*N_p + ix] = sum;
    }
}

int main(int argc, char* argv[])
{
    // 1.组织数据
	int Axy = M * K;
	int Bxy = K * N;
	int Cxy = M * N;

    float *h_A, *h_B, *deviceRef;
	h_A = (float*)malloc(Axy * sizeof(float));
	h_B = (float*)malloc(Bxy * sizeof(float));
    deviceRef = (float*)malloc(Cxy * sizeof(float));

    initial(h_A, Axy);
    initial(h_B, Bxy);
    printf("1");

    // 2.申请设备端内存
    float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A, Axy * sizeof(float));
	cudaMalloc((void**)&d_B, Bxy * sizeof(float));
	cudaMalloc((void**)&d_C, Cxy * sizeof(float));

    cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

    // 3.组织线程配置，调用核函数
    int dimx = 2;
    int dimy = 2;
    dim3 block(dimx, dimy);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    multiplicateMatrixOnDevice<<<grid,block>>> (d_A, d_B, d_C, M, K, N);
    cudaMemcpy(deviceRef, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    printMatrix(deviceRef, M, N);

    cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(deviceRef);

    return 0;
}