#include <iostream>
#include <cuda_runtime.h>

#define M 512
#define K 512
#define N 512

#define BLOCK_SIZE 32 
#define width 512

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

__global__ void multiplicateMatrixShareMemory(float* Md, float* Nd, float* Pd)
{
    // 开辟共享内存
    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 标识计算结果Pd对应的行列，这样就确认好了矩阵单个元素对应的线程
    int Row = by * BLOCK_SIZE + ty;
    int Col = bx * BLOCK_SIZE + tx;

    float Pvalue = 0;

    // 注: 同一BLOCK_SIZE内的块相加后的结果才是对应的Pd单元素的结果
    // 这里利用for循环，计算同一BLOCK_SIZE内的块
    // width / BLOCK_SIZE对应块的数目
    for (int m = 0; m < width / BLOCK_SIZE; m++)
    {
        // 这里对应
        // Mds[ty][tx] = Md[Row][tx]
        // Nds[ty][tx] = Nd[ty][Col]
        // 但全局内存中二维数组是以一维数组的形式保存的，共享内存则不一样，所以这里需要对Md和Nd进行索引转换
        // 实现从两个矩阵中各取一个元素存入共享内存
        Mds[ty][tx] = Md[Row * width + (m * BLOCK_SIZE + tx)];
        Nds[ty][tx] = Nd[(m * BLOCK_SIZE + ty) * width + Col];
        // 等待所有线程都将对应元素存入共享内存中
        __syncthreads();

        // 累加块相乘后的子集
        for (int k = 0; k < BLOCK_SIZE; k++)
        {
            Pvalue += Mds[ty][k] + Nds[k][tx];
        }
        // 同步结果，保证完全计算完当前块后，再进下一个for循环，计算下一个块
    }
    // 最后把结果写入全局内存Pd中
    Pd[Row * width + Col] = Pvalue;
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
    // int dimx = 2;
    // int dimy = 2;
    // dim3 block(dimx, dimy);
    // dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    // multiplicateMatrixOnDevice<<<grid, block>>> (d_A, d_B, d_C, M, K, N);
    dim3 grid(width / BLOCK_SIZE, width / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    multiplicateMatrixShareMemory<<<grid, block>>> (d_A, d_B, d_C);
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