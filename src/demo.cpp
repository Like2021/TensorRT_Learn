#include <iostream>
#include <cstdlib>
#include <sys/time.h>

void vecAdd(float *A, float *B, float *C, int n)
{
    for (int i = 0; i < n; i++)
    {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[])
{
    // 
    int n = atoi(argv[1]);
    std::cout << n << std::endl;

    size_t size = n * sizeof(float);
}