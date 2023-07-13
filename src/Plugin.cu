#include <iostream>
#include <cuda.h>
#include <NvInferPlugin.h>

int* iptr = new int(3);

class MyPluginFunc
{
    int initialize();

    size_t getWorkspaceSize();

    int enqueue(int batchSize, const void* const* inputs);
};
