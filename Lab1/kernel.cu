
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>

#define MAX_THREADS_X_IN_BLOCK_SIZE 1024
#define BLOCK_X_AMOUNT 1024
#define MAX_CHUNK_ARRAY_SIZE (MAX_THREADS_X_IN_BLOCK_SIZE * BLOCK_X_AMOUNT)

using namespace std;

cudaError_t arrayCalculations(double* c, const double* a, const double* b, unsigned int size);
unsigned int loadFileToArray(std::ifstream &file, double* a, unsigned int sizeToRead);
int saveArrayToFile(double* c, string fileName, unsigned int sizeToWrite);

__global__ void calculateKernel(double* c, const double* a, const double* b, long long size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < size; i += stride)
        c[i] = (a[i] + b[i]) * 4 + 5;
}

int main()
{
    // 2 ** 20 - max possible array chunk
    const unsigned int arraySize = MAX_CHUNK_ARRAY_SIZE;

    double a[arraySize] = {};
    double b[arraySize] = {};
    double c[arraySize] = {};

    //long long calculationLength = std::pow(2, 33);
    long long calculationLength = (long long)std::pow(2, 21);
    string fileName1 = "..\\test_file1_floats.txt";
    string fileName2 = "..\\test_file2_floats.txt";
    string fileNameResult = "..\\result.txt";

    long long currentFilePossition = 0;
    int numberOfIterations = (int) (calculationLength / MAX_CHUNK_ARRAY_SIZE);

    cudaError_t cudaStatus;
    std::ifstream file1(fileName1);
    std::ifstream file2(fileName2);

    for (size_t i = 0; i < numberOfIterations; i++)
    {
        // setArrays from files
        unsigned int setSize1 = loadFileToArray(file1, a, arraySize);
        unsigned int setSize2 = loadFileToArray(file2, b, arraySize);

        if (setSize1 == 0 || setSize2 == 0)
        {
            printf("Unable to open file for reading.");
            return 1;
        }
        if (setSize1 != setSize2)
        {
            printf("Lines in files should be of the same length.");
            return 1;
        }

        // Add vectors in parallel.
        cudaStatus = arrayCalculations(c, a, b, setSize1);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "arrayCalculations failed!");
            return 1;
        }

        // save result array to file
        int result = saveArrayToFile(c, fileNameResult, setSize1);

        if (result == -1)
        {
            printf("Unable to open file for writing.");
            return 1;
        }

        currentFilePossition += setSize1;
    }

    file1.close();
    file2.close();

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// recieves files with values from [0, 100)
unsigned int loadFileToArray(std::ifstream &file, double* a, unsigned int sizeToRead) {

    std::string line;

    if (!file.is_open())
    {
        return 0;
    }

    unsigned int loop = 0;
    double currentValue = 0;

    while (!file.eof() && loop < sizeToRead)
    {
        std::getline(file, line);
        currentValue = std::stod(line) / 100;
        a[loop] = currentValue;
        loop++;
    }

    return loop;
}

int saveArrayToFile(double* c, string fileName, unsigned int sizeToWrite)
{
    ofstream file(fileName, std::ios_base::app);
    
    if (!file.is_open())
    {
        return -1;
    }

    for (unsigned int count = 0; count < sizeToWrite; count++) {
        std::ostringstream ss;
        ss << c[count];
        ss << "\n";
        file << ss.str();
    }
    file.close();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t arrayCalculations(double* c, const double* a, const double* b, unsigned int size)
{
    double* dev_a = 0;
    double* dev_b = 0;
    double* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)& dev_c, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& dev_a, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)& dev_b, size * sizeof(double));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    int numBlocks = (size + MAX_THREADS_X_IN_BLOCK_SIZE - 1) / MAX_THREADS_X_IN_BLOCK_SIZE;;
    int numThreads = MAX_THREADS_X_IN_BLOCK_SIZE;
    if (size < MAX_THREADS_X_IN_BLOCK_SIZE)
    {
        numBlocks = 1;
        numThreads = size;
    }
    calculateKernel <<<numBlocks, numThreads>>> (dev_c, dev_a, dev_b, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "arrayCalculations launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching arrayCalculations!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}
