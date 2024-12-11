#include <stdio.h>
#include <stdlib.h>

__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

// Compute vector sum C_h = A_h + B_h
void vecAdd(float *A_h, float *B_h, float *C_h, int n)
{
    // Part 1: allocate A, B and C in device memory
    float *A_d;
    float *B_d;
    float *C_d;

    int size = n * sizeof(float);
    cudaError_t err;

    err = cudaMalloc((void **)&A_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&B_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **)&C_d, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device memory - %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);

    // Part 2: call kernel to launch a grid of thread to perfom the operation
    vecAddKernel<<<ceil(n / 256.0), 256>>>(A_d, B_d, C_d, n);

    // Part 3: copy C to host and free device vectors
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main()
{
    // Declare variables
    int n = 65536;
    float *A_h = (float *)malloc(n * sizeof(float));
    float *B_h = (float *)malloc(n * sizeof(float));
    float *C_h = (float *)malloc(n * sizeof(float));

    // Init of vectors
    for (int i = 0; i < n; i++)
    {
        A_h[i] = i + 1.25f;
        B_h[i] = i * 1.25f;
    }

    // Perform vector addition
    vecAdd(A_h, B_h, C_h, n);

    // Print results
    for (int i = 0; i < n; i++)
    {
        printf("elem %d is %.2f\n", i, C_h[i]);
    }

    // Free memory !!
    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}