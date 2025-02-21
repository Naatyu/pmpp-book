#include <stdio.h>

#define THREADS_PER_BLOCK_X 2
#define THREADS_PER_BLOCK_Y 2

__global__ void squareMatMulKernel(float *M, float *N, float *P, int size)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < size && col < size)
    {
        float Pvalue = 0;
        for (int i = 0; i < size; i++)
        {
            Pvalue += M[row * size + i] * N[col + i * size];
        }
        P[row * size + col] = Pvalue;
    }
}

__global__ void rowSquareMatMulKernel(float *M, float *N, float *P, int size)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < size)
    {
        for (int n = 0; n < size; n++){
            float Pvalue = 0;
            for (int i = 0; i < size; i++)
            {
                Pvalue += M[row * size + i] * N[n + i * size];
            }
            P[row * size + n] = Pvalue;
        }
    }
}

__global__ void colSquareMatMulKernel(float *M, float *N, float *P, int size)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (col < size)
    {
        for (int n = 0; n < size; n++){
            float Pvalue = 0;
            for (int i = 0; i < size; i++)
            {
                Pvalue += M[n * size + i] * N[col + i * size];
            }
            P[n * size + col] = Pvalue;
        }
    }
}

__global__ void squareVecMatMulKernel(float *M, float *N, float *P, int size){
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < size){
        float Pvalue = 0;
        for (int i = 0; i < size; i++){
            Pvalue += M[row * size + i] * N[row];
        }
        P[row] = Pvalue;
    }
}

void squareMatMul(float *M, float *N, float *P, int size)
{
    // Compute size
    int allocSize = size * size * sizeof(float);

    // Allocate memory in device
    float *M_d;
    float *N_d;
    float *P_d;
    cudaMalloc((void **)&M_d, allocSize);
    cudaMalloc((void **)&N_d, allocSize);
    cudaMalloc((void **)&P_d, allocSize);

    // Copy to device
    cudaMemcpy(M_d, M, allocSize, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N, allocSize, cudaMemcpyHostToDevice);
    cudaMemcpy(P_d, P, allocSize, cudaMemcpyHostToDevice);

    // Call kernel
    dim3 dimGrid(ceil(THREADS_PER_BLOCK_X), ceil(THREADS_PER_BLOCK_Y), 1);
    dim3 dimBlock(2, 2, 1);
    squareMatMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, size);

    // Copy back P
    cudaMemcpy(P, P_d, allocSize, cudaMemcpyDeviceToHost);

    // Free
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(P_d);

    return;
}

int main()
{
    const int size = 3;
    float M[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    float N[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float P[size * size];

    squareMatMul(M, N, P, size);

    for (int i = 0; i < size; i++)
    {
        for (int k = 0; k < size; k++)
        {
            printf("%.2f ", P[i * size + k]);
        }
        printf("\n");
    }

    return 0;
}