#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
using namespace std;

__global__ void check_array(int* array1, int* array2, int N){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
        if (array1[index] != array2[index]){
           printf("Error at index %d: %d != %d\n", index, array1[index], array2[index]);
        }
    }
}

void check(int* array1, int* array2, int N){
    int block_size = 1024;
    int grid_size = (N + block_size - 1) / block_size;

    int* d_array1;
    int* d_array2;
    int* output = (int*)malloc(N * sizeof(int));
    cudaMalloc(&d_array1, N * sizeof(int));
    cudaMalloc(&d_array2, N * sizeof(int));
    cudaMemcpy(d_array1, array1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_array2, array2, N * sizeof(int), cudaMemcpyHostToDevice);

    check_array<<<grid_size, block_size>>>(d_array1, d_array2, N);

    cudaFree(d_array1);
    cudaFree(d_array2);

    return;

}