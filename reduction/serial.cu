#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

#define BLOCK_SIZE 1024

__global__ void SerialKernel(const float* input, float* output, size_t n) {
    double sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += input[i];
    }
    *output = sum;
}

void ReduceByKernel(const float* input, float* output, size_t n) {
    float* d_input;
    float* d_output;
    clock_t start = clock();
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
    clock_t start2 = clock();
    SerialKernel<<<1, 1>>>(d_input, d_output, n);
    clock_t end2 = clock();
    cudaMemcpy(output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    printf("ReduceByKernel\ncalculation time: %fus\n", (double)(end2 - start2) / CLOCKS_PER_SEC * 1000 * 1000);
    printf("I/O time: %fus\n", (double)(end - end2 + start2 - start) / CLOCKS_PER_SEC * 1000 * 1000);
    printf("Total time: %fus\n", (double)(end - start) / CLOCKS_PER_SEC * 1000 * 1000);
    cudaFree(d_input);
    cudaFree(d_output);
}


__global__ void cudaGenRandomNumArray(float* array, int N, curandState* states){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < N){
        curand_init(0, index, 0, &states[index]);
        array[index] = ((int)(curand_uniform(&states[index]) * 2.0)) / 8.0;
    }
}

void generate_array(float* array, int N){
    float* d_array;
    int block_test_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    curandState* states;

    cudaMalloc(&states, N * sizeof(curandState));
    cudaMalloc(&d_array,N*sizeof(int));
    cudaGenRandomNumArray<<<block_test_num,1024>>>(d_array, N, states);
    cudaMemcpy(array,d_array,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_array);
    cudaFree(states);
}
