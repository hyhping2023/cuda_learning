#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void TwoPassSimpleKernel(const float* input, float* part_sum, size_t n) {
    extern __shared__ float shm[];
    size_t block_size = n / gridDim.x;
    size_t blk_start = block_size * blockIdx.x;
    size_t blk_end = min(n, blk_start + block_size);
    
    input += blk_start;
    part_sum += blockIdx.x;

    size_t thread_size = block_size / blockDim.x;
    size_t thr_begin = threadIdx.x * thread_size;
    size_t thr_end = min(block_size, thr_begin + thread_size);

    float sum = 0.0f;
    for (size_t i = thr_begin; i < thr_end; i++) {
        sum += input[i];
    }
    shm[threadIdx.x] = sum;
    __syncthreads();
    sum = 0.0f;
    if (threadIdx.x == 0) {
        for (size_t i = 0; i < blockDim.x; ++i) {
            sum += shm[i];
        }
        *part_sum = sum;
    }
} 

void ReduceByTwoPass(const float* input, float* sum, size_t n) {
    const int32_t thread_num_per_block = 1024;  // tuned
    int32_t block_num = (n - 1) / thread_num_per_block + 1;
    block_num = min(block_num, 1024);
    // the first pass reduce input[0:n] to part[0:block_num]
    // part_sum[i] stands for the result of i-th block

    float* d_input;
    float* d_part;
    float* d_output;
    float* part = new float[block_num];

    clock_t start = clock();
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_part, block_num * sizeof(float));
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    size_t shm_size = thread_num_per_block * sizeof(float);  // float per thread
    clock_t start1 = clock();

    TwoPassSimpleKernel<<<block_num, thread_num_per_block, shm_size>>>(d_input, d_part, n);
    TwoPassSimpleKernel<<<1, thread_num_per_block, shm_size>>>(d_part, d_output, block_num * thread_num_per_block);
    clock_t end2 = clock();
    printf("ReduceByTwoPass\ncalculation time: %fus\n", (double)(end2 - start1) / CLOCKS_PER_SEC * 1000 * 1000);
    
    cudaMemcpy(sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    printf("I/O time: %fus\n", (double)(end - end2 + start1 - start) / CLOCKS_PER_SEC * 1000 * 1000);
    cudaFree(d_input);
    cudaFree(d_part);
    cudaFree(d_output);
}