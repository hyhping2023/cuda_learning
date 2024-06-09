#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void TwoPassInterleavedKernel(const float* input, float* part_sum, size_t n){
    int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_thread_num = gridDim.x * blockDim.x;

    double sum = 0.0f;
    for (int32_t i = gtid; i<n; i += total_thread_num){
        sum += input[i];
    }

    extern __shared__ float shm[];
    shm[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0){
        sum = 0.0f;
        for (size_t i=0; i < blockDim.x; ++i){
            sum += shm[i];
        }
        part_sum[blockIdx.x] = sum;
    }
}

void ReduceByTwoPassWithBetterAddress(const float* input, float* sum, size_t n) {
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

    TwoPassInterleavedKernel<<<block_num, thread_num_per_block, shm_size>>>(d_input, d_part, n);

    cudaMemcpy(part, d_part, block_num*sizeof(float), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < block_num; ++i){
    //     if (part[i] == 0){
    //         printf("part(%d): %f\n", i, part[i]);
    //         break;
    //     }
    //     printf("part(%d): %f\n", i, part[i]);
    // }
    // printf("%d\n", block_num);
    TwoPassInterleavedKernel<<<1, block_num, shm_size>>>(d_part, d_output, block_num);
    clock_t end2 = clock();
    printf("ReduceByTwoPassWithBetterAddress\ncalculation time: %fus\n", (double)(end2 - start1) / CLOCKS_PER_SEC * 1000 * 1000);
    
    cudaMemcpy(sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    printf("I/O time: %fus\n", (double)(end - end2 + start1 - start) / CLOCKS_PER_SEC * 1000 * 1000);
    printf("Total time: %fus\n", (double)(end - start) / CLOCKS_PER_SEC * 1000 * 1000);
    cudaFree(d_input);
    cudaFree(d_part);
    cudaFree(d_output);
}