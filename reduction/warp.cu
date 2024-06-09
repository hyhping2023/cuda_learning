#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

__global__ void TwoPassWarpSyncKernel(const float* input, float* part_sum, size_t n){
    int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int32_t total_thread_num = gridDim.x * blockDim.x;

    double sum = 0.0f;
    for (int32_t i = gtid; i < n; i += total_thread_num){
        sum += input[i];
    }

    extern __shared__ float shm[];
    shm[threadIdx.x] = sum;
    __syncthreads();
    //保证连续内存读写
    for (int32_t activare_thread_num = blockDim.x / 2; activare_thread_num > 32; activare_thread_num /= 2){
        if (threadIdx.x < activare_thread_num){
            shm[threadIdx.x] += shm[threadIdx.x + activare_thread_num];
        }
        __syncthreads();
    }
    //一个warp32个线程，warp内必然同步（注意这里是无分支情况），不需要syncwarp
    if (threadIdx.x < 32){
        volatile float* vshm = shm;
        if (blockDim.x >= 64){
            vshm[threadIdx.x] += vshm[threadIdx.x + 32];
        }
        vshm[threadIdx.x] += vshm[threadIdx.x + 16];
        vshm[threadIdx.x] += vshm[threadIdx.x + 8];
        vshm[threadIdx.x] += vshm[threadIdx.x + 4];
        vshm[threadIdx.x] += vshm[threadIdx.x + 2];
        vshm[threadIdx.x] += vshm[threadIdx.x + 1];
        if (threadIdx.x == 0) {
            part_sum[blockIdx.x] = vshm[0];
        }
    }
}

__global__ void TwoPassSharedOptimizedKernel(const float* input, float* part_sum, size_t n) {
    int32_t gtid = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
    int32_t total_thread_num = gridDim.x * blockDim.x;
    // reduce
    //   input[gtid + total_thread_num * 0]
    //   input[gtid + total_thread_num * 1]
    //   input[gtid + total_thread_num * 2]
    //   input[gtid + total_thread_num * ...]
    double sum = 0.0f;
    for (int32_t i = gtid; i < n; i += total_thread_num) {
        sum += input[i];
    }
    // store sum to shared memory
    extern __shared__ float shm[];
    shm[threadIdx.x] = sum;
    __syncthreads();
    // reduce shm to part_sum
    for (int32_t active_thread_num = blockDim.x / 2; active_thread_num >= 1;
        active_thread_num /= 2) {
        if (threadIdx.x < active_thread_num) {
        shm[threadIdx.x] += shm[threadIdx.x + active_thread_num];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part_sum[blockIdx.x] = shm[0];
    }
}

void ReduceByTwoPassSharedOptimizedKernel(const float* input, float* sum, size_t n) {
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

    TwoPassSharedOptimizedKernel<<<block_num, thread_num_per_block, shm_size>>>(d_input, d_part, n);

    cudaMemcpy(part, d_part, block_num*sizeof(float), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < block_num; ++i){
    //     if (part[i] == 0){
    //         printf("part(%d): %f\n", i, part[i]);
    //         break;
    //     }
    //     printf("part(%d): %f\n", i, part[i]);
    // }
    // printf("%d\n", block_num);
    TwoPassSharedOptimizedKernel<<<1, block_num, shm_size>>>(d_part, d_output, block_num);
    clock_t end2 = clock();
    printf("TwoPassSharedOptimizedKernel\ncalculation time: %fus\n", (double)(end2 - start1) / CLOCKS_PER_SEC * 1000 * 1000);
    
    cudaMemcpy(sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    printf("I/O time: %fus\n", (double)(end - end2 + start1 - start) / CLOCKS_PER_SEC * 1000 * 1000);
    printf("Total time: %fus\n", (double)(end - start) / CLOCKS_PER_SEC * 1000 * 1000);
    cudaFree(d_input);
    cudaFree(d_part);
    cudaFree(d_output);
}

void ReduceByTwoPassWarpSyncKernel(const float* input, float* sum, size_t n) {
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

    TwoPassWarpSyncKernel<<<block_num, thread_num_per_block, shm_size>>>(d_input, d_part, n);

    cudaMemcpy(part, d_part, block_num*sizeof(float), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < block_num; ++i){
    //     if (part[i] == 0){
    //         printf("part(%d): %f\n", i, part[i]);
    //         break;
    //     }
    //     printf("part(%d): %f\n", i, part[i]);
    // }
    // printf("%d\n", block_num);
    TwoPassWarpSyncKernel<<<1, block_num, shm_size>>>(d_part, d_output, block_num);
    clock_t end2 = clock();
    printf("TwoPassWarpSyncKernel\ncalculation time: %fus\n", (double)(end2 - start1) / CLOCKS_PER_SEC * 1000 * 1000);
    
    cudaMemcpy(sum, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    clock_t end = clock();
    printf("I/O time: %fus\n", (double)(end - end2 + start1 - start) / CLOCKS_PER_SEC * 1000 * 1000);
    printf("Total time: %fus\n", (double)(end - start) / CLOCKS_PER_SEC * 1000 * 1000);
    cudaFree(d_input);
    cudaFree(d_part);
    cudaFree(d_output);
}
