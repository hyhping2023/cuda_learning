#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#define part_size 1024

/*
This version uses shared memory to reduce the number of global memory reads.
*/

__device__ void get_part_by_shm(int* shm){
    if (threadIdx.x == 0){
        int sum = 0;
        for (int i = 0;i < blockDim.x;i++){
            sum += shm[i];
            shm[i] = sum;
        }
    }
    __syncthreads();
}

__global__ void block_to_part_shm(int* input, int N, int part_num, int* part, int* result){
    __shared__ int shm[part_size];
    
    for (int start = blockIdx.x;start < part_num;start+=gridDim.x){
        int index = start*blockDim.x + threadIdx.x;
        // if (index == 1024){
        //     printf("we have 1024 %d %d %d\n",threadIdx.x,blockIdx.x,blockDim.x);
        // }
        shm[threadIdx.x] = input[index];//index < N ? input[index]:0;
        __syncthreads();
        get_part_by_shm(shm);
        __syncthreads();
        if (threadIdx.x == blockDim.x-1){
            part[start] = shm[threadIdx.x];
        }
        if(index < N){
            result[index] = shm[threadIdx.x];
        }
    }

}

__global__ void part_to_kernel_2(int* part,int part_num){
    int sum = 0;
    for (int i = 0;i < part_num;i++){
        sum += part[i];
        part[i] = sum;
        //printf("%d %d\n",i,part[i]);
    }
}

__global__ void part_to_origin_2(int* part,int* output,int N,int part_num){
    for (int part_i = blockIdx.x;part_i < part_num;part_i+=gridDim.x){
        if (part_i == 0){
            continue;
        }
        int index = part_i*blockDim.x + threadIdx.x;
        if (index < N){
            output[index] += part[part_i-1];
        }
    }
}

void exclusive_scan_2(int N, int* input, int* output){
    int part_num = (N+part_size-1)/part_size;
    int block_num = min(128,part_num);

    int* d_part;
    int* d_result;
    int* d_input;
    cudaMalloc(&d_part,part_num*sizeof(int));
    cudaMalloc(&d_result,N*sizeof(int));
    cudaMalloc(&d_input,N*sizeof(int));
    cudaMemcpy(d_input,input,N*sizeof(int),cudaMemcpyHostToDevice);

    block_to_part_shm<<<block_num,part_size>>>(d_input,N,part_num,d_part,d_result);
    part_to_kernel_2<<<1,1>>>(d_part,part_num);
    part_to_origin_2<<<block_num,part_size>>>(d_part,d_result,N,part_num);

    cudaMemcpy(output,d_result,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_part);
    cudaFree(d_result);
    cudaFree(d_input);

}

