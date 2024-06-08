#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define part_size 1024

__device__ void warp_scan(int* shm_data, int lane){
    if (lane == 0){
        int sum =0;
        for (int i = 0;i < 32;i++){
            sum += shm_data[i];
            shm_data[i] = sum;
        }
    }
}

__device__ void scan_block(int* shm_data){
    int warp_id = (threadIdx.x >> 5);  //以32个为一组，求组的个数
    int lane_id = threadIdx.x & 31; // 31 = 00011111 可以得到组内序号，相当于 %32， 线程内编号
    __shared__ int warp_sum[32];  // 相同组内编号的线程和 blockDim.x / WarpSize = 32

    warp_scan(shm_data + 32*warp_id, lane_id);
    __syncthreads();

    if (lane_id == 31){   //默认一个block有1024个线程
        warp_sum[warp_id] = shm_data[32*warp_id + 31];
    }
    __syncthreads();

    if (warp_id == 0){
        warp_scan(warp_sum + lane_id, lane_id);
    }
    __syncthreads();

    if (warp_id != 0){
        *(shm_data + 32*warp_id + lane_id) += warp_sum[warp_id-1];
    }
    __syncthreads();

}

__global__ void block_to_part_shm_3(int* input, int N, int part_num, int* part, int* result){
    __shared__ int shm[part_size];
    
    for (int start = blockIdx.x;start < part_num;start+=gridDim.x){
        int index = start*blockDim.x + threadIdx.x;
        shm[threadIdx.x] = index < N ? input[index]:0;
        __syncthreads();
        scan_block(shm);
        __syncthreads();
        if (threadIdx.x == blockDim.x-1){
            part[start] = shm[threadIdx.x];
        }
        if(index < N){
            result[index] = shm[threadIdx.x];
        }
    }

}

__global__ void part_to_kernel_3(int* part,int part_num){
    int sum = 0;
    for (int i = 0;i < part_num;i++){
        sum += part[i];
        part[i] = sum;
    }
}

__global__ void part_to_origin_3(int* part,int* output,int N,int part_num){
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

void exclusive_scan_3(int N, int* input, int* output){
    int part_num = (N+part_size-1)/part_size;
    int block_num = min(128,part_num);

    int* d_part;
    int* d_result;
    int* d_input;
    cudaMalloc(&d_part,part_num*sizeof(int));
    cudaMalloc(&d_result,N*sizeof(int));
    cudaMalloc(&d_input,N*sizeof(int));
    cudaMemcpy(d_input,input,N*sizeof(int),cudaMemcpyHostToDevice);

    block_to_part_shm_3<<<block_num,part_size>>>(d_input,N,part_num,d_part,d_result);
    part_to_kernel_3<<<1,1>>>(d_part,part_num);
    part_to_origin_3<<<block_num,part_size>>>(d_part,d_result,N,part_num);

    cudaMemcpy(output,d_result,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_part);
    cudaFree(d_result);
    cudaFree(d_input);

}
