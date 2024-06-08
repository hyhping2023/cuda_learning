#include <cuda.h>
#include <cuda_runtime.h>

/*
This is the basic version of exclusive scan.
*/
__global__ void block_to_part(int* input, int N, int part_num, int*part, int* result){
    if (threadIdx.x != 0){
        return;
    }
    
    for (int start = blockIdx.x;start < part_num;start+=gridDim.x){
        int part_begin = start*blockDim.x;
        int part_end = min((start+1)*blockDim.x,N);
        int part_sum = 0;
        for (int i = part_begin;i < part_end;i++){
            part_sum += input[i];
            result[i] = part_sum;
        }
        part[start] = part_sum;
    }

}

__global__ void part_to_kernel(int* part,int part_num){
    int sum = 0;
    for (int i = 0;i < part_num;i++){
        sum += part[i];
        part[i] = sum;
    }
}

__global__ void part_to_origin(int* part,int* output,int N,int part_num){
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

void exclusive_scan_1(int N, int* input, int* output){
    int part_size = 1024;
    int part_num = (N+part_size-1)/part_size;
    int block_num = min(128,part_num);

    int* d_part;
    int* d_result;
    int* d_input;
    cudaMalloc(&d_part,part_num*sizeof(int));
    cudaMalloc(&d_result,N*sizeof(int));
    cudaMalloc(&d_input,N*sizeof(int));
    cudaMemcpy(d_input,input,N*sizeof(int),cudaMemcpyHostToDevice);

    block_to_part<<<block_num,part_size>>>(d_input,N,part_num,d_part,d_result);
    part_to_kernel<<<1,1>>>(d_part,part_num);
    part_to_origin<<<block_num,part_size>>>(d_part,d_result,N,part_num);

    cudaMemcpy(output,d_result,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_part);
    cudaFree(d_result);
    cudaFree(d_input);

}


__global__ void cudaGenRandomNumArray(int* array, int N){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < N){
        array[index] = index;
    }
}

#define BLOCK_SIZE 1024

void generate_array(int* array, int N){
    int* d_array;
    int block_test_num = (N+BLOCK_SIZE-1)/BLOCK_SIZE;

    cudaMalloc(&d_array,N*sizeof(int));
    cudaGenRandomNumArray<<<block_test_num,1024>>>(d_array,N);
    cudaMemcpy(array,d_array,N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}
