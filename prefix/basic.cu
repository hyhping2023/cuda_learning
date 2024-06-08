#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define BLOCK_SIZE 512
#define ARRAY_SIZE 500000000

__global__ void vecAdd(float* a,float* b, float* c){
    int index =blockIdx.x*blockDim.x + threadIdx.x;
    c[index] = a[index] + b[index];
}

int main(){
    float *h_a,*h_b,*h_c;
    size_t size = ARRAY_SIZE*sizeof(float);

    h_a = (float* )malloc(size);
    h_b = (float* )malloc(size);
    h_c = (float* )malloc(size);
    for (int i=0;i<ARRAY_SIZE;++i){
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
        h_c[i] = 0.0f;
    }

    float *a,*b,*c;
    cudaMalloc(&a,size);
    cudaMalloc(&b,size);
    cudaMalloc(&c,size);
    cudaMemcpy(a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(b,h_b,size,cudaMemcpyHostToDevice);

    int BLOCK_PER_GRID = (ARRAY_SIZE + BLOCK_SIZE -1)/BLOCK_SIZE;
    vecAdd<<<BLOCK_PER_GRID,BLOCK_SIZE>>>(a,b,c);

    cudaMemcpy(h_c,c,size,cudaMemcpyDeviceToHost);

    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    for (int i=0;i<10;++i){
        printf("%f\n",h_c[i]);
    }
}
