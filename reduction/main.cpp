#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

//此处如果TEST过大会出现float精度问题！！！
//计算不是真计算，还要考虑I/O时间
#define TEST 1024 * 1024 * 128

void generate_array(float* array, int N);
void ReduceByKernel(const float* input, float* output, size_t n);
void ReduceByTwoPass(const float* input, float* sum, size_t n);
void ReduceByTwoPassWithBetterAddress(const float* input, float* sum, size_t n);
void ReduceByTwoPassSharedOptimizedKernel(const float* input, float* sum, size_t n);
void ReduceByTwoPassWarpSyncKernel(const float* input, float* sum, size_t n);

int main(){
    float* array = new float[TEST];
    float* output = new float;
    generate_array(array,TEST);
    printf("ARRAY: %f\n", array[0]);
    double sum =0.0;
    clock_t start = clock();
    for (int i = 0 ;i<TEST;++i){
        sum+=array[i];
        // if (i && (i+1) % 100000 == 0.0){
        //     printf("%d, %f\n", i+1 , sum *8.0);
        // }
        // if (4.0*sum  < i + 1 ){
        //     printf("!!!%d, %f\n", i , sum * 4.0);
        //     break;
        // }
        // if (array[i] != 0.125){
        //     printf("Wrong at %d, %f\n", i, array[i]);
        //     break;
        // }
    }
    clock_t end = clock();
    printf("CPU\ncalculation time: %fus\n", (double)(end - start) / CLOCKS_PER_SEC * 1000 * 1000);
  
    printf("ARRAY SUM: %f\n", sum);

    // ReduceByKernel(array, output, TEST);
    // printf("OUTPUT: %f\n", *output);

    output = new float;

    for (int times =0 ; times < 10; times++){
        printf("\n\n\nTIMES %d \n", times+1);

        ReduceByKernel(array, output, TEST);
        printf("OUTPUT: %f\n\n", *output);

        ReduceByTwoPass(array, output, TEST);
        printf("OUTPUT: %f\n\n", *output);

        ReduceByTwoPassWithBetterAddress(array, output, TEST);
        printf("OUTPUT: %f\n\n", *output);

        ReduceByTwoPassSharedOptimizedKernel(array, output, TEST);
        printf("OUTPUT: %f\n\n", *output);

        ReduceByTwoPassWarpSyncKernel(array, output, TEST);
        printf("OUTPUT: %f\n\n", *output);
    }

    delete[] array;
    delete[] output;
    return 0;

}
