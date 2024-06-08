#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

#define TEST 1024 *2

void generate_array(float* array, int N);
void ReduceByKernel(const float* input, float* output, size_t n);
void ReduceByTwoPass(const float* input, float* sum, size_t n);

int main(){
    float* array = new float[TEST];
    float* output = new float;
    generate_array(array,TEST);
    printf("ARRAY: %f\n", array[0]);

    ReduceByKernel(array, output, TEST);
    printf("OUTPUT: %f\n", *output);

    output = new float;

    ReduceByTwoPass(array, output, TEST);
    printf("OUTPUT: %f\n", *output);

    delete[] array;
    delete[] output;
    return 0;

}
