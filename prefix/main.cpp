#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

using namespace std;

#define TEST 1073741824

void exclusive_scan_1(int N, int* input, int* output);
void exclusive_scan_2(int N, int* input, int* output);
void exclusive_scan_3(int N, int* input, int* output);
void generate_array(int* array, int N);
void check(int* array1, int* array2, int N);

int main(){
    int* array = new int[TEST];
    generate_array(array,TEST);

    // for (int i = 0;i < 10;i++){
    //     printf("%d\n",array[i]);
    // }

    int* output = new int[TEST];

    clock_t start_time,end_time;
    start_time = clock();

    exclusive_scan_1(TEST,array,output);

    end_time = clock();
    printf("Time: %f s\n",(double)(end_time - start_time)/CLOCKS_PER_SEC);

    // for (int i = 0;i < 10;i++){
    //     printf("%d\n",output[i]);
    // }

    // delete[] output;

    int* output2 = new int[TEST];
    sleep(5);

    start_time = clock();

    exclusive_scan_2(TEST,array,output2);

    end_time = clock();
    printf("Time: %f s\n",(double)(end_time - start_time)/CLOCKS_PER_SEC);

    // for (int i = 1022;i < 1280;i++){
    //     printf("%d %d\n",output[i],output2[i]);
    // }

    // delete[] output;

    int* output3 = new int[TEST];
    sleep(5);

    start_time = clock();

    exclusive_scan_3(TEST,array,output3);

    end_time = clock();
    printf("Time: %f s\n",(double)(end_time - start_time)/CLOCKS_PER_SEC);

    // for (int i = 0;i < 10;i++){
    //     printf("%d\n",output[i]);
    // }

    // delete[] output;

    printf("checking 1 2\n");
    check(output,output2,TEST);
    printf("checking 1 3\n");
    check(output,output3,TEST);
    printf("checking 2 3\n");
    check(output2,output3,TEST);

    delete[] array;
    delete[] output;
    delete[] output2;
    delete[] output3;
    return 0;
}
