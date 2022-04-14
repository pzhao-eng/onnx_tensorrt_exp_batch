#include "print_array.h"

__global__ void print_device_kernel(float *array, int size)
{
    int idx = threadIdx.x;
    if (idx >= size) return;
    printf("%f ", array[idx]);
}

void print_device(void *arr, int size)
{
    float *f_arr = reinterpret_cast<float *>(arr);
    print_device_kernel<<<1, size>>>(f_arr, size);
}