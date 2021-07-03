#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>

void initDevice(int devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d: %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

long long cpu_msec()
{
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return tp.tv_sec*1000 + tp.tv_usec/1000;
}

int is_matrix_equal(float *a, float *b, int m , int n) {
    int equal = 1;

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (a[i*n + j] != b[i*n + j]) {
                equal = 0;
                break;
            }
        }
    }
    return equal;
}

void matrix_print(float *a, int m, int n)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", a[i*n + j]);
        }
        printf("\n");
    }
}