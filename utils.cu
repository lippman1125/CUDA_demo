#include <stdio.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "utils.h"

void initDevice(int devNum)
{
    int dev = devNum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp,dev));
    printf("Using device %d: %s\n",dev,deviceProp.name);
    CHECK(cudaSetDevice(dev));
}

void initialDataRand(float* ip, int size)
{
    time_t t;
    srand((unsigned )time(&t));
    for(int i=0;i<size;i++)
    {
        ip[i]=(float)(rand()&0xffff)/1000.0f;
    }
}

void initialDataOne(float *ip, int size)
{
    for(int i=0; i<size; i++)
    {
        ip[i]=(float)1.0;
    }
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

int is_matrix_equal2(float *a, float *b, int m , int n) {
    int equal = 1;
    int cnt = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (abs(a[i*n + j] - b[i*n + j])>=1.0) {
                equal = 0;
                printf("%f, %f\n", a[i*n+j], b[i*n+j]);
            } else if (abs(a[i*n + j] - b[i*n + j])>1e-6 && abs(a[i*n + j] - b[i*n + j])<1.0) {
                cnt++;
            }
        }
    }
    printf("slight diff cnt=%d\n", cnt);
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

void gpu_info_display(void)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int dev;
    for (dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
        printf("Total amount of global memory                   %u bytes\n", deviceProp.totalGlobalMem);
        printf("Number of mltiprocessors                        %d\n", deviceProp.multiProcessorCount);
        printf("Total amount of constant memory:                %u bytes\n", deviceProp.totalConstMem);
        printf("Total amount of shared memory per block         %u bytes\n", deviceProp.sharedMemPerBlock);
        printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
        printf("Warp size                                       %d\n", deviceProp.warpSize);
        printf("Maximum number of threada per block:            %d\n", deviceProp.maxThreadsPerBlock);
        printf("Maximum sizes of each dimension of a block:     %d x %d x %d\n", deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
        printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("Maximum memory pitch :                          %u bytes\n", deviceProp.memPitch);
        printf("Texture alignmemt                               %u bytes\n", deviceProp.texturePitchAlignment);
        printf("Clock rate                                      %.2f GHz\n", deviceProp.clockRate*1e-6f);
    }
    printf("\nTest PASSED\n");
}
