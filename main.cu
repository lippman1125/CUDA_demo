#include <cuda_runtime.h>
#include <stdio.h>
#include "utils.h"

extern __global__ void MatrixMultiply(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k);
extern void MatrixMultiply_CPU_Native(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k);
extern void MatrixMultiply_CPU_OPT1(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k);
extern void MatrixMultiply_CPU_OPT2(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k, int b);

__global__ void Print(void)
{
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;
    printf("%d, %d\n", idx, idy);
}

int main(int argc, char**argv)
{
    if (argc < 6) {
        printf("Usage: \n");
        printf("    matrix_multiply  m  n  k  b_h b_w\n");
    }
    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);
    int b_h = atoi(argv[4]);
    int b_w = atoi(argv[5]); 
    printf("matrix A is [%d %d]\n", m, k);
    printf("matrix B is [%d %d]\n", k, n);
    printf("matrix C is [%d %d]\n", m, n);
    printf("block is [%d %d]\n", b_h, b_w);
    // initialize gpu device
    initDevice(0);
    
    int bytes_a = m*k*sizeof(float);
    int bytes_b = k*n*sizeof(float);
    int bytes_c = m*n*sizeof(float);

    //Malloc
    printf("Malloc memory on Host\n");
    float * a_host = (float*)malloc(bytes_a);
    if (a_host == NULL) {
        printf("matrix a_host malloc fail\n");
        return 0;
    }
    float * b_host = (float*)malloc(bytes_b);
    if (b_host == NULL) {
        printf("matrix b_host malloc fail\n");
        free(a_host);
        return 0;
    }
    float * c_host = (float*)malloc(bytes_c*4); 
    if (c_host == NULL) {
        printf("matrix c_host malloc fail\n");
        free(b_host);
        free(a_host);
        return 0;
    }
    float * c_host_opt1 = (float*)((char*)c_host + bytes_c);
    float * c_host_opt2 = (float*)((char*)c_host + bytes_c*2);
    float * c_from_dev  = (float*)((char*)c_host + bytes_c*3); 

    printf("Malloc memory on Device\n");
    // Cuda Malloc
    float *a_dev=NULL;
    float *b_dev=NULL;
    float *c_dev=NULL;
    CHECK(cudaMalloc((void**)&a_dev, bytes_a));
    CHECK(cudaMalloc((void**)&b_dev, bytes_b));
    CHECK(cudaMalloc((void**)&c_dev, bytes_c));

    CHECK(cudaMemcpy(a_dev, a_host, bytes_a, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_dev, b_host, bytes_b, cudaMemcpyHostToDevice));  

    // record time
    long long s_t;
    long long e_t;

    printf("CPU Native Execution...\n");
    // CPU matrix multiply native
    s_t = cpu_msec();
    MatrixMultiply_CPU_Native(a_host, b_host, c_host, m, n, k);
    e_t = cpu_msec();
    printf("CPU Native Execution Time elapsed %llu msec\n", e_t - s_t);

    printf("CPU OPT1 Execution...\n");
    // CPU matrix multiply op1
    s_t = cpu_msec();
    MatrixMultiply_CPU_OPT1(a_host, b_host, c_host_opt1, m, n, k);
    e_t = cpu_msec();
    if (is_matrix_equal(c_host, c_host_opt1, m, n)) {
        printf("CPU OPT1 Execution Time elapsed %llu msec\n", e_t - s_t);
    }
    
    printf("CPU OPT2 Execution...\n");
    // CPU matrix multiply op2
    s_t = cpu_msec();
    MatrixMultiply_CPU_OPT2(a_host, b_host, c_host_opt2, m, n, k, 4);
    e_t = cpu_msec();
    if (is_matrix_equal(c_host, c_host_opt2, m, n)) {
        printf("CPU OPT2 Execution Time elapsed %llu msec\n", e_t - s_t);
    }
    // CUDA matrix_multiply
    dim3 block(b_h, b_w);
    dim3 grid((m-1)/b_h+1, (n-1)/b_w+1);

    printf("GPU Execution...\n");
    s_t = cpu_msec();
    #// MatrixMultiply<<<grid, block>>>(a_dev, b_dev, c_dev, m, n, k);
    Print<<<grid, block>>>();
    cudaError_t cudaStatus = cudaGetLastError();
    printf("CUDA error code=%d, reason=%s", cudaStatus, cudaGetErrorString(cudaStatus));
    CHECK(cudaDeviceSynchronize());
    e_t = cpu_msec();
    CHECK(cudaMemcpy(c_from_dev, c_dev, bytes_c, cudaMemcpyDeviceToHost));
    if (is_matrix_equal(c_host, c_from_dev, m, n)) {
        printf("GPU Execution configuration<<<(%d,%d), (%d,%d)>>> Time elapsed %llu msec\n",
            grid.x,grid.y,block.x,block.y, e_t - s_t);
    }

    return 0;
}
