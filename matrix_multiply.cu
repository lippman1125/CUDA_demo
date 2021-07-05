#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

__global__ void MatrixMultiply(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=iy*n + ix;
    // printf("ix=%d, iy=%d\n", ix, iy);
    if (iy < m && ix<n)
    {
        mat_c[idx] = 0.0;
        for (int i = 0; i < k; i++) {
            mat_c[idx] += mat_a[iy*k + i] * mat_b[i*n + ix];
        }
        // printf("mat_c[%d]=%f\n", idx, mat_c[idx]);
    }
}

#define BLOCKSIZE (32)
__global__ void MatrixMultiplySmem(float *mat_a, float *mat_b, float *mat_c, int m, int n, int k){
    int xb = blockIdx.x;
    int yb = blockIdx.y;
    int x = threadIdx.x;
    int y = threadIdx.y;
    int blockSize = BLOCKSIZE;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    //该线程负责的结果子块C，对应的A和B用于计算的起始子块
    //假设分成9个子块
    //  A11 A12 A13    B11 B12 B13
    //  A21 A22 A23  * B21 B22 B23 
    //  A31 A32 A33    B31 B32 B33  ,则计算C22时这样计算：A21*B12+A22*B22+A23*B32
    // find row    
    float *BeginA = mat_a + yb* blockSize*k;
    float *EndA = BeginA + k;
    // find col
    float *BeginB = mat_b + blockSize*xb;
    int stepA = blockSize;
    int stepB = blockSize*n;
    float tsum = 0;
    //每一个block A和B的子块首地址
    for (; BeginA < EndA; BeginA += stepA, BeginB += stepB){
        // 每个线程load一个元素到shared mem中
        As[y][x] = *(BeginA + y*k + x);
        Bs[y][x] = *(BeginB + y*n + x);
        __syncthreads();//同步
        for (int k = 0; k < blockSize;k++){
            tsum = tsum + As[y][k] * Bs[k][x];
        }
        __syncthreads();//同步，确保该块内所有线程都完成了计算。下次循环，要重复利用共享内存。
    }
    //写入结果 注意坐标的计算方法
    mat_c[yb*blockSize*n + y*n + xb*blockSize + x]=tsum;
}

void MatrixMultiply_CPU_Native(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k)
{
    // access matrix a & c in order
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            mat_c[i*n +j] = 0.0;
            for (int g = 0; g < k; g++) {
                mat_c[i*n + j] += mat_a[i*k + g] * mat_b[g*n + j];
            }
        }
    }
}

void MatrixMultiply_CPU_OPT1(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k)
{
    // access matrix a & b & c in order
    for (int i = 0; i < m; i++) {
        for (int g = 0; g < k; g++) {
            float tmp = mat_a[i*k + g];
            for (int j = 0; j < n; j++) {
                mat_c[i*n + j] += tmp * mat_b[g*n + j];
            }
        }
    }    
}

void MatrixMultiply_CPU_OPT2(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k, int b)
{
    assert(m%b == 0);
    assert(n%b == 0);

    int mb = m/b;
    int nb = n/b;
    int kb = k/b;
    int kkb = k%b;
    // printf("mb=%d, b=%d, kb=%d, kkb=%d\n", mb, nb, kb, kkb);
    
    // split into blocks 
    for (int i = 0; i < b; i++) {
        for (int g = 0; g < b; g++) {
            for (int j = 0; j < b; j++) {
                int im = i * mb;
                int ik = g * kb;
                int in = j * nb;
                // printf("im=%d, ik=%d, in=%d\n", im, ik, in);
                // compute each block
                for (int ii = im; ii < im + mb; ii++) {
                    for (int gg = ik; gg < ik + kb; gg++) {
                        float tmp = mat_a[ii*k + gg];
                        for (int jj = in; jj < in + nb; jj++) {
                            // printf("ii=%d, jj=%d, gg=%d\n", ii, jj, gg);
                            mat_c[ii*n + jj] += tmp * mat_b[gg*n + jj];        
                        }
                    }
                }
            }
        }
    }
    if (kkb) {
        // compute remain
        for (int i = 0; i < m; i++) {
            for (int g = b*nb; g < k; g++) {
                int tmp = mat_a[i*k + g];
                for (int j = 0; j < n; j++) {
                    mat_c[i*n + j] += tmp * mat_b[g*n + j];        
                }
            }
        }  
    }

}
