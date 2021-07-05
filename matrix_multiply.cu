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
