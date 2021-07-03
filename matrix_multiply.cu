#include<assert.h>

__global__ void MatrixMultiply(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k)
{
    int ix=threadIdx.x+blockDim.x*blockIdx.x;
    int iy=threadIdx.y+blockDim.y*blockIdx.y;
    int idx=iy*n + ix;
    if (iy < m && iy<n)
    {
        for (int i = 0; i < k; k++) {
            mat_c[idx] = mat_a[iy*k + k] * mat_b[k*n + ix];
        }
    }
}

void MatrixMultiply_CPU_Native(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k)
{
    // access matrix a & c in order
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            for (int g = 0; g < k; g++) {
                mat_c[i*n + j] = mat_a[i*k + g] * mat_b[g*n + j];
            }
        }
    }
}

void MatrixMultiply_CPU_OPT1(float * mat_a, float * mat_b, float * mat_c, int m, int n, int k)
{
    // access matrix a & b & c in order
    for (int i = 0; i < m; i++) {
        for (int g = 0; g < k; g++) {
            int tmp = mat_a[i*k + g];
            for (int j = 0; j < n; j++) {
                mat_c[i*n + j] = tmp * mat_b[g*n + j];
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
    
    // split into blocks 
    for (int i = 0; i < b; i++) {
        for (int g = 0; g < b; g++) {
            for (int j = 0; j < b; j++) {
                int im = i * mb;
                int ik = g * kb;
                int in = j * nb;
                // compute each block
                for (int ii = im; ii < im + mb; ii++) {
                    for (int gg = ik; g < ik + kb; gg++) {
                        int tmp = mat_a[ii*k + gg];
                        for (int jj = in; jj < in + nb; jj++) {
                            mat_c[ii*n + jj] = tmp * mat_b[gg*n + jj];        
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
                    mat_c[i*n + j] = tmp * mat_b[g*n + j];        
                }
            }
        }  
    }

}