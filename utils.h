#ifndef _UTILS_H_
#define _UTILS_H_

#define CHECK(call)                             \
{                                               \
  const cudaError_t error=call;                 \
  if(error!=cudaSuccess)                        \
  {                                             \
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}

void initDevice(int devNum);
void initialDataOne(float *ip, int size);
void initialDataRand(float *ip, int size);
long long cpu_msec();
int is_matrix_equal(float *a, float *b, int m, int n);
int is_matrix_equal2(float *a, float *b, int m, int n);
void matrix_print(float *a, int m, int n);
void gpu_info_display(void);

#endif /*_UTILS_H_*/
