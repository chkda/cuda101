#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) {
    c[tid] = a[tid] + b[tid];
  }
}

#define cudaCheckError(ans)                                                    \
  {                                                                            \
    cudaAssert((ans), __FILE__, __LINE__);                                     \
  }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

int main() {
  int N = 1 << 20;
  size_t size = N * sizeof(float);

  float *a_h = (float *)malloc(size);
  float *b_h = (float *)malloc(size);
  float *c_h = (float *)malloc(size);

  for (int i = 0; i < N; i++) {
    a_h[i] = 1.0f;
    b_h[i] = 2.0f;
  }

  float *a_d, *b_d, *c_d;
  cudaCheckError(cudaMalloc(&a_d, size));
  cudaCheckError(cudaMalloc(&b_d, size));
  cudaCheckError(cudaMalloc(&c_d, size));

  cudaCheckError(cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice));

  int threadsPerBlock = 256;
  int blocksPerGrid = (int)ceil((float)N / threadsPerBlock);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, N);

  cudaCheckError(cudaGetLastError());

  cudaCheckError(cudaDeviceSynchronize());

  cudaCheckError(cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost));

  bool success = true;
  for (int i = 0; i < N; i++) {
    if (c_h[i] != 3.0f) {
      printf("Verification failed at index %d: %f != 3.0f\n", i, c_h[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("vector addition successful");
  }

  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
  free(a_h);
  free(b_h);
  free(c_h);

  return 0;
}
