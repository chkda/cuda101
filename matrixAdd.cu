#include <cstddef>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void matrixAdd(const float *A, const float *B, float *C, int nRows,
                          int nCols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < nRows && col < nCols) {
    int idx = row * nCols + col;
    C[idx] = A[idx] + B[idx];
  }
}

#define cudaCheckError(ans)                                                    \
  {                                                                            \
    cudaAssert(ans, __FILE__, __LINE__);                                       \
  }

inline void cudaAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

void printConfig(dim3 gridDim, dim3 blockDim, int nRows, int nCols) {
  printf("Matrix Dimensions: %d x %d\n", nRows, nCols);
  printf("Block dimensions: %d x %d\n", blockDim.x, blockDim.y);
  printf("Grid dimensions: %d x %d\n", gridDim.x, gridDim.y);
  printf("Total threads: %d\n",
         gridDim.x * gridDim.y * blockDim.x * blockDim.y);
}

void initMatrix(float *matrix, int rows, int cols, float value) {
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = value;
  }
}

int main() {
  int nRows = 1024;
  int nCols = 1024;
  size_t size = nRows * nCols * sizeof(float);

  float *A_h = (float *)malloc(size);
  float *B_h = (float *)malloc(size);
  float *C_h = (float *)malloc(size);

  initMatrix(A_h, nRows, nCols, 1.0f);
  initMatrix(B_h, nRows, nCols, 2.0f);

  float *A_d, *B_d, *C_d;
  cudaCheckError(cudaMalloc(&A_d, size));
  cudaCheckError(cudaMalloc(&B_d, size));
  cudaCheckError(cudaMalloc(&C_d, size));

  dim3 blockDim(16, 16, 1);
  dim3 gridDim((int)ceil((float)nCols / blockDim.x),
               (int)ceil((float)nRows / blockDim.y), 1);

  printConfig(gridDim, blockDim, nRows, nCols);

  cudaCheckError(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));
  cudaCheckError(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  matrixAdd<<<gridDim, blockDim>>>(A_d, B_d, C_d, nRows, nCols);

  cudaCheckError(cudaGetLastError());

  cudaCheckError(cudaDeviceSynchronize());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaCheckError(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

  float milliSeconds = 0;
  cudaEventElapsedTime(&milliSeconds, start, stop);
  printf("Kernel execution time:%f\n", milliSeconds);

  bool success = true;

  for (int i = 0; i < nRows * nCols; i++) {
    if (C_h[i] != 3.0f) {
      printf("Verification failed at index %d: %f != 3.0f\n", i, C_h[i]);
      success = false;
      break;
    }
  }

  if (success) {
    printf("vector addition successful");
  }

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
  free(A_h);
  free(B_h);
  free(C_h);

  return 0;
}
