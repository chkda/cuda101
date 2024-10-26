#include <stdio.h>

__global__ void helloWorldFromGPU(void) { printf("Hello world from gpu"); }

int main() {
  printf("Hello world from CPU");
  helloWorldFromGPU<<<1, 10>>>();
  return 0;
}
