#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>

int main() {
  // Create a vector on the host
  thrust::host_vector<int> h_vec(10);

  // Generate random numbers
  thrust::generate(h_vec.begin(), h_vec.end(), rand);

  // Transfer to device and sort
  thrust::device_vector<int> d_vec = h_vec;
  thrust::sort(d_vec.begin(), d_vec.end());

  // Copy back to host
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());

  // Print sorted vector
  std::cout << "Sorted vector: ";
  for (int x : h_vec) {
    std::cout << x << " ";
  }
  std::cout << std::endl;

  return 0;
}
