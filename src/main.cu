#include <cuda.h>
#include <memory>

template <typename T, int N> struct DeviceArray {
  T *a;
  DeviceArray() { cudaMalloc((void **)a, N * sizeof(T)); }
  ~DeviceArray() { cudaFree(a); }
  __device__ __host__ int size() const { return N; }
  __device__ __host__ T &operator[](int i) { return a[i]; }
  __device__ __host__ const T &operator[](int i) const { return a[i]; }
  __device__ __host__ T *raw() { return a; }
  __device__ __host__ const T *raw() const { return a; }
};

using Array = DeviceArray<int, 100>;

__global__ void vec_add(Array &a, Array &b) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < a.size()) a[i] += b[i];
}

int main() { return 0; }