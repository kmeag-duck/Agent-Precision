// Sigmoid activation: y[i] = 1 / (1 + exp(-x[i])).

#include <math.h>

__global__ void sigmoid(const double* __restrict__ x,
                        double* __restrict__ y,
                        int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) y[i] = 1.0 / (1.0 + exp(-x[i]));
}
