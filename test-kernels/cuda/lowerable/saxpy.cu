// SAXPY: y[i] = a * x[i] + y[i].

__global__ void saxpy(double a,
                      const double* __restrict__ x,
                      double* __restrict__ y,
                      int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) y[i] = a * x[i] + y[i];
}
