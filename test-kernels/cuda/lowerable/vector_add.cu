// Element-wise vector addition: z[i] = x[i] + y[i].

__global__ void vector_add(const double* __restrict__ x,
                           const double* __restrict__ y,
                           double* __restrict__ z,
                           int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) z[i] = x[i] + y[i];
}
