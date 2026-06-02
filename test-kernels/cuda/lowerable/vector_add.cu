// Element-wise vector addition: z[i] = x[i] + y[i].
//
// Verdict: LOWERABLE to float.
// Why: single op per element, no accumulation. Relative error bounded by
// ulp(float) regardless of N.
// Suggested test: x, y ~ U(-1, 1), N = 1<<20, block = 256. rtol = 1e-6.

__global__ void vector_add(const double* __restrict__ x,
                           const double* __restrict__ y,
                           double* __restrict__ z,
                           int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) z[i] = x[i] + y[i];
}
