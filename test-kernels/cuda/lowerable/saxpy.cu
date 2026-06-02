// y[i] = a * x[i] + y[i], with |a| <= 10 and |x|, |y| ~ O(1).
//
// Verdict: LOWERABLE to float.
// Why: one FMA per element, no inter-element accumulation, output stays O(1).
// Suggested test: a = 2.5, x, y ~ U(-1, 1), N = 1<<20. rtol = 1e-6.

__global__ void saxpy(double a,
                      const double* __restrict__ x,
                      double* __restrict__ y,
                      int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) y[i] = a * x[i] + y[i];
}
