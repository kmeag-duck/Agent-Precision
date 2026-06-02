// Sigmoid activation: y[i] = 1 / (1 + exp(-x[i])).
//
// Verdict: LOWERABLE to float (and routinely run in fp16/bf16 in ML).
// Why: output is bounded in (0, 1); the exp itself is well-conditioned for
// |x| <= 20 or so. Float gives ~ulp(float) relative error across the range.
// Suggested test: x ~ U(-10, 10), N = 1<<20. rtol = 1e-5.

#include <math.h>

__global__ void sigmoid(const double* __restrict__ x,
                        double* __restrict__ y,
                        int N) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) y[i] = 1.0 / (1.0 + exp(-x[i]));
}
