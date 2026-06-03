// Per-thread serial accumulator of the harmonic series S = sum_{i=1}^{N} 1/i.
// Each thread sums a contiguous chunk of `per_thread` terms; partials are
// reduced on the host (out of scope here).

__global__ void harmonic_sum(double* __restrict__ partial,
                             int N, int per_thread) {
  const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
  const int start = tid * per_thread + 1;
  const int end   = min(start + per_thread, N + 1);
  double s = 0.0;
  for (int i = start; i < end; ++i) s += 1.0 / (double)i;
  partial[tid] = s;
}
