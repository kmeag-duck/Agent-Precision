// Per-thread serial accumulator of the harmonic series:
//   S = sum_{i=1}^{N} 1/i,   N typically 1e8 to 1e9.
// Each thread sums a contiguous chunk of `per_thread` terms into its own
// accumulator; partial sums are reduced on the host (or in a follow-up
// kernel — out of scope here).
//
// Verdict: MUST STAY double.
// Why: with per_thread = 1024 and N = 1e9, a thread's accumulator climbs to
// ~7 while the terms it is adding shrink toward 1/N ~ 1e-9. In float,
// ulp(7) ~ 1e-6, so any term smaller than that vanishes — and once the
// accumulator passes ~1.0 most subsequent terms in the tail blocks are
// silently truncated. The float result for N = 1e9 underestimates the true
// sum by several percent.
// Suggested test: N = 1<<27, per_thread = 1024, blockDim = 256. Compare
// against a Kahan-summed double reference; require rtol = 1e-5.

__global__ void harmonic_sum(double* __restrict__ partial,
                             int N, int per_thread) {
  const int tid   = blockIdx.x * blockDim.x + threadIdx.x;
  const int start = tid * per_thread + 1;
  const int end   = min(start + per_thread, N + 1);
  double s = 0.0;
  for (int i = start; i < end; ++i) s += 1.0 / (double)i;
  partial[tid] = s;
}
