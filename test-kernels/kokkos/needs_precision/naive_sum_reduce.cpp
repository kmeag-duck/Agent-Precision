// Sum a vector whose elements span several orders of magnitude.
//
// Verdict: MUST STAY double.
// Why: per-thread serial accumulation in parallel_reduce concentrates many
// terms into one accumulator. With elements decaying like 1/sqrt(i+1) and
// N >= 1e7, partial sums grow to ~6000 while tail terms are ~3e-4 — in float,
// ulp(6000) ~ 5e-4, so most tail contributions are silently dropped and the
// reduced result is off by >1%. Even with a tree reduce, per-thread chunks
// still saturate first.
// Suggested test: x(i) = 1/std::sqrt(double(i+1)), N = 1<<23. Compare against
// a Kahan-summed reference in double; require rtol = 1e-4.

#include <Kokkos_Core.hpp>

double naive_sum(Kokkos::View<const double*> x) {
  double sum = 0.0;
  const int N = x.extent(0);
  Kokkos::parallel_reduce("naive_sum", N,
    KOKKOS_LAMBDA(int i, double& acc) { acc += x(i); },
    sum);
  return sum;
}
