// One-pass naive variance via E[x^2] - E[x]^2.
//
// Verdict: MUST STAY double (and even double is fragile — Welford is the
// correct fix; this kernel intentionally encodes the unsafe formulation).
// Why: catastrophic cancellation. When the mean is large relative to the
// spread (e.g. x ~ U(1e6, 1e6 + 1)), sxx/N and mean*mean are both ~1e12
// while their difference is ~1/12. In float, ulp(1e12) ~ 6e4, so the
// subtraction destroys every significant digit.
// Suggested test: x(i) = 1e6 + uniform(0, 1), N = 1<<20. True variance
// ~ 0.0833. Require absolute error < 0.01 from a Welford reference.

#include <Kokkos_Core.hpp>
#include <Kokkos_Pair.hpp>

Kokkos::pair<double, double>
mean_variance_naive(Kokkos::View<const double*> x) {
  const int N = x.extent(0);
  double sx = 0.0, sxx = 0.0;
  Kokkos::parallel_reduce("sum_x", N,
    KOKKOS_LAMBDA(int i, double& a) { a += x(i); }, sx);
  Kokkos::parallel_reduce("sum_xx", N,
    KOKKOS_LAMBDA(int i, double& a) { a += x(i) * x(i); }, sxx);
  const double mean = sx / N;
  const double var  = sxx / N - mean * mean;
  return {mean, var};
}
