// Two-pass naive variance: mean = sum(x)/N, var = sum(x^2)/N - mean^2.

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
