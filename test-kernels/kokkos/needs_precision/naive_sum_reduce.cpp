// Sum a vector via Kokkos parallel_reduce: sum = sum_i x(i).

#include <Kokkos_Core.hpp>

double naive_sum(Kokkos::View<const double*> x) {
  double sum = 0.0;
  const int N = x.extent(0);
  Kokkos::parallel_reduce("naive_sum", N,
    KOKKOS_LAMBDA(int i, double& acc) { acc += x(i); },
    sum);
  return sum;
}
