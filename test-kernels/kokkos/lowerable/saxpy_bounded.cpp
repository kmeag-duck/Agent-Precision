// SAXPY: y(i) = a*x(i) + y(i).

#include <Kokkos_Core.hpp>

void saxpy(double a,
           Kokkos::View<const double*> x,
           Kokkos::View<double*> y) {
  const int N = y.extent(0);
  Kokkos::parallel_for("saxpy", N, KOKKOS_LAMBDA(int i) {
    y(i) = a * x(i) + y(i);
  });
}
