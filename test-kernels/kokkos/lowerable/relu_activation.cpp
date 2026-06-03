// ReLU activation: y(i) = max(0, x(i)).

#include <Kokkos_Core.hpp>

void relu(Kokkos::View<const double*> x,
          Kokkos::View<double*> y) {
  const int N = y.extent(0);
  Kokkos::parallel_for("relu", N, KOKKOS_LAMBDA(int i) {
    const double xi = x(i);
    y(i) = xi > 0.0 ? xi : 0.0;
  });
}
