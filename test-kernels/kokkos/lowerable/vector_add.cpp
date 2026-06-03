// Element-wise vector addition: z(i) = x(i) + y(i).

#include <Kokkos_Core.hpp>

void vector_add(Kokkos::View<const double*> x,
                Kokkos::View<const double*> y,
                Kokkos::View<double*> z) {
  const int N = z.extent(0);
  Kokkos::parallel_for("vector_add", N, KOKKOS_LAMBDA(int i) {
    z(i) = x(i) + y(i);
  });
}
