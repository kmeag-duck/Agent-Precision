// Element-wise vector addition: z(i) = x(i) + y(i).
//
// Verdict: LOWERABLE to float.
// Why: single op per element, no accumulation, no cancellation. Relative
// error per element is bounded by ulp(float) ~ 1.2e-7 regardless of N.
// Suggested test: x, y ~ U(-1, 1), N = 1<<20. Tolerance: rtol = 1e-6.

#include <Kokkos_Core.hpp>

void vector_add(Kokkos::View<const double*> x,
                Kokkos::View<const double*> y,
                Kokkos::View<double*> z) {
  const int N = z.extent(0);
  Kokkos::parallel_for("vector_add", N, KOKKOS_LAMBDA(int i) {
    z(i) = x(i) + y(i);
  });
}
