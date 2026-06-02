// y(i) = a*x(i) + y(i), with |a| <= 10, |x|, |y| ~ O(1).
//
// Verdict: LOWERABLE to float.
// Why: one FMA per element, no inter-element accumulation. Output magnitudes
// stay O(1) so no exponent-range issues either.
// Suggested test: a = 2.5, x, y ~ U(-1, 1), N = 1<<20. Tolerance: rtol = 1e-6.

#include <Kokkos_Core.hpp>

void saxpy(double a,
           Kokkos::View<const double*> x,
           Kokkos::View<double*> y) {
  const int N = y.extent(0);
  Kokkos::parallel_for("saxpy", N, KOKKOS_LAMBDA(int i) {
    y(i) = a * x(i) + y(i);
  });
}
