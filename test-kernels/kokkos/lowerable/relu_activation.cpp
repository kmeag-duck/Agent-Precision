// ReLU: y(i) = max(0, x(i)).
//
// Verdict: LOWERABLE to float (often even fp16/bf16 in ML practice).
// Why: pure pointwise selection; no arithmetic that can lose precision.
// Suggested test: x ~ N(0, 1), N = 1<<20. Tolerance: exact match for x > 0,
// exact zero otherwise.

#include <Kokkos_Core.hpp>

void relu(Kokkos::View<const double*> x,
          Kokkos::View<double*> y) {
  const int N = y.extent(0);
  Kokkos::parallel_for("relu", N, KOKKOS_LAMBDA(int i) {
    const double xi = x(i);
    y(i) = xi > 0.0 ? xi : 0.0;
  });
}
