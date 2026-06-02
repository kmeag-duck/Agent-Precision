// Evaluate T_n(x) via the Chebyshev recurrence
//   T_0 = 1, T_1 = x, T_{k+1} = 2x*T_k - T_{k-1}
// for many x values in [-1, 1].
//
// Verdict: MUST STAY at least double; long double (80-bit on x86 Linux)
// recommended. This is a host-only kernel — long double is not a portable
// device type in Kokkos, so we restrict to RangePolicy<Kokkos::Serial> on
// HostSpace views.
// Why: the recurrence amplifies roundoff. For x = cos(theta) with theta
// small, neighboring T_k differ by O(theta^2), and float roundoff dominates
// after a few hundred iterations. At n = 10000, float yields O(1) errors;
// double loses ~10 decimal digits; long double retains ~14.
// Suggested test: x(i) = 1 - 1e-3 * uniform(0, 1), n = 10000, N = 1<<16.
// Tolerance against a long-double reference: rtol = 1e-3.

#include <Kokkos_Core.hpp>

void chebyshev_many(Kokkos::View<const long double*, Kokkos::HostSpace> x,
                    Kokkos::View<long double*, Kokkos::HostSpace> out,
                    int n) {
  const int N = x.extent(0);
  Kokkos::parallel_for("chebyshev",
    Kokkos::RangePolicy<Kokkos::Serial>(0, N),
    [=] (int i) {
      long double tkm1 = 1.0L;
      long double tk   = x(i);
      for (int k = 1; k < n; ++k) {
        const long double tkp1 = 2.0L * x(i) * tk - tkm1;
        tkm1 = tk;
        tk   = tkp1;
      }
      out(i) = tk;
    });
}
