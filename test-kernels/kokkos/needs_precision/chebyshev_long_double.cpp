// Evaluate T_n(x) via the Chebyshev recurrence
//   T_0 = 1, T_1 = x, T_{k+1} = 2x*T_k - T_{k-1}
// for many x values in [-1, 1].

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
