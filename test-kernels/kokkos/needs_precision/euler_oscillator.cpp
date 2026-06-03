// Forward-Euler integration of an ensemble of simple harmonic oscillators.
// State per particle: (y, v). Update each step:
//   y_{n+1} = y_n + dt * v_n
//   v_{n+1} = v_n - dt * omega^2 * y_n

#include <Kokkos_Core.hpp>

void evolve(Kokkos::View<double*> y,
            Kokkos::View<double*> v,
            double dt, double omega, int n_steps) {
  const int N = y.extent(0);
  const double w2 = omega * omega;
  for (int step = 0; step < n_steps; ++step) {
    Kokkos::parallel_for("euler_step", N, KOKKOS_LAMBDA(int i) {
      const double y_new = y(i) + dt * v(i);
      const double v_new = v(i) - dt * w2 * y(i);
      y(i) = y_new;
      v(i) = v_new;
    });
  }
}
