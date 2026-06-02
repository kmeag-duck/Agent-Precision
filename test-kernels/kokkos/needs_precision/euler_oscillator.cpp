// Forward-Euler integration of an ensemble of simple harmonic oscillators.
// State per particle: (y, v). Update each step:
//   y_{n+1} = y_n + dt * v_n
//   v_{n+1} = v_n - dt * omega^2 * y_n
//
// Verdict: MUST STAY double.
// Why: long-trajectory time integration accumulates roundoff at every step.
// With dt = 1e-4 and n_steps = 1e6 (~16 periods at omega = 1), float gives
// noticeably wrong amplitude/phase; at n_steps >= 1e7 the float trajectory
// diverges completely from the double reference.
// Suggested test: y0 = 1, v0 = 0, omega = 1, dt = 1e-4, n_steps = 1e6,
// N_particles = 1<<14. Compare final (y, v) to double reference;
// require per-particle |delta| < 1e-3.

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
