// Direct-summation gravitational N-body step: per-particle force from all
// others (softened: F_i = sum_{j!=i} G*m_j*(r_j-r_i) / (|r_j-r_i|^2 + eps^2)^{3/2}),
// then symplectic-Euler kick + drift.

#include <Kokkos_Core.hpp>
#include <cmath>

void nbody_step(
    Kokkos::View<double*> x,  Kokkos::View<double*> y,  Kokkos::View<double*> z,
    Kokkos::View<double*> vx, Kokkos::View<double*> vy, Kokkos::View<double*> vz,
    Kokkos::View<const double*> m,
    double G, double eps, double dt) {
  const int N = x.extent(0);
  const double eps2 = eps * eps;

  Kokkos::parallel_for("nbody_force", N, KOKKOS_LAMBDA(int i) {
    const double xi = x(i), yi = y(i), zi = z(i);
    double ax = 0.0, ay = 0.0, az = 0.0;
    for (int j = 0; j < N; ++j) {
      if (j == i) continue;
      const double dx = x(j) - xi;
      const double dy = y(j) - yi;
      const double dz = z(j) - zi;
      const double r2     = dx * dx + dy * dy + dz * dz + eps2;
      const double inv_r  = 1.0 / Kokkos::sqrt(r2);
      const double inv_r3 = inv_r * inv_r * inv_r;
      const double s      = G * m(j) * inv_r3;
      ax += s * dx;
      ay += s * dy;
      az += s * dz;
    }
    vx(i) += dt * ax;
    vy(i) += dt * ay;
    vz(i) += dt * az;
    x(i)  += dt * vx(i);
    y(i)  += dt * vy(i);
    z(i)  += dt * vz(i);
  });
}
