// Direct-summation gravitational N-body: per-particle force from all others,
// then symplectic-Euler kick + drift. Source is all-double; the table below
// is what a correct per-variable rewriter should land on.
//
// Per-variable verdicts:
//   x, y, z          (positions)             MUST stay double. Pairs of distant
//                                            bodies subtract positions of
//                                            magnitude ~box_size to produce
//                                            displacements of magnitude
//                                            ~interparticle spacing —
//                                            catastrophic cancellation in float.
//   m                (mass)                   FLOAT OK — bounded, no accumulation.
//   vx, vy, vz       (velocities)             FLOAT OK for short integrations;
//                                            stay double for long runs. The
//                                            position update dt*v is still
//                                            performed in double regardless.
//   dx, dy, dz       (pairwise deltas)        FLOAT OK — these *are* the small
//                                            differences; their dynamic range
//                                            is set by interparticle spacing.
//   r2, inv_r, inv_r3, s   (derived)          FLOAT OK once eps > 0 regularizes
//                                            the singularity.
//   ax, ay, az       (per-i force accumulator) MUST stay double. The net force
//                                            is the small residual of large,
//                                            opposing per-neighbor pulls;
//                                            float accumulator loses the
//                                            cancellation entirely.
//   G, eps, eps2, dt (scalar parameters)      FLOAT OK.
//
// Verdict: MIXED.
// Suggested test: Plummer sphere, N = 1<<14, eps = 1e-3, dt = 1e-3,
// n_steps = 1000. Compare per-particle radius and velocity magnitude to an
// all-double reference; require rtol = 1e-4.

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
