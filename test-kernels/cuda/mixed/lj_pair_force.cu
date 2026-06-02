// Lennard-Jones pair-force kernel. O(N^2) for simplicity; a real MD code would
// iterate a neighbor list, but the precision question is orthogonal to that.
//   F_i = sum_{j != i, r<rcut} 24*eps * (2*(sigma/r)^12 - (sigma/r)^6) / r^2 * (r_j - r_i)
//
// Per-variable verdicts:
//   x, y, z         (positions)               MUST stay double. Same argument
//                                             as N-body: pairwise differences
//                                             subtract nearby coordinates.
//   xi, yi, zi      (per-i local copies)      MUST stay double (it's a
//                                             position).
//   dx, dy, dz      (pairwise deltas)         FLOAT OK after the subtraction.
//   r2, inv_r2,
//   inv_r6, inv_r12, fmag (derived locals)    FLOAT OK — bounded after the
//                                             r < rcut filter; LJ force is
//                                             well-conditioned once short-
//                                             range overlaps are excluded.
//   fxi, fyi, fzi   (per-i force accumulator) MUST stay double. Net force on
//                                             an atom is the small residual
//                                             of many opposing neighbor pulls;
//                                             this is the cancellation-prone
//                                             sum.
//   fx_out, fy_out, fz_out (output force arrays)
//                                             MUST stay double (downstream
//                                             integration accumulates these
//                                             into velocity over many steps).
//   sigma, epsilon, rcut2, s2 (scalar params) FLOAT OK.
//
// Verdict: MIXED.
// Suggested test: 32^3 LJ fluid (N = 32768) at reduced density 0.8,
// rcut = 2.5 * sigma, configuration relaxed for 1000 NVE steps before
// measurement. Compare per-particle force vector to an all-double reference;
// require rtol = 1e-4.

#include <math.h>

__global__ void lj_pair_force(
    const double* __restrict__ x,
    const double* __restrict__ y,
    const double* __restrict__ z,
    double* __restrict__ fx_out,
    double* __restrict__ fy_out,
    double* __restrict__ fz_out,
    int N, double sigma, double epsilon, double rcut2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const double xi = x[i], yi = y[i], zi = z[i];
  double fxi = 0.0, fyi = 0.0, fzi = 0.0;
  const double s2 = sigma * sigma;

  for (int j = 0; j < N; ++j) {
    if (j == i) continue;
    const double dx = x[j] - xi;
    const double dy = y[j] - yi;
    const double dz = z[j] - zi;
    const double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 > rcut2 || r2 == 0.0) continue;

    const double inv_r2  = s2 / r2;
    const double inv_r6  = inv_r2 * inv_r2 * inv_r2;
    const double inv_r12 = inv_r6 * inv_r6;
    const double fmag    = 24.0 * epsilon * (2.0 * inv_r12 - inv_r6) / r2;

    fxi += fmag * dx;
    fyi += fmag * dy;
    fzi += fmag * dz;
  }

  fx_out[i] = fxi;
  fy_out[i] = fyi;
  fz_out[i] = fzi;
}
