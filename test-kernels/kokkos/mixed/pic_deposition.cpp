// Cloud-in-cell (CIC) charge deposition for a particle-in-cell code. Each
// particle deposits its charge into the 8 surrounding grid cells in 3D,
// weighted by trilinear factors derived from its fractional cell offset.
//
// Per-variable verdicts:
//   px, py, pz   (particle positions)          MUST stay double. Computing the
//                                              fractional offset requires
//                                              subtracting the cell base
//                                              position from the particle
//                                              position; float would lose
//                                              sub-cell resolution once the
//                                              grid origin is far from zero.
//   xp, yp, zp   (positions in cell units,
//                 = (p - origin) * dx_inv)     MUST stay double for the same
//                                              reason; the subtraction has
//                                              already happened by float would
//                                              still have lost bits if applied
//                                              earlier.
//   ix, iy, iz   (integer cell indices)        int.
//   fx, fy, fz   (fractional offsets in [0,1]) FLOAT OK — bounded by
//                                              construction.
//   wx[2], wy[2], wz[2]   (1D CIC weights)     FLOAT OK — in [0, 1].
//   w            (3D weight = wx*wy*wz*qp)     FLOAT OK.
//   qp           (per-particle charge)         FLOAT OK — bounded.
//   rho(i,j,k)   (grid charge density)         MUST stay double. Many particles
//                                              deposit into the same cell —
//                                              this is the classic naive-sum
//                                              accumulator problem, made worse
//                                              by atomic contention masking
//                                              ordering.
//   dx_inv, origin_{x,y,z}                     dx_inv FLOAT OK; the origin
//                                              MUST stay double (it is the
//                                              quantity being subtracted from
//                                              positions).
//
// Verdict: MIXED.
// Suggested test: 64^3 grid, 1e6 particles uniformly distributed, origin far
// from zero (e.g. (1e6, 1e6, 1e6)). Compare per-cell rho and the total
// integrated charge to an all-double reference; require rtol = 1e-5 per cell
// and rtol = 1e-10 on the total.

#include <Kokkos_Core.hpp>

void cic_deposit(
    Kokkos::View<const double*> px,
    Kokkos::View<const double*> py,
    Kokkos::View<const double*> pz,
    Kokkos::View<const double*> q,
    Kokkos::View<double***> rho,
    double dx_inv,
    double origin_x, double origin_y, double origin_z) {
  const int Np = px.extent(0);
  const int Nx = rho.extent(0);
  const int Ny = rho.extent(1);
  const int Nz = rho.extent(2);

  Kokkos::parallel_for("cic_deposit", Np, KOKKOS_LAMBDA(int p) {
    const double xp = (px(p) - origin_x) * dx_inv;
    const double yp = (py(p) - origin_y) * dx_inv;
    const double zp = (pz(p) - origin_z) * dx_inv;

    const int ix = (int)xp;
    const int iy = (int)yp;
    const int iz = (int)zp;
    if (ix < 0 || ix >= Nx - 1) return;
    if (iy < 0 || iy >= Ny - 1) return;
    if (iz < 0 || iz >= Nz - 1) return;

    const double fx = xp - (double)ix;
    const double fy = yp - (double)iy;
    const double fz = zp - (double)iz;

    const double wx[2] = {1.0 - fx, fx};
    const double wy[2] = {1.0 - fy, fy};
    const double wz[2] = {1.0 - fz, fz};
    const double qp = q(p);

    for (int kz = 0; kz < 2; ++kz)
      for (int ky = 0; ky < 2; ++ky)
        for (int kx = 0; kx < 2; ++kx) {
          const double w = wx[kx] * wy[ky] * wz[kz] * qp;
          Kokkos::atomic_add(&rho(ix + kx, iy + ky, iz + kz), w);
        }
  });
}
