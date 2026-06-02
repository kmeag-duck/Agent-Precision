// 7-point stencil heat-equation update:
//   u_new(i,j,k) = u(i,j,k) + alpha*dt/h^2 *
//     ( u(i+1,j,k)+u(i-1,j,k) + u(i,j+1,k)+u(i,j-1,k)
//       + u(i,j,k+1)+u(i,j,k-1) - 6*u(i,j,k) )
//
// This is the canonical "compute increment in low precision, accumulate in
// high precision" pattern that pays off in long-time PDE integration.
//
// Per-variable verdicts (assumes n_steps >= 1e5):
//   u, u_new   (field arrays)                MUST stay double. Long-time
//                                            integration accumulates per-step
//                                            roundoff into the field; storing
//                                            u in float loses ~ulp(float)
//                                            *per step*, so a 1e5-step run
//                                            shifts the field by ~1e-2
//                                            relative — useless for any
//                                            convergence study.
//   center, neighbors[6]   (local reads)     Read as double from memory; the
//                                            subsequent local arithmetic
//                                            (subtraction of neighboring
//                                            values) is bounded by the
//                                            stencil stride, so the laplacian
//                                            CAN be computed in float without
//                                            material loss.
//   lap         (sum of neighbor terms - 6*center)
//                                            FLOAT OK — bounded magnitude,
//                                            this is the small "differential"
//                                            quantity by design.
//   delta       (alpha*dt/h^2 * lap)         FLOAT OK — small per-step nudge.
//   alpha_dt_over_h2 (scalar coefficient)    FLOAT OK.
//   u_new[idx] = center + delta              addition MUST be performed in
//                                            double (center is double; delta
//                                            promotes) to preserve field bits.
//
// Verdict: MIXED. This kernel exemplifies the typical physics pattern: store
// state in double, compute its time-derivative in float, add increments in
// double.
// Suggested test: 128^3 grid, Gaussian initial bump centered in the box,
// alpha*dt/h^2 = 0.1, n_steps = 1e5. Compare max-norm and L2-norm of u
// against an all-double reference; require rtol = 1e-5.

__global__ void heat_step(
    const double* __restrict__ u,
    double* __restrict__ u_new,
    int Nx, int Ny, int Nz,
    double alpha_dt_over_h2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int k = blockIdx.z * blockDim.z + threadIdx.z;
  if (i <= 0 || i >= Nx - 1) return;
  if (j <= 0 || j >= Ny - 1) return;
  if (k <= 0 || k >= Nz - 1) return;

  const int idx     = (k * Ny + j) * Nx + i;
  const int stride_y = Nx;
  const int stride_z = Nx * Ny;

  const double center = u[idx];
  const double lap =
      u[idx + 1]        + u[idx - 1]
    + u[idx + stride_y] + u[idx - stride_y]
    + u[idx + stride_z] + u[idx - stride_z]
    - 6.0 * center;

  const double delta = alpha_dt_over_h2 * lap;
  u_new[idx] = center + delta;
}
