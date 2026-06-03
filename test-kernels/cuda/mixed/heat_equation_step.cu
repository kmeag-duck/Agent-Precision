// 7-point stencil heat-equation update:
//   u_new(i,j,k) = u(i,j,k) + alpha*dt/h^2 *
//     ( u(i+1,j,k)+u(i-1,j,k) + u(i,j+1,k)+u(i,j-1,k)
//       + u(i,j,k+1)+u(i,j,k-1) - 6*u(i,j,k) )

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
