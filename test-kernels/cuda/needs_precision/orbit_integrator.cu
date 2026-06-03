// Symplectic-Euler integration of an ensemble of 2D Kepler orbits around
// a central unit mass (G = M = 1). Each thread advances one particle one step:
//   v -= dt * r / |r|^3
//   r += dt * v

#include <math.h>

__global__ void leapfrog_step(double* __restrict__ x,
                              double* __restrict__ y,
                              double* __restrict__ vx,
                              double* __restrict__ vy,
                              int N, double dt) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  const double r2     = x[i] * x[i] + y[i] * y[i];
  const double inv_r3 = 1.0 / (r2 * sqrt(r2));

  vx[i] -= dt * x[i] * inv_r3;
  vy[i] -= dt * y[i] * inv_r3;
  x[i]  += dt * vx[i];
  y[i]  += dt * vy[i];
}
