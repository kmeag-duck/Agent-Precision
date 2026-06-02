// Symplectic-Euler integration of an ensemble of 2D Kepler orbits around
// a central unit mass (G = M = 1). Each thread advances one particle:
//   v -= dt * r / |r|^3
//   r += dt * v
//
// Verdict: MUST STAY double for any meaningful integration time.
// Why: roundoff accumulates linearly in the step count for a symplectic
// integrator (and quadratically in energy drift for non-symplectic ones).
// At dt = 1e-3 over 1e6 steps (~160 orbits for r0 = 1), float trajectories
// visibly spiral or open up; double remains closed to within plotting
// precision over the same window.
// Suggested test: initial r = (1, 0), v = (0, 1), N = 1<<14 particles
// (vary initial conditions slightly per index), dt = 1e-3, n_steps = 1e6.
// Driver loop on host calls this kernel n_steps times.
// Compare final radius to double reference; require rtol = 1e-3.

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
