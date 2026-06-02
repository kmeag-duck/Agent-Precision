# Test Kernel Verdicts — Quick Reference

Ground-truth labels for the 17 test kernels under `test-kernels/`. Source is
always all-`double` (or `long double`); the verdict is what a correct
precision-lowering rewriter should produce, and the tolerance is the bar an
output should clear when compared to an all-double reference on the suggested
inputs.

The directory split (`lowerable/`, `needs_precision/`, `mixed/`) is the
ground-truth label — the workflow must decide from file content, not paths.

---

## Lowerable — float is acceptable for the whole kernel

| File | Computes | Test inputs | Tolerance |
|---|---|---|---|
| `kokkos/lowerable/vector_add.cpp`     | `z = x + y` (pointwise)                  | `x, y ~ U(-1, 1)`, `N = 1<<20`     | `rtol = 1e-6` |
| `kokkos/lowerable/saxpy_bounded.cpp`  | `y = a*x + y`, bounded `a`, `x`, `y`     | `a = 2.5`, `x, y ~ U(-1, 1)`, `N = 1<<20` | `rtol = 1e-6` |
| `kokkos/lowerable/relu_activation.cpp`| `y = max(0, x)`                           | `x ~ N(0, 1)`, `N = 1<<20`         | exact (no arithmetic loss) |
| `cuda/lowerable/vector_add.cu`        | `z = x + y` (pointwise)                  | `x, y ~ U(-1, 1)`, `N = 1<<20`     | `rtol = 1e-6` |
| `cuda/lowerable/saxpy.cu`             | `y = a*x + y`, bounded                    | `a = 2.5`, `x, y ~ U(-1, 1)`, `N = 1<<20` | `rtol = 1e-6` |
| `cuda/lowerable/sigmoid.cu`           | `y = 1 / (1 + exp(-x))`                   | `x ~ U(-10, 10)`, `N = 1<<20`      | `rtol = 1e-5` |

Common pattern: one op per element, no inter-element accumulation, output
magnitudes bounded. Per-element relative error is bounded by `ulp(float)` ≈
1.2e-7 regardless of `N`.

---

## Needs precision — must stay double (or long double)

| File | Computes | Failure mode in float | Test inputs | Tolerance |
|---|---|---|---|---|
| `kokkos/needs_precision/naive_sum_reduce.cpp`     | `sum(x)` with `x(i) = 1/sqrt(i+1)`         | per-thread accumulator saturates; tail terms below `ulp` of partial sum | `N = 1<<23`                           | `rtol = 1e-4` vs Kahan ref |
| `kokkos/needs_precision/two_pass_variance.cpp`    | `var = E[x²] - E[x]²`                      | catastrophic cancellation when mean ≫ spread | `x ~ U(1e6, 1e6 + 1)`, `N = 1<<20`   | absolute err < 0.01 vs Welford ref |
| `kokkos/needs_precision/euler_oscillator.cpp`     | Forward-Euler SHO, many steps              | per-step roundoff accumulates → phase/amplitude drift | `dt = 1e-4`, `n_steps = 1e6`, `N = 1<<14` | per-particle abs err < 1e-3 |
| `kokkos/needs_precision/chebyshev_long_double.cpp`| `T_n(x)` recurrence (host, long double)    | recurrence amplifies roundoff exponentially | `x = 1 - 1e-3*U(0,1)`, `n = 10000`, `N = 1<<16` | `rtol = 1e-3` vs long-double ref |
| `cuda/needs_precision/harmonic_sum.cu`            | `sum 1/i` via per-thread chunks            | accumulator scale grows; tail terms vanish | `N = 1<<27`, `per_thread = 1024`     | `rtol = 1e-5` vs Kahan ref |
| `cuda/needs_precision/mandelbrot_zoom.cu`         | Mandelbrot escape-time at deep zoom        | pixel coords below `ulp(float)` of center → banding | center `(-0.7436..., 0.1318...)`, `scale = 1e-7`, `512×512` | ≥95% of pixels match exactly |
| `cuda/needs_precision/orbit_integrator.cu`        | Symplectic-Euler Kepler orbit              | trajectory roundoff → orbit opens or spirals | `r = (1, 0)`, `v = (0, 1)`, `dt = 1e-3`, `n_steps = 1e6` | `rtol = 1e-3` on final radius |

Common patterns: serial accumulation of many small terms; catastrophic
cancellation of nearby values; long-trajectory roundoff accumulation;
coordinate-resolution requirements below `ulp(float)`.

---

## Mixed — some variables can be lowered, others can't

Each kernel is written in all-double. The rewriter should produce the
per-variable mix below. Anything not listed in "stays double" is fair game
for float.

### `kokkos/mixed/nbody_force.cpp` — Direct-summation gravitational N-body

| Variable | Verdict |
|---|---|
| `x, y, z` (positions)             | **double** — pairwise differences subtract nearby coords |
| `ax, ay, az` (force accumulator)  | **double** — net force = small residual of opposing pulls |
| `vx, vy, vz` (velocities)         | borderline — float for short runs, double for long; context-dependent |
| `m` (mass)                        | float |
| `dx, dy, dz` (pairwise deltas)    | float — dynamic range is interparticle spacing |
| `r2, inv_r, inv_r3, s`            | float — bounded once `eps > 0` |
| `G, eps, eps2, dt`                | float |

**Test:** Plummer sphere, `N = 1<<14`, `eps = 1e-3`, `dt = 1e-3`, `n_steps = 1000`. `rtol = 1e-4` on per-particle radius and velocity magnitude.

### `kokkos/mixed/pic_deposition.cpp` — Cloud-in-cell charge deposition

| Variable | Verdict |
|---|---|
| `px, py, pz` (particle positions) | **double** — origin subtraction needs sub-cell resolution |
| `xp, yp, zp` (cell-unit positions)| **double** — already subtracted, but float would have lost bits |
| `origin_{x,y,z}`                  | **double** — it is the subtracted quantity |
| `rho(i,j,k)` (grid density)       | **double** — many-particle accumulator |
| `fx, fy, fz` (fractional offsets) | float — bounded in `[0, 1]` |
| `wx[2], wy[2], wz[2], w`          | float — bounded in `[0, 1]` |
| `qp` (per-particle charge)        | float |
| `dx_inv`                          | float |
| `ix, iy, iz`                      | int |

**Test:** 64³ grid, 1e6 particles uniformly distributed, origin at `(1e6, 1e6, 1e6)`. Per-cell `rtol = 1e-5`, total integrated charge `rtol = 1e-10`.

### `cuda/mixed/lj_pair_force.cu` — Lennard-Jones pair force

| Variable | Verdict |
|---|---|
| `x, y, z, xi, yi, zi` (positions) | **double** — pairwise difference argument |
| `fxi, fyi, fzi` (per-i accumulator) | **double** — sum of opposing neighbor forces |
| `fx_out, fy_out, fz_out`          | **double** — feeds downstream integration |
| `dx, dy, dz`                      | float — bounded after subtraction |
| `r2, inv_r2, inv_r6, inv_r12, fmag` | float — bounded by `r < rcut` filter |
| `sigma, epsilon, rcut2, s2`       | float |

**Test:** 32³ LJ fluid (`N = 32768`) at reduced density 0.8, `rcut = 2.5σ`, after 1000 NVE relaxation steps. `rtol = 1e-4` on per-particle force.

### `cuda/mixed/heat_equation_step.cu` — 7-point diffusion stencil

| Variable | Verdict |
|---|---|
| `u, u_new` (field arrays)          | **double** — per-step roundoff accumulates over long integration |
| `center + delta` (write-back)      | **double** — addition must preserve field bits |
| `center, neighbors` (local reads)  | read as double; local arithmetic can use float |
| `lap` (laplacian sum)              | float — bounded "differential" quantity |
| `delta = alpha_dt_over_h2 * lap`   | float — small per-step nudge |
| `alpha_dt_over_h2`                 | float |

**Test:** 128³ grid, Gaussian initial bump, `α·dt/h² = 0.1`, `n_steps = 1e5`. `rtol = 1e-5` on max-norm and L2-norm of `u`.

---

## The recurring physics pattern

Three of the four mixed kernels are instances of **subtract → compute → accumulate**:

1. **Subtract** nearby large coordinates → small deltas. The originals must be double; the deltas inherit a collapsed dynamic range and tolerate float.
2. **Compute** in float on those deltas — squared distances, weights, force magnitudes, laplacians. Bounded local quantities.
3. **Accumulate** back into a high-precision quantity (per-particle force, grid density, time-integrated field). The accumulator must be double; the per-iteration increment can promote on the way in.

A workflow that learns to identify these three roles in a kernel covers most
of the physics-code surface area.
