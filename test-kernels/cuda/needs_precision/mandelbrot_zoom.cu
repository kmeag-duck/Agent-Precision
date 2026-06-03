// Mandelbrot escape-time iteration at a configurable zoom.
//   z_{n+1} = z_n^2 + c,   escape if |z|^2 > 4,   up to max_iter steps.

__global__ void mandelbrot(int* __restrict__ iters,
                           int W, int H,
                           double cx, double cy, double scale,
                           int max_iter) {
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= W || py >= H) return;

  const double cr = cx + (px - W * 0.5) * scale;
  const double ci = cy + (py - H * 0.5) * scale;

  double zr = 0.0, zi = 0.0;
  int n = 0;
  while (n < max_iter && zr * zr + zi * zi < 4.0) {
    const double zr_new = zr * zr - zi * zi + cr;
    zi = 2.0 * zr * zi + ci;
    zr = zr_new;
    ++n;
  }
  iters[py * W + px] = n;
}
