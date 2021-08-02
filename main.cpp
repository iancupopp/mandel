#include <SDL2/SDL.h>
#include <iostream>
#include <deque>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define CALCULATED (1 << 0)
#define SCANNED (1 << 1)

const int kWidth = 800;
const int kHeight = 800;
#ifdef _OPENMP
const int kChunks = omp_get_max_threads();
#else
const int kChunks = 1;
#endif
const int kChunkSize = kHeight / kChunks;
const int kDefaultIt = 100;

int pixel_mask[kWidth * kHeight], max_it = kDefaultIt;
unsigned *pixel_color;
double xx0 = -2, yy0 = 2, xx1 = 2, yy1 = 2 - 4 * kHeight / (double)kWidth;
double x_offset = (xx1 - xx0) / kWidth, y_offset = (yy0 - yy1) / kHeight;
bool running = true;

struct Vec2d {
  double x, y;
};

unsigned ColorFormula(int it) {
  unsigned c = 0;
  double t = it / (double)max_it;
  c += unsigned(9 * (1 - t) * t * t * t * 255) << 16;
  c += unsigned(15 * (1 - t) * (1 - t) * t * t * 255) << 8;
  c += unsigned(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
  c += unsigned(SDL_ALPHA_OPAQUE) << 24;
  return c;
}

void OneMandel(const int kPixel) {
  double cr, ci, zr, zi, zr2, zi2;
  int iterations;
  cr = xx0 + (kPixel % kWidth) * x_offset;
  ci = yy0 - (kPixel / kHeight) * y_offset;
  zr = zi = zr2 = zi2 = iterations = 0;
  while (zr2 + zi2 <= 4 && iterations < max_it) {
    zi = 2 * zi * zr + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr;
    zi2 = zi * zi;
    ++iterations;
  }
  pixel_color[kPixel] = ColorFormula(iterations);
}

void FourMandel(const int kPixels[4], int sz) {
  __m256d _x, _y, _x_offset, _y_offset, _xx0, _yy0,
  _zr, _zi, _cr, _ci, _a, _b, _zr2, _zi2, _two, _four, _mask1;
  __m256i _n, _iterations, _mask2;

  _x = _mm256_set_pd(kPixels[0] % kWidth, kPixels[1] % kWidth,
                     kPixels[2] % kWidth, kPixels[3] % kWidth);
  _y = _mm256_set_pd(kPixels[0] / kWidth, kPixels[1] / kWidth,
                         kPixels[2] / kWidth, kPixels[3] / kWidth);

  _xx0 = _mm256_set1_pd(xx0);
  _yy0 = _mm256_set1_pd(yy0);
  _x_offset = _mm256_set1_pd(x_offset);
  _y_offset = _mm256_set1_pd(-y_offset);

  _zr = _mm256_setzero_pd();
  _zi = _mm256_setzero_pd();
  _cr = _mm256_fmadd_pd(_x, _x_offset, _xx0);
  _ci = _mm256_fmadd_pd(_y, _y_offset, _yy0);

  _iterations = _mm256_set1_epi64x(max_it);
  _n = _mm256_set_epi64x(0, sz >= 2 ? 0 : 1e9, sz >= 3 ? 0 : 1e9, sz == 4 ? 0 : 1e9);
  _two = _mm256_set1_pd(2.0);
  _four = _mm256_set1_pd(4.0);

  repeat:

  _zr2 = _mm256_mul_pd(_zr, _zr);
  _zi2 = _mm256_mul_pd(_zi, _zi);
  _a = _mm256_sub_pd(_zr2, _zi2);
  _a = _mm256_add_pd(_a, _cr);
  _b = _mm256_mul_pd(_zr, _zi);
  _b = _mm256_fmadd_pd(_b, _two, _ci);

  _zr = _a;
  _zi = _b;

  _a = _mm256_add_pd(_zr2, _zi2);

  _mask1 = _mm256_cmp_pd(_a, _four, _CMP_LT_OQ);
  _mask2 = _mm256_cmpgt_epi64(_iterations, _n);
  _mask2 = _mm256_and_si256(_mm256_castpd_si256(_mask1), _mask2);
  _mask2 = _mm256_and_si256(_mm256_set1_epi64x(1LL), _mask2);

  _n = _mm256_add_epi64(_mask2, _n);

  if (_mm256_testz_si256(_mask2, _mask2) != 1) goto repeat;

#if defined(_WIN32)
  for (int i = 0; i < sz; ++i)
    pixel_color[kPixels[i]] = ColorFormula(int(_n.m256i_i64[3 - i]));
#endif

#if defined(__linux__)
  for (int i = 0; i < sz; ++i)
    pixel_color[kPixels[i]] = ColorFormula(int(_n[3 - i]));
#endif
}

class BorderTracer {
private:
  std::deque<int> scan_queue;
  std::deque<int> calc_queue;
  int first_row, last_row;
public:
  BorderTracer(int fr, int lr) {
    first_row = fr;
    last_row = lr;
    Run();
  }

  void ClearMask() const {
    for (int i = first_row * kWidth; i < (last_row + 1) * kWidth; ++i)
      pixel_mask[i] = 0;
  }

  void Run() {
    ClearMask();
    AddEdges();
    while (!scan_queue.empty()) {
      auto pixel = scan_queue.front();
      scan_queue.pop_front();

      int up = pixel - kWidth, down = pixel + kWidth, left = pixel - 1, right = pixel + 1;
      bool up_ok = (up >= first_row * kWidth), down_ok = (down < (last_row + 1) * kWidth),
          left_ok = (left / kWidth == pixel / kWidth), right_ok = (right / kWidth == pixel / kWidth);

      if (up_ok) AddCalcQueue(up);
      if (down_ok) AddCalcQueue(down);
      if (left_ok) AddCalcQueue(left);
      if (right_ok) AddCalcQueue(right);
      FlushCalcQueue();

      bool up_diff = up_ok && (pixel_color[up] != pixel_color[pixel]);
      bool down_diff = down_ok && (pixel_color[down] != pixel_color[pixel]);
      bool left_diff = left_ok && (pixel_color[left] != pixel_color[pixel]);
      bool right_diff = right_ok && (pixel_color[right] != pixel_color[pixel]);

      if (up_diff) AddScanQueue(up);
      if (down_diff) AddScanQueue(down);
      if (left_diff) AddScanQueue(left);
      if (right_diff) AddScanQueue(right);

      if (up_ok && left_ok && (up_diff || left_diff)) AddScanQueue(up - 1);
      if (up_ok && right_ok && (up_diff || right_diff)) AddScanQueue(up + 1);
      if (down_ok && left_ok && (down_diff || left_diff)) AddScanQueue(down - 1);
      if (down_ok && right_ok && (down_diff || right_diff)) AddScanQueue(down + 1);
    }
   FillRest();
  }

  void FillRest() const {
    for (int i = (first_row + 1) * kWidth + 1; i < last_row * kWidth; ++i)
      if (!(pixel_mask[i] & CALCULATED))
        pixel_color[i] = pixel_color[i - 1];
  }

  void AddEdges() {
    for (int i = first_row; i <= last_row; ++i) {
      AddScanQueue(i * kWidth);
      AddScanQueue(i * (kWidth + 1) - 1);
    }
    for (int j = 1; j < kWidth - 1; ++j) {
      AddScanQueue(first_row * kWidth + j);
      AddScanQueue(last_row * kWidth + j);
    }
    FlushCalcQueue();
  }

  void AddScanQueue(int pixel) {
    if (pixel_mask[pixel] & SCANNED)
      return;
    pixel_mask[pixel] |= SCANNED;
    scan_queue.push_back(pixel);
    AddCalcQueue(pixel);
  }

  void AddCalcQueue(int pixel) {
    if (pixel_mask[pixel] & CALCULATED)
      return;
    pixel_mask[pixel] |= CALCULATED;
#ifdef __AVX2__
    calc_queue.push_back(pixel);
#else
    OneMandel(pixel);
#endif
  }

  void FlushCalcQueue() {
#ifdef __AVX2__
    int points[4] = {0, 0, 0, 0}, i;
    while (!calc_queue.empty()) {
      for (i = 0; i < 4 && !calc_queue.empty(); ++i) {
        points[i] = calc_queue.back();
        calc_queue.pop_back();
      }
      FourMandel(points, i);
    }
  }
#endif
};

void HandleInput(SDL_Event event) {
  switch (event.type) {
    case SDL_QUIT: running = false; break;
    case SDL_MOUSEWHEEL: {
      double x_dist = (xx1 - xx0) * 0.1;
      double y_dist = (yy0 - yy1) * 0.1;
      int mouse_x, mouse_y;
      SDL_GetMouseState(&mouse_x, &mouse_y);
      double left = (double)mouse_x / kWidth;
      double up = (double)mouse_y / kHeight;
      if (event.wheel.y > 0) {
        xx0 += left * x_dist;
        xx1 -= (1 - left) * x_dist;
        yy0 -= up * y_dist;
        yy1 += (1 - up) * y_dist;
      }
      else {
        xx0 -= left * x_dist;
        xx1 += (1 - left) * x_dist;
        yy0 += up * y_dist;
        yy1 -= (1 - up) * y_dist;
      }
      x_offset = (xx1 - xx0) / kWidth;
      y_offset = (yy0 - yy1) / kHeight;
      break;
    }
    case SDL_KEYDOWN: {
      switch (event.key.keysym.sym) {
        case SDLK_UP:  max_it += 50; break;
        case SDLK_DOWN: if (max_it > 100) max_it -= 50; break;
        default: break;
      }
    }
    default: break;
  }
}

int main(int argc, char* argv[]) {
  SDL_Window* window = nullptr;
  SDL_Renderer* renderer = nullptr;
  SDL_Texture* texture = nullptr;

  if (SDL_Init(SDL_INIT_VIDEO) < 0)
    std::cout << "SDL could not initialize! SDL_Error: " << SDL_GetError() << "\n";
  else {
    window = SDL_CreateWindow("Mandelbrot by Iancu (Border tracing + Multithreading + AVX2)", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, kWidth, kHeight,
                              SDL_WINDOW_SHOWN);
    if (!window)
      std::cout << "Window could not be created! SDL_Error: " << SDL_GetError() << "\n";
    else {
      renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
      texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, kWidth, kHeight);

      SDL_Event event;
      while (running) {
        while (SDL_PollEvent(&event))
          HandleInput(event);

        SDL_RenderClear(renderer);

        int pitch;
        SDL_LockTexture(texture, nullptr, (void**)(&pixel_color), &pitch);
#pragma omp parallel for
        for (int i = 0; i < kChunks; ++i)
          BorderTracer bt(i * kChunkSize, (i + 1) * kChunkSize - 1);
        SDL_UnlockTexture(texture);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
      }
    }
  }

  SDL_DestroyTexture(texture);
  SDL_DestroyRenderer(renderer);
  SDL_DestroyWindow(window);
  SDL_Quit();
  return 0;
}
