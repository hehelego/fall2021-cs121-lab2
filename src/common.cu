#include "common.cuh"

const CopyKind CopyKind::H2H = CopyKind(0);
const CopyKind CopyKind::H2D = CopyKind(1);
const CopyKind CopyKind::D2H = CopyKind(2);
const CopyKind CopyKind::D2D = CopyKind(3);

static const u32 Prime1 = 2654435761U;
static const u32 Prime2 = 2246822519U;
static const u32 Prime3 = 3266489917U;
static const u32 Prime4 = 668265263U;
static const u32 Prime5 = 374761393U;

__host__ __device__ u32 __rotateLeft(u32 v, u32 n) { return (v << n) | (v >> (32 - n)); }

__host__ __device__ u32 xxHash32(u32 seed, u32 value) {
  u32 state = seed + Prime5;
  state = state + value * Prime3;
  state = __rotateLeft(state, 17) * Prime4;

  // loop unrolling optimization
  {
    auto *text = (u8 *)&value;
    state = state + text[0] * Prime5;
    state = __rotateLeft(state, 11) * Prime1;
    state = state + text[1] * Prime5;
    state = __rotateLeft(state, 11) * Prime1;
    state = state + text[2] * Prime5;
    state = __rotateLeft(state, 11) * Prime1;
    state = state + text[3] * Prime5;
    state = __rotateLeft(state, 11) * Prime1;
  }

  state ^= state >> 15;
  state *= Prime2;
  state ^= state >> 13;
  state *= Prime3;
  state ^= state >> 16;

  return state;
}
