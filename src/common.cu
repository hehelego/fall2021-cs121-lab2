#include "common.cuh"
#include <random>

u32 getRandSeed() {
  std::random_device rand_dev;
  return rand_dev();
}

void batchRandomGen(u32 *d_array, u32 n) {
  curandGenerator_t rng;
  CURAND_CALL(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MT19937));
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(rng, getRandSeed()));
  CURAND_CALL(curandGenerate(rng, d_array, n));
}
