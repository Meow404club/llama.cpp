// meow: GQA6 single-pass packing for D=256 (ratio exactly 6, e.g. Qwen3.5 hybrid H24/H4).
// All 6 query heads of a KV group are processed in one pass over the KV cache,
// tripling arithmetic intensity per KV byte (FA is KV-bandwidth bound at long context).

#include "../fattn-mma-f16.cuh"

DECL_FATTN_MMA_F16_CASE(256, 256, 16, 6);
