
// use CUDA
#define USE_CUDA false

// use SIMD
#define USE_SIMD true

// vertical mirror
#define USE_FAST_VERTICAL_MIRROR true

// board to array conversion
#define FAST_ARR_CONVERT true

// flip calculating
#define FLIP_CALC_MODE 3
#define LAST_FLIP_CALC_MODE 4

// transpose table cut
#define USE_MID_TC true
#define USE_END_TC true

// parity ordering
#define USE_END_PO true

// stability cut
#define USE_MID_SC false
#define USE_END_SC true

// parallel search
#define USE_MULTI_THREAD true

// prob cut
#define USE_MID_MPC true
#define USE_END_MPC true

// legal calculation
#define LEGAL_CALCULATION_MODE 5

// pop_count
#define USE_BUILTIN_POPCOUNT true

// NTZ
#define USE_MINUS_NTZ false

// evaluation calculation
#define USE_FAST_DIFF_EVAL false

// evaluation function step width
// 0: 1 discs
// 1: 2 discs
// 2: 1/2 discs
// 3: 1/4 discs
// 4: 1/8 discs
// 5: 1/16 discs
// 6: 1/32 discs
#define EVALUATION_STEP_WIDTH_MODE 0