/*
    Egaroucid Project

    @date 2021-2022
    @author Takuto Yamana (a.k.a Nyanyan)
    @license GPL-3.0 license
*/

// use SIMD
#define USE_SIMD true

// vertical mirror
#define USE_FAST_VERTICAL_MIRROR true

// parity ordering
#define USE_END_PO true

// stability cut
#define USE_END_SC true

// prob cut
#define USE_MID_MPC true
#define USE_END_MPC true

// pop_count
#define USE_BUILTIN_POPCOUNT true

// NTZ
#define USE_BUILTIN_NTZ true
#define USE_MINUS_NTZ false

// last parity ordering optimisation
#define LAST_PO_OPTIMISE true

// NWS best move transposition table
#define USE_NWS_CHILD_TT false