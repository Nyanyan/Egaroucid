/*
    Egaroucid Project

    @file last_flip_common.hpp
        Common tables for last flip calculation
    @date 2021-2026
    @author Takuto Yamana
    @author Toshihiko Okuhara
    @license GPL-3.0-or-later
*/

#pragma once
#include <array>
#include <cstdint>
#include "common.hpp"

constexpr uint64_t LAST_FLIP_INNER_BOX_MASK = 0x00003C3C3C3C0000ULL;

constexpr uint_fast8_t LAST_FLIP_DIAGONAL_LINE_MASK_T[15] = {
    0x01, 0x03, 0x07,
    0x0F, 0x1F, 0x3F,
    0x7F, 0xFF, 0xFE,
    0xFC, 0xF8, 0xF0,
    0xE0, 0xC0, 0x80
};

constexpr uint64_t LAST_FLIP_DIAG_MASK[HW2] = {
    0x8040201008040201ULL, 0x0080402010080403ULL, 0x0000804020110A00ULL, 0x0000008041221400ULL, 0x0000000182442800ULL, 0x0000010204885000ULL, 0x0001020408102041ULL, 0x0102040810204080ULL,
    0x4020100804020101ULL, 0x8040201008040201ULL, 0x00804020110A0000ULL, 0x0000804122140000ULL, 0x0000018244280000ULL, 0x0001020488500000ULL, 0x0102040810204080ULL, 0x0204081020408001ULL,
    0x2010080402010204ULL, 0x4020100804020408ULL, 0x0000000102000810ULL, 0x0000010204001020ULL, 0x0001020408002040ULL, 0x0102040810204080ULL, 0x0204081020402010ULL, 0x0408102040804020ULL,
    0x1008040201020408ULL, 0x2010080402040810ULL, 0x0000010200081020ULL, 0x0001020400102040ULL, 0x0102040810204080ULL, 0x0204081000408000ULL, 0x0408102040201008ULL, 0x0810204080402010ULL,
    0x0804020102040810ULL, 0x1008040204081020ULL, 0x0001020008102040ULL, 0x0102040810204080ULL, 0x0204080020408000ULL, 0x0408100040800000ULL, 0x0810204020100804ULL, 0x1020408040201008ULL,
    0x0402010204081020ULL, 0x0804020408102040ULL, 0x0102040810204080ULL, 0x0204001020408000ULL, 0x0408002040800000ULL, 0x0810004080000000ULL, 0x1020402010080402ULL, 0x2040804020100804ULL,
    0x0000020408102040ULL, 0x0102040810204080ULL, 0x00000A1120408000ULL, 0x0000142241800000ULL, 0x0000284482010000ULL, 0x0000508804020100ULL, 0x8040201008040201ULL, 0x0000402010080402ULL,
    0x0102040810204080ULL, 0x0004081020408000ULL, 0x000A112040800000ULL, 0x0014224180000000ULL, 0x0028448201000000ULL, 0x0050880402010000ULL, 0x0020100804020100ULL, 0x8040201008040201ULL
};

constexpr uint_fast16_t LAST_FLIP_DIAG_OFFSET[HW2] = {
    0, 256, 2048, 2176, 2304, 2432, 256, 0,
    256, 256, 2048, 2176, 2304, 2432, 256, 256,
    512, 512, 3328, 3264, 3072, 512, 512, 512,
    768, 768, 3296, 3136, 768, 3072, 768, 768,
    1024, 1024, 3200, 1024, 3136, 3264, 1024, 1024,
    1280, 1280, 1280, 3200, 3296, 3328, 1280, 1280,
    1536, 1536, 2560, 2688, 2816, 2944, 1536, 1536,
    1792, 1536, 2560, 2688, 2816, 2944, 1536, 1792
};

constexpr uint64_t LAST_FLIP_BOX_D7_MASK[28] = {
          0x8040201008040201ULL, 0x0080402010000402ULL, 0x0000804020000804ULL, 0x0000008040001008ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0x4020100800020100ULL, 0x8040201008040201ULL, 0x0080402000080402ULL, 0x0000804000100804ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0x2010080002010000ULL, 0x4020100004020100ULL, 0x8040201008040201ULL, 0x0080400010080402ULL, 0ULL, 0ULL,
    0ULL, 0ULL, 0x1008000201000000ULL, 0x2010000402010000ULL, 0x4020000804020100ULL, 0x8040201008040201ULL
};

constexpr uint_fast16_t LAST_FLIP_BOX_D7_OFFSET[28] = {
          512, 3072, 3264, 3328, 0, 0,
    0, 0, 3072, 768, 3136, 3296, 0, 0,
    0, 0, 3264, 3136, 1024, 3200, 0, 0,
    0, 0, 3328, 3296, 3200, 1280
};

constexpr int LAST_FLIP_DIAG_TABLE_SIZE = 3360;

constexpr uint_fast8_t last_flip_popcount_constexpr(uint64_t x) {
    uint_fast8_t res = 0;
    while (x) {
        x &= x - 1;
        ++res;
    }
    return res;
}

constexpr uint64_t last_flip_pdep_constexpr(uint_fast16_t x, uint64_t mask) {
    uint64_t res = 0;
    uint64_t bit = 1;
    while (mask) {
        const uint64_t lsb = mask & (~mask + 1);
        if (x & bit) {
            res |= lsb;
        }
        mask ^= lsb;
        bit <<= 1;
    }
    return res;
}

constexpr bool last_flip_is_on_board(int x, int y) {
    return 0 <= x && x < HW && 0 <= y && y < HW;
}

constexpr uint_fast8_t last_flip_count_ray(uint64_t player, int x, int y, int dx, int dy) {
    uint_fast8_t n = 0;
    x += dx;
    y += dy;
    while (last_flip_is_on_board(x, y)) {
        if (player & (1ULL << (y * HW + x))) {
            return n;
        }
        ++n;
        x += dx;
        y += dy;
    }
    return 0;
}

constexpr uint_fast8_t last_flip_count_diag(uint64_t player, uint_fast8_t place) {
    const int x = place & 7;
    const int y = place >> 3;
    return last_flip_count_ray(player, x, y, 1, 1) +
           last_flip_count_ray(player, x, y, -1, -1) +
           last_flip_count_ray(player, x, y, 1, -1) +
           last_flip_count_ray(player, x, y, -1, 1);
}

constexpr uint16_t last_flip_make_diag_entry(uint_fast8_t place, uint64_t mask, uint_fast16_t pattern) {
    const uint64_t place_bit = 1ULL << place;
    const uint64_t player = last_flip_pdep_constexpr(pattern, mask) & ~place_bit;
    const uint64_t opponent = (~player) & mask & ~place_bit;
    return (uint16_t)((last_flip_count_diag(opponent, place) << 8) | last_flip_count_diag(player, place));
}

constexpr bool last_flip_diag_table_is_big_enough() {
    for (uint_fast8_t place = 0; place < HW2; ++place) {
        const uint_fast16_t n_patterns = 1U << last_flip_popcount_constexpr(LAST_FLIP_DIAG_MASK[place]);
        if (LAST_FLIP_DIAG_OFFSET[place] + n_patterns > LAST_FLIP_DIAG_TABLE_SIZE) {
            return false;
        }
    }
    for (uint_fast8_t i = 0; i < 28; ++i) {
        if (LAST_FLIP_BOX_D7_MASK[i]) {
            const uint_fast16_t n_patterns = 1U << last_flip_popcount_constexpr(LAST_FLIP_BOX_D7_MASK[i]);
            if (LAST_FLIP_BOX_D7_OFFSET[i] + n_patterns > LAST_FLIP_DIAG_TABLE_SIZE) {
                return false;
            }
        }
    }
    return true;
}

static_assert(last_flip_diag_table_is_big_enough());

inline std::array<uint16_t, LAST_FLIP_DIAG_TABLE_SIZE> N_LAST_FLIP_DIAG_BOTH{};

inline void last_flip_common_init() {
    static bool initialized = false;
    if (initialized) {
        return;
    }
    initialized = true;
    for (uint_fast8_t place = 0; place < HW2; ++place) {
        const uint64_t mask = LAST_FLIP_DIAG_MASK[place];
        const uint_fast16_t offset = LAST_FLIP_DIAG_OFFSET[place];
        const uint_fast16_t n_patterns = 1U << last_flip_popcount_constexpr(mask);
        for (uint_fast16_t pattern = 0; pattern < n_patterns; ++pattern) {
            N_LAST_FLIP_DIAG_BOTH[offset + pattern] = last_flip_make_diag_entry(place, mask, pattern);
        }
    }
    for (uint_fast8_t i = 0; i < 28; ++i) {
        const uint64_t mask = LAST_FLIP_BOX_D7_MASK[i];
        if (mask) {
            const uint_fast8_t place = i + 18;
            const uint_fast16_t offset = LAST_FLIP_BOX_D7_OFFSET[i];
            const uint_fast16_t n_patterns = 1U << last_flip_popcount_constexpr(mask);
            for (uint_fast16_t pattern = 0; pattern < n_patterns; ++pattern) {
                N_LAST_FLIP_DIAG_BOTH[offset + pattern] = last_flip_make_diag_entry(place, mask, pattern);
            }
        }
    }
}

inline uint_fast16_t last_flip_pext(uint64_t x, uint64_t mask) {
#if USE_BIT_GATHER_OPTIMIZE
    return (uint_fast16_t)_pext_u64(x, mask);
#else
    uint_fast16_t res = 0;
    uint_fast16_t bit = 1;
    while (mask) {
        const uint64_t lsb = mask & (~mask + 1);
        if (x & lsb) {
            res |= bit;
        }
        mask ^= lsb;
        bit <<= 1;
    }
    return res;
#endif
}

inline uint_fast16_t count_last_flip_diag_both(uint64_t player, const uint_fast8_t place) {
    const uint64_t place_bit = 1ULL << place;
    uint_fast16_t res = N_LAST_FLIP_DIAG_BOTH[LAST_FLIP_DIAG_OFFSET[place] + last_flip_pext(player, LAST_FLIP_DIAG_MASK[place])];
    if (LAST_FLIP_INNER_BOX_MASK & place_bit) {
        const uint_fast8_t box_idx = place - 18;
        res += N_LAST_FLIP_DIAG_BOTH[LAST_FLIP_BOX_D7_OFFSET[box_idx] + last_flip_pext(player, LAST_FLIP_BOX_D7_MASK[box_idx])];
    }
    return res;
}
