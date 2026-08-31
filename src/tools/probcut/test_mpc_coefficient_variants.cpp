/* Regression checks for compile-time MPC coefficient candidates. */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "../../engine/engine_all.hpp"

namespace {

void require(bool condition, const std::string &message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_near(double actual, double expected, const std::string &message) {
    if (std::abs(actual - expected) > 1.0e-14) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

} // namespace

int main() {
#if USE_MPC_PRE_CALCULATION
    mpc_init();
#endif

    constexpr double mid_coefficients[6][7] = {
        {
            0.8335834703936896, -4.71778909968251,
            1.1467905781538477, -0.5274699259330169,
            6.5091001393587335, 3.9546352081550378,
            1.5719077939546169 + MPC_PROBCUT_G_OFFSET
        },
        {
            0.82401064177795602, -4.707256854709466,
            1.195936197029992, -0.51602485095990669,
            6.2720872514018389, 4.1311395597893092,
            1.8452655938095068
        },
        {
            0.83654221440046617, -4.7141339417062573,
            1.1595957590980255, -0.52741323914425797,
            6.4488921340359813, 3.9629396156378216,
            1.8652402791855631
        },
        {
            0.83953848974506118, -4.7144109706221009,
            1.1562994186860498, -0.5266829769832837,
            6.4194366178442852, 3.9616392064435049,
            1.8537318252529329
        },
        {
            0.81802231199785347, -4.7208452849479272,
            1.1454184012830171, -0.52880205200077901,
            6.4682004696076323, 3.9772925548531921,
            1.9224434471584249
        },
        {
            0.82597593080465503, -4.71706030657485,
            1.1552618089448403, -0.52298829335719277,
            6.458107516892567, 3.9807918334215708,
            1.8945888345974715
        }
    };
    constexpr int mid_offsets[] = {0, -4, -2, 0, 2, 4};
    constexpr double actual_mid_coefficients[] = {
        probcut_a, probcut_b, probcut_c, probcut_d,
        probcut_e, probcut_f, probcut_g
    };
    for (int i = 0; i < 7; ++i) {
        require_near(
            actual_mid_coefficients[i],
            mid_coefficients[MID_MPC_RECALIBRATED_VARIANT][i],
            "mid coefficient " + std::to_string(i)
        );
    }

    for (int deep_depth = USE_MPC_MIN_DEPTH; deep_depth <= 60; ++deep_depth) {
        int expected_shallow =
            ((deep_depth * MPC_DEPTH_NUMERATOR / MPC_DEPTH_DENOMINATOR) & ~1) +
            (deep_depth & 1);
        expected_shallow += MID_MPC_RECALIBRATED_VARIANT == 0
            ? MID_MPC_SHALLOW_DEPTH_OFFSET
            : mid_offsets[MID_MPC_RECALIBRATED_VARIANT];
        expected_shallow = std::max(expected_shallow, deep_depth & 1);
        expected_shallow = std::min(expected_shallow, deep_depth - 2);
#if MID_MPC_POLICY_USE_DEPTH_TABLE
        if (
            MID_MPC_POLICY_MIN_DEPTH <= deep_depth &&
            deep_depth <= MID_MPC_POLICY_MAX_DEPTH
        ) {
            expected_shallow = MID_MPC_POLICY_SHALLOW_DEPTH[
                deep_depth - MID_MPC_POLICY_MIN_DEPTH
            ];
        }
#endif
        require(
            mpc_shallow_depth<false>(deep_depth) == expected_shallow,
            "mid shallow depth " + std::to_string(deep_depth)
        );
        // A search of this depth is reachable for every board occupancy from
        // the initial four discs through n_discs + depth <= 64.
        for (int n_discs = 4; n_discs + deep_depth <= HW2; ++n_discs) {
            const std::string coordinates =
                " at n_discs=" + std::to_string(n_discs) +
                ", shallow=" + std::to_string(expected_shallow) +
                ", deep=" + std::to_string(deep_depth);
            for (const bool high : {true, false}) {
                const std::string direction = high ? " high" : " low";
                const double sigma_0 = probcut_sigma_mid_direction(
                    n_discs, 0, deep_depth, high
                );
                const double sigma_search = probcut_sigma_mid_direction(
                    n_discs, expected_shallow, deep_depth, high
                );
                require(
                    sigma_0 > 0.0,
                    "mid static sigma is not positive" + direction + coordinates
                );
                require(
                    sigma_search > 0.0,
                    "mid search sigma is not positive" + direction + coordinates
                );
                for (int mpc_level = 0; mpc_level < N_SELECTIVITY_LEVEL; ++mpc_level) {
                    const int expected_static = probcut_error_mid(mpc_level, sigma_0);
                    const int expected_search = probcut_error_mid(
                        mpc_level, sigma_search
                    );
#if USE_MPC_PRE_CALCULATION
                    const int actual_static = high
                        ? mpc_error[mpc_level][n_discs][0][deep_depth]
                        : mpc_error_low[mpc_level][n_discs][0][deep_depth];
                    const int actual_search = high
                        ? mpc_error[mpc_level][n_discs][expected_shallow][deep_depth]
                        : mpc_error_low[mpc_level][n_discs][expected_shallow][deep_depth];
                    require(
                        actual_static == expected_static,
                        "mid precalculated static error differs from formula" +
                            direction + coordinates
                    );
                    require(
                        actual_search == expected_search,
                        "mid precalculated search error differs from formula" +
                            direction + coordinates
                    );
#endif
                    require(
                        mpc_static_error<false>(
                            mpc_level, n_discs, deep_depth, high
                        ) == expected_static,
                        "mid static helper differs from formula" + direction + coordinates
                    );
                    int helper_search = 0;
                    int helper_eval = 0;
                    mpc_search_errors<false>(
                        mpc_level, n_discs, expected_shallow, deep_depth,
                        &helper_search, &helper_eval, high
                    );
                    require(
                        helper_search == expected_search,
                        "mid search helper differs from formula" + direction + coordinates
                    );
#if USE_MPC_PRE_CALCULATION
                    const int expected_eval =
                        (expected_static + expected_search + 1) / 2;
#else
                    const int expected_eval = static_cast<int>(std::ceil(
                        MPC_ERROR_SCALE * MPC_SELECTIVITY_Z_MID[mpc_level] *
                        0.5 * (sigma_0 + sigma_search)
                    ));
#endif
                    require(
                        helper_eval == expected_eval,
                        "mid evaluation helper differs from its formula" +
                            direction + coordinates
                    );
                }
            }
        }
    }

    constexpr double end_coefficients[3][6] = {
        {
            -1.3182333120273682, -6.99290557735024,
            -0.05280654146244756, 0.48284187178125065,
            5.289589936037036, 11.940601436361513
        },
        {
            -1.0, -11.448275, -0.009220543666708233,
            0.1244938461474535, 1.5497336326511917,
            6.930503478663672
        },
        {
            -1.0, -5.73455, -0.061512649507899114,
            0.49761744911268796, 3.3512295919218524,
            8.023007749425771
        }
    };
    constexpr double actual_end_coefficients[] = {
        probcut_end_a, probcut_end_b, probcut_end_c,
        probcut_end_d, probcut_end_e, probcut_end_f
    };
    for (int i = 0; i < 6; ++i) {
        require_near(
            actual_end_coefficients[i],
            end_coefficients[END_MPC_SIGMA_MODEL_VARIANT][i],
            "end coefficient " + std::to_string(i)
        );
    }
    // mpc_init fills the entire [n_discs=0..64][shallow=0..60] endgame
    // table.  Check that full index set, including the static (depth zero)
    // entries used by mpc_static_error.
    for (int n_discs = 0; n_discs <= HW2; ++n_discs) {
        for (int shallow_depth = 0; shallow_depth < HW2 - 3; ++shallow_depth) {
            const std::string coordinates =
                " at n_discs=" + std::to_string(n_discs) +
                ", shallow=" + std::to_string(shallow_depth);
            for (const bool high : {true, false}) {
                const double sigma = probcut_sigma_end_direction(
                    n_discs, shallow_depth, high
                );
                require(
                    sigma > 0.0,
                    std::string(high ? "upper" : "lower") +
                    " end sigma is not positive" + coordinates
                );
                for (int mpc_level = 0; mpc_level < N_SELECTIVITY_LEVEL; ++mpc_level) {
                    const int expected = probcut_error_end(mpc_level, sigma);
#if USE_MPC_PRE_CALCULATION
                    const int precalculated = high
                        ? mpc_error_end[mpc_level][n_discs][shallow_depth]
                        : mpc_error_end_low[mpc_level][n_discs][shallow_depth];
                    require(
                        precalculated == expected,
                        "end precalculated error differs from formula" + coordinates
                    );
#endif
                    int helper_search = 0;
                    int helper_eval = 0;
                    mpc_search_errors<true>(
                        mpc_level, n_discs, shallow_depth, 0,
                        &helper_search, &helper_eval, high
                    );
                    require(
                        helper_search == expected,
                        "end search helper differs from formula" + coordinates
                    );
                    const double sigma_0 = probcut_sigma_end_direction(
                        n_discs, 0, high
                    );
                    const int expected_static = probcut_error_end(mpc_level, sigma_0);
                    require(
                        mpc_static_error<true>(mpc_level, n_discs, 0, high) ==
                            expected_static,
                        "end static helper differs from formula" + coordinates
                    );
#if USE_MPC_PRE_CALCULATION
                    const int expected_eval = (expected_static + expected + 1) / 2;
#else
                    const int expected_eval = static_cast<int>(std::ceil(
                        MPC_ERROR_SCALE * MPC_SELECTIVITY_Z_END[mpc_level] *
                        0.5 * (sigma_0 + sigma)
                    ));
#endif
                    require(
                        helper_eval == expected_eval,
                        "end evaluation helper differs from its formula" + coordinates
                    );
                }
            }
        }
    }

    std::cout << "MPC coefficient variant tests passed\n";
    return 0;
}
