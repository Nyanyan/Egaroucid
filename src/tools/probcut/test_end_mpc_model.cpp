/*
    Focused regression tests for the recalibrated endgame MPC model.

    Example:
        clang++ -O2 -mtune=native -march=native -pthread -std=c++20 \
            test_end_mpc_model.cpp -o test_end_mpc_model.exe
*/

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

void require_equal(int actual, int expected, const std::string &message) {
    if (actual != expected) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

} // namespace

int main() {
    require(use_recalibrated_end_mpc(MPC_74_LEVEL, 10), "74% depth 10");
    require(use_recalibrated_end_mpc(MPC_93_LEVEL, 18), "93% depth 18");
    require(!use_recalibrated_end_mpc(MPC_98_LEVEL, 14), "98% fallback");
    require(!use_recalibrated_end_mpc(MPC_74_LEVEL, 9), "depth 9 fallback");
    require(!use_recalibrated_end_mpc(MPC_74_LEVEL, 19), "depth 19 fallback");

    constexpr int expected_depths[END_MPC_MODEL_SIZE] = {
        4, 5, 4, 5, 4, 7, 6, 7, 6
    };
    for (int depth = END_MPC_MODEL_MIN_DEPTH;
         depth <= END_MPC_MODEL_MAX_DEPTH; ++depth) {
        const int index = depth - END_MPC_MODEL_MIN_DEPTH;
        require_equal(
            end_mpc_shallow_depth(depth),
            expected_depths[index],
            "shallow depth " + std::to_string(depth)
        );
        for (int level = MPC_74_LEVEL; level <= MPC_93_LEVEL; ++level) {
            require_equal(
                end_mpc_shallow_error(level, depth, true),
                std::ceil(
                    END_MPC_SHALLOW_CUSHION *
                    END_MPC_SHALLOW_LOWER_TAIL[level] *
                    END_MPC_SHALLOW_SIGMA[index]
                ),
                "precomputed fail-high error"
            );
            require_equal(
                end_mpc_shallow_error(level, depth, false),
                std::ceil(
                    END_MPC_SHALLOW_CUSHION *
                    END_MPC_SHALLOW_UPPER_TAIL[level] *
                    END_MPC_SHALLOW_SIGMA[index]
                ),
                "precomputed fail-low error"
            );
        }
        require_equal(
            end_mpc_static_threshold(depth, 0, true),
            std::ceil(
                -END_MPC_STATIC_BIAS + END_MPC_STATIC_CUSHION *
                END_MPC_STATIC_LOWER_TAIL * END_MPC_STATIC_SIGMA[index]
            ),
            "precomputed static fail-high threshold"
        );
        require_equal(
            end_mpc_static_threshold(depth, 0, false),
            std::floor(
                -END_MPC_STATIC_BIAS - END_MPC_STATIC_CUSHION *
                END_MPC_STATIC_UPPER_TAIL * END_MPC_STATIC_SIGMA[index]
            ),
            "precomputed static fail-low threshold"
        );
    }

    require_equal(
        end_mpc_shallow_error(MPC_74_LEVEL, 10, true), 6,
        "74% depth 10 fail-high error"
    );
    require_equal(
        end_mpc_shallow_error(MPC_74_LEVEL, 10, false), 6,
        "74% depth 10 fail-low error"
    );
    require_equal(
        end_mpc_shallow_error(MPC_93_LEVEL, 18, true), 8,
        "93% depth 18 fail-high error"
    );
    require_equal(
        end_mpc_shallow_error(MPC_93_LEVEL, 18, false), 9,
        "93% depth 18 fail-low error"
    );
    require_equal(END_MPC_SHALLOW_GATE_SLACK, 4, "admission slack");

    require_equal(
        end_mpc_static_threshold(10, 3, true), 21,
        "static depth 10 fail-high threshold"
    );
    require_equal(
        end_mpc_static_threshold(10, 3, false), -24,
        "static depth 10 fail-low threshold"
    );

    std::cout << "Endgame MPC model tests passed" << std::endl;
    return 0;
}
