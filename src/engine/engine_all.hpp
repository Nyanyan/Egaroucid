/*
    Egaroucid Project

    @file engine_all.hpp
        Include all things about Egaroucid's engine
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "ai.hpp"
#include "book_accuracy.hpp"
#include "book_enlarge.hpp"
#include "human_like_ai.hpp"
#include "local_strategy.hpp"
#include "minimax.hpp"
#include "perft.hpp"
#include "principal_variation.hpp"
#include "problem_generator.hpp"
#include "random_board_generator.hpp"
#include "umigame.hpp"
#include "version.hpp"
#if TEST_ENDGAME_ACCURACY
#include "test.hpp"
#endif