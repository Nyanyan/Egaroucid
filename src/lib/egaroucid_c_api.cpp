#include <algorithm>
#include <filesystem>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <egaroucid/egaroucid.h>

#include "engine/engine_all.hpp"

struct egaroucid_engine {
    bool searching;
};

namespace {

std::mutex g_init_mutex;
bool g_initialized = false;
bool g_book_available = false;
bool g_core_initialized = false;

struct ResourceLayout {
    std::filesystem::path resource_dir;
    std::string binary_prefix;
};

void set_default_result(egaroucid_search_result *result) {
    if (result == nullptr) {
        return;
    }
    result->size = static_cast<uint32_t>(sizeof(egaroucid_search_result));
    result->move = -1;
    result->value = 0;
    result->depth = -1;
    result->nodes = 0ULL;
    result->nps = 0.0;
    result->is_end_search = 0;
}

ResourceLayout resolve_resource_layout(const char *resource_dir) {
    ResourceLayout layout;
    std::filesystem::path input_path;
    if (resource_dir != nullptr && resource_dir[0] != '\0') {
        input_path = std::filesystem::path(resource_dir);
    } else {
        input_path = std::filesystem::path(EXE_DIRECTORY_PATH) / "resources";
    }
    input_path = input_path.lexically_normal();

    std::error_code ec;
    if (std::filesystem::exists(input_path / "eval.egev2", ec)) {
        layout.resource_dir = input_path;
    } else if (std::filesystem::exists(input_path / "resources" / "eval.egev2", ec)) {
        layout.resource_dir = input_path / "resources";
    } else {
        layout.resource_dir = input_path;
    }

    std::filesystem::path binary_dir = layout.resource_dir.parent_path();
    layout.binary_prefix = binary_dir.generic_string();
    if (!layout.binary_prefix.empty() && layout.binary_prefix.back() != '/') {
        layout.binary_prefix.push_back('/');
    }
    return layout;
}

void init_default_thread_pool() {
    const int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
    const int n_threads = std::min(48, hardware_threads);
    const int worker_threads = std::max(0, n_threads - 1);
    thread_pool.resize(worker_threads);
}

void init_core_modules_locked() {
    if (g_core_initialized) {
        return;
    }
    bit_init();
    mobility_init();
    flip_init();
    g_core_initialized = true;
}

bool ensure_core_modules_initialized() {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    try {
        init_core_modules_locked();
        return true;
    } catch (...) {
        g_core_initialized = false;
        return false;
    }
}

bool board_from_array(const int board[HW2], int player, Board *board_out) {
    int arr[HW2];
    for (int i = 0; i < HW2; ++i) {
        if (board[i] == EGAROUCID_BLACK) {
            arr[i] = BLACK;
        } else if (board[i] == EGAROUCID_WHITE) {
            arr[i] = WHITE;
        } else if (board[i] == EGAROUCID_EMPTY) {
            arr[i] = VACANT;
        } else {
            return false;
        }
    }
    board_out->translate_from_arr(arr, player);
    return true;
}

int to_public_move(int internal_policy) {
    if (is_valid_policy(internal_policy)) {
        return HW2_M1 - internal_policy;
    }
    return -1;
}

int to_internal_move(int public_move) {
    if (is_valid_policy(public_move)) {
        return HW2_M1 - public_move;
    }
    return -1;
}

void set_default_move_outputs(int moves_out[HW2], int *n_moves_out, uint64_t *moves_mask_out) {
    if (moves_out != nullptr) {
        for (int i = 0; i < HW2; ++i) {
            moves_out[i] = -1;
        }
    }
    if (n_moves_out != nullptr) {
        *n_moves_out = 0;
    }
    if (moves_mask_out != nullptr) {
        *moves_mask_out = 0ULL;
    }
}

int fill_public_moves_from_internal_mask(uint64_t internal_mask, int moves_out[HW2], uint64_t *moves_mask_out) {
    int count = 0;
    uint64_t public_mask = 0ULL;
    for (int public_move = 0; public_move < HW2; ++public_move) {
        const int internal_move = to_internal_move(public_move);
        if (1ULL & (internal_mask >> internal_move)) {
            if (moves_out != nullptr) {
                moves_out[count] = public_move;
            }
            public_mask |= (1ULL << public_move);
            ++count;
        }
    }
    if (moves_mask_out != nullptr) {
        *moves_mask_out = public_mask;
    }
    return count;
}

}  // namespace

extern "C" {

const char *egaroucid_version(void) {
    return EGAROUCID_ENGINE_VERSION;
}

egaroucid_status egaroucid_global_init(const char *resource_dir) {
    std::lock_guard<std::mutex> lock(g_init_mutex);
    try {
        const ResourceLayout layout = resolve_resource_layout(resource_dir);
        const std::string eval_file = (layout.resource_dir / "eval.egev2").generic_string();
        const std::string mo_end_file = (layout.resource_dir / "eval_move_ordering_end.egev").generic_string();
        const std::string book_file = (layout.resource_dir / "book.egbk3").generic_string();

        init_default_thread_pool();
        global_searching = true;
        init_core_modules_locked();
        last_flip_init();
        endsearch_init();
#if USE_MPC_PRE_CALCULATION
        mpc_init();
#endif
        move_ordering_init();
#if USE_CHANGEABLE_HASH_LEVEL
        hash_resize(DEFAULT_HASH_LEVEL, DEFAULT_HASH_LEVEL, layout.binary_prefix, false);
#else
        hash_tt_init(layout.binary_prefix, false);
#endif
        stability_init();
        if (!evaluate_init(eval_file, mo_end_file, false)) {
            g_initialized = false;
            g_book_available = false;
            return EGAROUCID_ERROR_NOT_INITIALIZED;
        }
        g_book_available = book_init(book_file, false);
        g_initialized = true;
        return EGAROUCID_OK;
    } catch (...) {
        g_initialized = false;
        g_book_available = false;
        return EGAROUCID_ERROR_NOT_INITIALIZED;
    }
}

egaroucid_engine *egaroucid_create(void) {
    try {
        egaroucid_engine *engine = new egaroucid_engine;
        engine->searching = true;
        return engine;
    } catch (...) {
        return nullptr;
    }
}

void egaroucid_destroy(egaroucid_engine *engine) {
    delete engine;
}

egaroucid_status egaroucid_search_array(
    egaroucid_engine *engine,
    const int board[64],
    int player,
    const egaroucid_search_options *options,
    egaroucid_search_result *result
) {
    const uint32_t requested_result_size = (result != nullptr) ? result->size : 0U;
    if (engine == nullptr || board == nullptr || options == nullptr || result == nullptr) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }
    if (options->size != sizeof(egaroucid_search_options)) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }
    if (requested_result_size != sizeof(egaroucid_search_result)) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }
    if (player != EGAROUCID_BLACK && player != EGAROUCID_WHITE) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }
    set_default_result(result);
    bool initialized = false;
    bool book_available = false;
    {
        std::lock_guard<std::mutex> lock(g_init_mutex);
        initialized = g_initialized;
        book_available = g_book_available;
    }
    if (!initialized) {
        return EGAROUCID_ERROR_NOT_INITIALIZED;
    }

    Board internal_board;
    if (!board_from_array(board, player, &internal_board)) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }

    const bool forced_pass = internal_board.get_legal() == 0ULL;
    const int search_level = std::clamp(options->level, 1, MAX_LEVEL);
    const int book_accuracy = std::clamp(options->book_accuracy_level, 0, BOOK_ACCURACY_LEVEL_INF);
    const bool use_book = options->use_book != 0 && book_available;
    const bool use_multi_thread = options->use_multi_thread != 0;
    const bool show_log = options->show_log != 0;
    (void)options->time_limit_ms;  // accepted for future API compatibility

    try {
        engine->searching = true;
        global_searching = true;
        Search_result internal_result = ai_searching(
            internal_board,
            search_level,
            use_book,
            book_accuracy,
            use_multi_thread,
            show_log,
            &engine->searching
        );

        result->move = forced_pass ? -1 : to_public_move(internal_result.policy);
        result->value = internal_result.value;
        result->depth = internal_result.depth;
        result->nodes = internal_result.nodes;
        result->nps = static_cast<double>(internal_result.nps);
        result->is_end_search = internal_result.is_end_search ? 1 : 0;

        if (internal_result.value == SCORE_UNDEFINED) {
            return EGAROUCID_ERROR_SEARCH_FAILED;
        }
        return EGAROUCID_OK;
    } catch (...) {
        return EGAROUCID_ERROR_SEARCH_FAILED;
    }
}

egaroucid_status egaroucid_get_legal_moves(
    const int board[64],
    int player,
    int legal_moves_out[64],
    int *n_legal_moves_out,
    uint64_t *legal_moves_mask_out
) {
    if (board == nullptr || n_legal_moves_out == nullptr) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }
    if (player != EGAROUCID_BLACK && player != EGAROUCID_WHITE) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }

    set_default_move_outputs(legal_moves_out, n_legal_moves_out, legal_moves_mask_out);
    if (!ensure_core_modules_initialized()) {
        return EGAROUCID_ERROR_NOT_INITIALIZED;
    }

    Board internal_board;
    if (!board_from_array(board, player, &internal_board)) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }

    try {
        const uint64_t legal_internal_mask = internal_board.get_legal();
        *n_legal_moves_out = fill_public_moves_from_internal_mask(
            legal_internal_mask,
            legal_moves_out,
            legal_moves_mask_out
        );
        return EGAROUCID_OK;
    } catch (...) {
        return EGAROUCID_ERROR_UNSUPPORTED;
    }
}

egaroucid_status egaroucid_get_flipped_discs(
    const int board[64],
    int player,
    int move,
    int flipped_out[64],
    int *n_flipped_out,
    uint64_t *flipped_mask_out
) {
    if (board == nullptr || n_flipped_out == nullptr) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }
    if (player != EGAROUCID_BLACK && player != EGAROUCID_WHITE) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }
    if (move != -1 && !is_valid_policy(move)) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }

    set_default_move_outputs(flipped_out, n_flipped_out, flipped_mask_out);
    if (move == -1) {
        return EGAROUCID_OK;
    }
    if (!ensure_core_modules_initialized()) {
        return EGAROUCID_ERROR_NOT_INITIALIZED;
    }

    Board internal_board;
    if (!board_from_array(board, player, &internal_board)) {
        return EGAROUCID_ERROR_INVALID_ARGUMENT;
    }

    try {
        const int internal_move = to_internal_move(move);
        const uint64_t legal_internal_mask = internal_board.get_legal();
        if ((legal_internal_mask & (1ULL << internal_move)) == 0ULL) {
            return EGAROUCID_OK;
        }
        Flip flip;
        const uint64_t flipped_internal_mask = calc_flip(&flip, &internal_board, static_cast<uint_fast8_t>(internal_move));
        *n_flipped_out = fill_public_moves_from_internal_mask(
            flipped_internal_mask,
            flipped_out,
            flipped_mask_out
        );
        return EGAROUCID_OK;
    } catch (...) {
        return EGAROUCID_ERROR_UNSUPPORTED;
    }
}

void egaroucid_stop(egaroucid_engine *engine) {
    if (engine != nullptr) {
        engine->searching = false;
    }
}

}  // extern "C"
