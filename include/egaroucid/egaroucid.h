#pragma once

#include <stdint.h>

#if defined(EGAROUCID_STATIC)
#define EGAROUCID_API
#elif defined(_WIN32)
#if defined(EGAROUCID_BUILDING_DLL)
#define EGAROUCID_API __declspec(dllexport)
#else
#define EGAROUCID_API __declspec(dllimport)
#endif
#else
#define EGAROUCID_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct egaroucid_engine egaroucid_engine;

typedef enum {
    EGAROUCID_OK = 0,
    EGAROUCID_ERROR_INVALID_ARGUMENT = -1,
    EGAROUCID_ERROR_NOT_INITIALIZED = -2,
    EGAROUCID_ERROR_SEARCH_FAILED = -3,
    EGAROUCID_ERROR_UNSUPPORTED = -4
} egaroucid_status;

typedef enum {
    EGAROUCID_EMPTY = -1,
    EGAROUCID_BLACK = 0,
    EGAROUCID_WHITE = 1
} egaroucid_disc;

typedef struct {
    uint32_t size;
    int level;
    int use_book;
    int book_accuracy_level;
    int use_multi_thread;
    int show_log;
    int time_limit_ms;
} egaroucid_search_options;

typedef struct {
    uint32_t size;
    int move;
    int value;
    int depth;
    uint64_t nodes;
    double nps;
    int is_end_search;
} egaroucid_search_result;

EGAROUCID_API const char *egaroucid_version(void);

EGAROUCID_API egaroucid_status egaroucid_global_init(const char *resource_dir);

EGAROUCID_API egaroucid_engine *egaroucid_create(void);
EGAROUCID_API void egaroucid_destroy(egaroucid_engine *engine);

EGAROUCID_API egaroucid_status egaroucid_search_array(
    egaroucid_engine *engine,
    const int board[64],
    int player,
    const egaroucid_search_options *options,
    egaroucid_search_result *result
);

EGAROUCID_API egaroucid_status egaroucid_get_legal_moves(
    const int board[64],
    int player,
    int legal_moves_out[64],
    int *n_legal_moves_out,
    uint64_t *legal_moves_mask_out
);

EGAROUCID_API egaroucid_status egaroucid_get_flipped_discs(
    const int board[64],
    int player,
    int move,
    int flipped_out[64],
    int *n_flipped_out,
    uint64_t *flipped_mask_out
);

EGAROUCID_API void egaroucid_stop(egaroucid_engine *engine);

#ifdef __cplusplus
}
#endif
