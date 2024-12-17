#include "./../../engine/ai.hpp"

int main() {
    init();
    int n = 10; // n per n_discs per cell
    int level = 10;

    constexpr int N_CELL_TYPES = 10;
    constexpr uint64_t cell_type_mask[N_CELL_TYPES] = {
        0x8100000000000081ULL, // corner
        0x4281000000008142ULL, // C
        0x2400810000810024ULL, // A
        0x1800008181000018ULL, // B
        0x0042000000004200ULL, // X
        0x0024420000422400ULL, // a
        0x0018004242001800ULL, // b
        0x0000240000240000ULL, // box corner
        0x0000182424180000ULL, // box edge
        0x0000001818000000ULL  // center
    };

    double res[HW2][N_CELL_TYPES];
    int count[HW2][N_CELL_TYPES];

    for (int i = 0; i < HW2; ++i) {
        for (int j = 0; j < N_CELL_TYPES; ++j) {
            res[i][j] = 0.0;
            count[i][j] = 0;
        }
    }

    Board board;
    Flip flip;
    for (int n_discs = 4; n_discs <= HW2; ++n_discs) {
        for (int cell_type = 0; cell_type < N_CELL_TYPES; ++cell_type) {
            for (int i = 0; i < n; ++i) {
                board.reset();
                while (board.n_discs() < n_discs && board.check_pass()) {
                    uint64_t legal = board.get_legal();
                    int random_idx = myrandrange(0, pop_count_ull(legal));
                    int t = 0;
                    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
                        if (t == random_idx) {
                            calc_flip(&flip, &board, cell);
                            break;
                        }
                        ++t;
                    }
                    board.move_board(&flip);
                }
                if (board.check_pass()) {
                    if ((board.player | board.opponent) & cell_type_mask[cell_type]) {
                        uint64_t can_be_masked = (board.player | board.opponent) & cell_type_mask[cell_type];
                        int random_idx = myrandrange(0, pop_count_ull(can_be_masked));
                        int t = 0;
                        uint64_t flipped = 0;
                        for (uint_fast8_t cell = first_bit(&can_be_masked); can_be_masked; cell = next_bit(&can_be_masked)) {
                            if (t == random_idx) {
                                flipped = 1ULL << cell;
                                break;
                            }
                            ++t;
                        }
                        Search_result complete_result = ai(board, level, true, 0, true, false);
                        int sgn = -1; // opponent -> player
                        if (board.player & flipped) { // player -> opponent
                            sgn = 1;
                        }
                        board.player ^= flipped;
                        board.opponent ^= flipped;
                            Search_result flipped_result = ai(board, level, true, 0, true, false);
                        board.player ^= flipped;
                        board.opponent ^= flipped;
                        double diff = sgn * (complete_result.value - flipped_result.value);
                        res[n_discs][cell_type] += diff;
                        ++count[n_discs][cell_type];
                    }
                }
            }
        }
    }

    for (int i = 0; i < HW2; ++i) {
        for (int j = 0; j < N_CELL_TYPES; ++j) {
            if (count[i][j]) {
                res[i][j] /= count[i][j];
            }
        }
    }

    for (int i = 0; i < HW2; ++i) {
        std::cout << "{";
        for (int j = 0; j < N_CELL_TYPES; ++j) {
            std::cout << res[i][j] << ", ";
        }
        std::cout << "}," << std::endl;
    }
}