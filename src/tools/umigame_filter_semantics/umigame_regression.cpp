/*
    Bounded real-book regression verifier for generalized Umigame conditions.

    The verifier loads a real Egaroucid book, walks book-reachable positions
    from the initial board, and compares calculate_umigame() with an independent
    oracle that applies a black-perspective score range and cumulative local
    loss budget recursively at every internal node.
*/

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../engine/engine_all.hpp"

namespace {

constexpr const char* DEFAULT_BOOK_FILE = "document/book.egbk3";
constexpr const char* DEFAULT_DEPTHS = "8,12,20,60";
constexpr const char* DEFAULT_RANGES = "-64:64,0:6,3:5,-6:0";
constexpr const char* DEFAULT_INTEGRATION_ERRORS = "0,2,6";
constexpr int DEFAULT_NODE_LIMIT = 100000;
constexpr int DEFAULT_FAIL_LIMIT = 10;

struct Score_range {
    int score_min;
    int score_max;
    std::string label;
};

struct Search_case {
    int depth;
    Score_range range;
    int integration_error;
};

struct Candidate {
    int policy;
    int score_black;
    Board child;
    int child_player;
};

struct Position {
    Board board;
    int player;
    std::string transcript;
};

struct Failure {
    std::string kind;
    std::string transcript;
    std::string move;
    std::string board;
    std::string detail;
    int player;
    Search_case search_case;
    Umigame_result expected;
    Umigame_result actual;
};

struct Board_hash {
    size_t operator()(const Board& board) const {
        return Book_hash()(board);
    }
};

struct Counters {
    int64_t visited_nodes = 0;
    int64_t checked_nodes = 0;
    int64_t checked_node_cases = 0;
    int64_t checked_moves = 0;
    int64_t shown_by_score_range = 0;
    int64_t hidden_by_score_range = 0;
    int64_t hidden_out_of_book = 0;
};

void hash_combine(size_t* seed, size_t value) {
    *seed ^= value + 0x9e3779b97f4a7c15ULL + (*seed << 6) + (*seed >> 2);
}

std::string player_to_string(int player) {
    return player == BLACK ? "BLACK" : "WHITE";
}

std::string result_to_string(const Umigame_result& result) {
    return "B" + std::to_string(result.b) + " W" + std::to_string(result.w);
}

Umigame_result make_result(int b, int w) {
    Umigame_result result;
    result.b = b;
    result.w = w;
    return result;
}

bool same_result(const Umigame_result& lhs, const Umigame_result& rhs) {
    return lhs.b == rhs.b && lhs.w == rhs.w;
}

std::vector<std::string> split_commas(const std::string& text) {
    std::vector<std::string> items;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (!item.empty()) {
            items.emplace_back(item);
        }
    }
    return items;
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    for (const std::string& item: split_commas(text)) {
        values.emplace_back(std::stoi(item));
    }
    if (values.empty()) {
        throw std::runtime_error("integer list must not be empty");
    }
    return values;
}

std::vector<Score_range> parse_score_ranges(const std::string& text) {
    std::vector<Score_range> ranges;
    for (const std::string& item: split_commas(text)) {
        size_t sep = item.find(':');
        if (sep == std::string::npos) {
            throw std::runtime_error("score range must be min:max, got " + item);
        }

        int score_min = std::stoi(item.substr(0, sep));
        int score_max = std::stoi(item.substr(sep + 1));
        if (score_min > score_max) {
            std::swap(score_min, score_max);
        }
        ranges.push_back(Score_range{
            score_min,
            score_max,
            std::to_string(score_min) + "_" + std::to_string(score_max),
        });
    }
    if (ranges.empty()) {
        throw std::runtime_error("score range list must not be empty");
    }
    return ranges;
}

struct Options {
    std::string book_file = DEFAULT_BOOK_FILE;
    std::vector<int> depths = parse_int_list(DEFAULT_DEPTHS);
    std::vector<Score_range> ranges = parse_score_ranges(DEFAULT_RANGES);
    std::vector<int> integration_errors = parse_int_list(DEFAULT_INTEGRATION_ERRORS);
    int node_limit = DEFAULT_NODE_LIMIT;
    int fail_limit = DEFAULT_FAIL_LIMIT;
    int threads = 1;
    int shard_index = 0;
    int shard_count = 1;
};

void require_positive(const char* name, int value) {
    if (value <= 0) {
        throw std::runtime_error(std::string(name) + " must be positive");
    }
}

void validate_positive_list(const char* name, const std::vector<int>& values) {
    for (int value: values) {
        if (value <= 0) {
            throw std::runtime_error(std::string(name) + " must contain only positive values");
        }
    }
}

bool range_accepts(const Score_range& range, int score_black) {
    return range.score_min <= score_black && score_black <= range.score_max;
}

int book_value_to_black_score(const Book_elem& elem, int player) {
    return player == BLACK ? elem.value : -elem.value;
}

bool lookup_child_after_move(Board board, int player, int policy, Candidate* candidate_out) {
    Flip flip;
    calc_flip(&flip, &board, policy);
    Board child = board.move_copy(&flip);
    int child_player = player ^ 1;
    int lookup_player = child_player;
    Board lookup = child.copy();

    if (lookup.get_legal() == 0ULL) {
        lookup.pass();
        lookup_player ^= 1;
    }
    if (!book.contain(&lookup)) {
        return false;
    }

    Book_elem elem = book.get(&lookup);
    *candidate_out = Candidate{
        policy,
        book_value_to_black_score(elem, lookup_player),
        child,
        child_player,
    };
    return true;
}

std::vector<Candidate> enumerate_registered_candidates(Board board, int player) {
    std::vector<Candidate> candidates;
    uint64_t legal = board.get_legal();
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        Candidate candidate;
        if (lookup_child_after_move(board, player, static_cast<int>(cell), &candidate)) {
            candidates.emplace_back(candidate);
        }
    }
    return candidates;
}

std::vector<Candidate> filter_candidates(const std::vector<Candidate>& candidates, const Score_range& range) {
    std::vector<Candidate> filtered;
    for (const Candidate& candidate: candidates) {
        if (range_accepts(range, candidate.score_black)) {
            filtered.emplace_back(candidate);
        }
    }
    return filtered;
}

int count_out_of_book_moves(Board board, int player) {
    int result = 0;
    uint64_t legal = board.get_legal();
    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        Candidate candidate;
        if (!lookup_child_after_move(board, player, static_cast<int>(cell), &candidate)) {
            ++result;
        }
    }
    return result;
}

std::string candidate_summary(Board board, int player, const Score_range& range) {
    std::vector<Candidate> candidates = enumerate_registered_candidates(board, player);
    std::ostringstream out;
    out << "range=" << range.score_min << ".." << range.score_max << " candidates:";
    if (candidates.empty()) {
        out << " <none>";
    }
    for (const Candidate& candidate: candidates) {
        out << " " << idx_to_coord(candidate.policy) << "@" << candidate.score_black
            << (range_accepts(range, candidate.score_black) ? ":keep" : ":drop");
    }
    int out_of_book = count_out_of_book_moves(board, player);
    if (out_of_book > 0) {
        out << " out_of_book=" << out_of_book;
    }
    return out.str();
}

class Umigame_oracle {
private:
    struct Cache_key {
        Board board;
        int player;
        int depth;
        int score_min;
        int score_max;
        int remaining_error;

        bool operator==(const Cache_key& other) const {
            return board == other.board
                && player == other.player
                && depth == other.depth
                && score_min == other.score_min
                && score_max == other.score_max
                && remaining_error == other.remaining_error;
        }
    };

    struct Cache_hash {
        size_t operator()(const Cache_key& key) const {
            size_t seed = Book_hash()(key.board);
            hash_combine(&seed, static_cast<size_t>(key.player));
            hash_combine(&seed, static_cast<size_t>(key.depth));
            hash_combine(&seed, static_cast<size_t>(key.score_min + HW2));
            hash_combine(&seed, static_cast<size_t>(key.score_max + HW2));
            hash_combine(&seed, static_cast<size_t>(key.remaining_error));
            return seed;
        }
    };

    std::unordered_map<Cache_key, Umigame_result, Cache_hash> cache;

public:
    Umigame_result solve(Board board, int player, const Search_case& search_case) {
        return solve(board, player, search_case, search_case.integration_error);
    }

private:
    Umigame_result solve(Board board, int player, const Search_case& search_case, int remaining_error) {
        if (!book.contain(&board) || board.n_discs() >= search_case.depth + 4) {
            return make_result(1, 1);
        }

        Cache_key key{
            representative_board(board),
            player,
            search_case.depth,
            search_case.range.score_min,
            search_case.range.score_max,
            remaining_error,
        };
        auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }

        if (board.get_legal() == 0ULL) {
            player ^= 1;
            board.pass();
        }

        std::vector<Candidate> all_candidates = enumerate_registered_candidates(board, player);
        int best_value = -INF;
        for (const Candidate& candidate: all_candidates) {
            int value = player == BLACK ? candidate.score_black : -candidate.score_black;
            best_value = std::max(best_value, value);
        }
        std::vector<std::pair<Candidate, int>> candidates;
        for (const Candidate& candidate: all_candidates) {
            int value = player == BLACK ? candidate.score_black : -candidate.score_black;
            int local_loss = best_value - value;
            if (local_loss <= remaining_error && range_accepts(search_case.range, candidate.score_black)) {
                candidates.emplace_back(candidate, remaining_error - local_loss);
            }
        }
        if (candidates.empty()) {
            Umigame_result result = make_result(1, 1);
            cache[key] = result;
            return result;
        }

        Umigame_result result;
        if (player == BLACK) {
            result.b = INF;
            result.w = 0;
            for (const std::pair<Candidate, int>& candidate: candidates) {
                Umigame_result child = solve(candidate.first.child, candidate.first.child_player, search_case, candidate.second);
                result.b = std::min(result.b, child.b);
                result.w += child.w;
            }
        } else {
            result.b = 0;
            result.w = INF;
            for (const std::pair<Candidate, int>& candidate: candidates) {
                Umigame_result child = solve(candidate.first.child, candidate.first.child_player, search_case, candidate.second);
                result.w = std::min(result.w, child.w);
                result.b += child.b;
            }
        }

        cache[key] = result;
        return result;
    }
};

class Failure_reporter {
private:
    std::vector<Failure> failures;
    int fail_limit;

public:
    explicit Failure_reporter(int fail_limit)
        : fail_limit(fail_limit) {}

    void add(
        const std::string& kind,
        const Position& position,
        const std::string& move,
        Search_case search_case,
        const Umigame_result& expected,
        const Umigame_result& actual,
        const std::string& detail
    ) {
        failures.push_back(Failure{
            kind,
            position.transcript,
            move,
            position.board.to_str(position.player),
            detail,
            position.player,
            search_case,
            expected,
            actual,
        });
    }

    bool limit_reached() const {
        return static_cast<int>(failures.size()) >= fail_limit;
    }

    bool empty() const {
        return failures.empty();
    }

    size_t size() const {
        return failures.size();
    }

    void print() const {
        for (size_t i = 0; i < failures.size(); ++i) {
            const Failure& failure = failures[i];
            std::cout
                << "\n[FAIL " << (i + 1) << "] " << failure.kind
                << " transcript=" << (failure.transcript.empty() ? "<root>" : failure.transcript)
                << " player=" << player_to_string(failure.player)
                << " depth=" << failure.search_case.depth
                << " range=" << failure.search_case.range.score_min << ".." << failure.search_case.range.score_max
                << " integration_error=" << failure.search_case.integration_error;
            if (!failure.move.empty()) {
                std::cout << " move=" << failure.move;
            }
            std::cout
                << "\nexpected=" << result_to_string(failure.expected)
                << " actual=" << result_to_string(failure.actual)
                << "\nboard=" << failure.board
                << "\ndetail=" << failure.detail << "\n";
        }
    }
};

void print_usage() {
    std::cout
        << "Usage: umigame_regression --book <book.egbk3> [--depths 8,12,20,60]\n"
        << "       [--ranges -64:64,0:6,3:5,-6:0]\n"
        << "       [--integration-errors 0,2,6]\n"
        << "       [--node-limit 100000] [--fail-limit 10] [--threads 1]\n"
        << "       [--shard-index 0] [--shard-count 1]\n";
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--book") {
            options.book_file = require_value(arg);
        } else if (arg == "--depths") {
            options.depths = parse_int_list(require_value(arg));
        } else if (arg == "--ranges") {
            options.ranges = parse_score_ranges(require_value(arg));
        } else if (arg == "--integration-errors") {
            options.integration_errors = parse_int_list(require_value(arg));
        } else if (arg == "--node-limit") {
            options.node_limit = std::stoi(require_value(arg));
        } else if (arg == "--fail-limit") {
            options.fail_limit = std::stoi(require_value(arg));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value(arg));
        } else if (arg == "--shard-index") {
            options.shard_index = std::stoi(require_value(arg));
        } else if (arg == "--shard-count") {
            options.shard_count = std::stoi(require_value(arg));
        } else if (arg == "--help") {
            print_usage();
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument " + arg);
        }
    }

    require_positive("node-limit", options.node_limit);
    require_positive("fail-limit", options.fail_limit);
    require_positive("threads", options.threads);
    require_positive("shard-count", options.shard_count);
    validate_positive_list("depths", options.depths);
    for (int error: options.integration_errors) {
        if (error < 0) {
            throw std::runtime_error("integration-errors must not contain negative values");
        }
    }
    if (options.shard_index < 0 || options.shard_index >= options.shard_count) {
        throw std::runtime_error("shard-index must be in [0, shard-count)");
    }
    return options;
}

std::vector<Search_case> make_search_cases(const Options& options) {
    std::vector<Search_case> search_cases;
    for (int depth: options.depths) {
        for (const Score_range& range: options.ranges) {
            for (int integration_error: options.integration_errors) {
                search_cases.push_back(Search_case{depth, range, integration_error});
            }
        }
    }
    return search_cases;
}

void initialize_engine(const Options& options) {
    bit_init();
    mobility_init();
    flip_init();
    last_flip_init();
    endsearch_init();
    move_ordering_init();
    stability_init();
    thread_pool.resize(options.threads - 1);

    if (!book_init(options.book_file, false)) {
        throw std::runtime_error("failed to load book: " + options.book_file);
    }
}

bool should_check_shard(int64_t node_ordinal, const Options& options) {
    return node_ordinal % options.shard_count == options.shard_index;
}

bool check_node(
    const Position& position,
    const std::vector<Search_case>& search_cases,
    Umigame_oracle* oracle,
    Failure_reporter* failures,
    Counters* counters
) {
    ++counters->checked_nodes;

    for (const Search_case& search_case: search_cases) {
        Umigame_condition condition(search_case.range.score_min, search_case.range.score_max, search_case.integration_error);

        Board actual_board = position.board.copy();
        Umigame_result actual = calculate_umigame(
            &actual_board,
            position.player,
            search_case.depth,
            condition
        );
        Umigame_result expected = oracle->solve(position.board, position.player, search_case);
        ++counters->checked_node_cases;

        if (!same_result(actual, expected)) {
            failures->add(
                "node",
                position,
                "",
                search_case,
                expected,
                actual,
                candidate_summary(position.board, position.player, search_case.range)
            );
            return failures->limit_reached();
        }

        std::vector<Candidate> candidates = enumerate_registered_candidates(position.board, position.player);
        counters->hidden_out_of_book += count_out_of_book_moves(position.board, position.player);
        for (const Candidate& candidate: candidates) {
            if (!range_accepts(search_case.range, candidate.score_black)) {
                ++counters->hidden_by_score_range;
                continue;
            }
            ++counters->shown_by_score_range;

            Board child_actual_board = candidate.child.copy();
            Umigame_result child_actual = calculate_umigame(
                &child_actual_board,
                candidate.child_player,
                search_case.depth,
                condition
            );
            Umigame_result child_expected = oracle->solve(
                candidate.child,
                candidate.child_player,
                search_case
            );
            ++counters->checked_moves;

            if (!same_result(child_actual, child_expected)) {
                failures->add(
                    "move",
                    position,
                    idx_to_coord(candidate.policy),
                    search_case,
                    child_expected,
                    child_actual,
                    "score_black=" + std::to_string(candidate.score_black)
                );
                return failures->limit_reached();
            }
        }
    }

    return false;
}

void enqueue_children(
    const Position& position,
    std::unordered_set<Board, Board_hash>* seen,
    std::vector<Position>* queue
) {
    for (const Candidate& candidate: enumerate_registered_candidates(position.board, position.player)) {
        Board key = representative_board(candidate.child);
        if (seen->insert(key).second) {
            queue->push_back(Position{
                candidate.child,
                candidate.child_player,
                position.transcript + idx_to_coord(candidate.policy),
            });
        }
    }
}

void print_summary(
    const Options& options,
    const Counters& counters,
    const Failure_reporter& failures,
    bool stopped_by_node_limit,
    bool stopped_by_fail_limit
) {
    std::cout << "book=" << std::filesystem::absolute(options.book_file).string() << "\n";
    std::cout << "book_size=" << book.size() << "\n";
    std::cout << "shard=" << options.shard_index << "/" << options.shard_count << "\n";
    std::cout << "visited_nodes=" << counters.visited_nodes << " node_limit=" << options.node_limit << "\n";
    std::cout << "checked_nodes=" << counters.checked_nodes << "\n";
    std::cout << "checked_node_cases=" << counters.checked_node_cases << "\n";
    std::cout << "checked_moves=" << counters.checked_moves << "\n";
    std::cout << "shown_by_score_range=" << counters.shown_by_score_range << "\n";
    std::cout << "hidden_by_score_range=" << counters.hidden_by_score_range << "\n";
    std::cout << "hidden_out_of_book=" << counters.hidden_out_of_book << "\n";
    std::cout << "failures=" << failures.size() << " fail_limit=" << options.fail_limit << "\n";

    if (stopped_by_fail_limit) {
        std::cout << "stop_reason=fail_limit\n";
    } else if (stopped_by_node_limit) {
        std::cout << "stop_reason=node_limit\n";
    } else {
        std::cout << "stop_reason=exhausted_reachable_test_frontier\n";
    }
    failures.print();
}

int run(const Options& options) {
    initialize_engine(options);

    std::vector<Search_case> search_cases = make_search_cases(options);

    Board root;
    root.reset();
    std::vector<Position> queue;
    queue.push_back(Position{root, BLACK, ""});

    std::unordered_set<Board, Board_hash> seen;
    seen.insert(representative_board(root));

    Counters counters;
    Failure_reporter failures(options.fail_limit);
    Umigame_oracle oracle;
    bool stopped_by_node_limit = false;
    bool stopped_by_fail_limit = false;

    for (size_t head = 0; head < queue.size(); ++head) {
        if (counters.visited_nodes >= options.node_limit) {
            stopped_by_node_limit = true;
            break;
        }

        Position position = queue[head];
        int64_t node_ordinal = counters.visited_nodes++;
        if (should_check_shard(node_ordinal, options)) {
            stopped_by_fail_limit = check_node(
                position,
                search_cases,
                &oracle,
                &failures,
                &counters
            );
            if (stopped_by_fail_limit) {
                break;
            }
        }

        enqueue_children(position, &seen, &queue);
    }

    print_summary(options, counters, failures, stopped_by_node_limit, stopped_by_fail_limit);
    return failures.empty() ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        return run(options);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 2;
    }
}
