/*
    Bounded real-book regression verifier for Umigame number semantics.

    The verifier intentionally lives outside the GUI. It loads a real Egaroucid
    book, walks book-reachable positions from the initial board, and compares
    calculate_umigame() with a small independent oracle for the two semantics
    that are easy to regress:

    - Errors per Move is a local mover-perspective loss from the best child.
    - Max Allowed Eval is display-only for the current candidate move; it is
      counted here for coverage, but not passed into recursive Umigame search.
*/

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../src/engine/engine_all.hpp"

namespace {

constexpr const char* DEFAULT_BOOK_FILE = "resources/book.egbk3";
constexpr const char* DEFAULT_DEPTHS = "8,12,20,60";
constexpr const char* DEFAULT_ERRORS = "0,2,4";
constexpr const char* DEFAULT_DISPLAY_FILTERS = "inf:inf,0:0,4:2,2:4";
constexpr int DEFAULT_NODE_LIMIT = 100000;
constexpr int DEFAULT_FAIL_LIMIT = 10;

struct Candidate {
    int policy;
    int mover_value;
    Board child;
};

struct Position {
    Board board;
    int player;
    std::string transcript;
};

struct Search_case {
    int depth;
    int max_move_loss;
};

struct Display_filter {
    int black_limit = HW2;
    int white_limit = HW2;
    std::string label = "BInf_WInf";
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

struct Child_lookup {
    Board child;
    int child_player;
    int score_black;
};

struct Counters {
    int64_t visited_nodes = 0;
    int64_t checked_nodes = 0;
    int64_t checked_node_cases = 0;
    int64_t checked_moves = 0;
    int64_t shown_by_display_filters = 0;
    int64_t hidden_by_display_filters = 0;
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

std::string lower_ascii(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
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

int parse_limit_value(const std::string& text) {
    std::string normalized = lower_ascii(text);
    if (normalized == "inf" || normalized == "infinity") {
        return HW2;
    }
    return std::stoi(text);
}

std::string display_limit_label(const std::string& text) {
    std::string normalized = lower_ascii(text);
    if (normalized == "inf" || normalized == "infinity") {
        return "Inf";
    }
    return text;
}

std::vector<Display_filter> parse_display_filters(const std::string& text) {
    std::vector<Display_filter> filters;
    for (const std::string& item: split_commas(text)) {
        size_t sep = item.find(':');
        if (sep == std::string::npos) {
            throw std::runtime_error("display filter must be B:W, got " + item);
        }

        std::string black_text = item.substr(0, sep);
        std::string white_text = item.substr(sep + 1);
        Display_filter filter;
        filter.black_limit = parse_limit_value(black_text);
        filter.white_limit = parse_limit_value(white_text);
        filter.label = "B" + display_limit_label(black_text) + "_W" + display_limit_label(white_text);
        filters.emplace_back(filter);
    }
    if (filters.empty()) {
        throw std::runtime_error("display filter list must not be empty");
    }
    return filters;
}

struct Options {
    std::string book_file = DEFAULT_BOOK_FILE;
    std::vector<int> depths = parse_int_list(DEFAULT_DEPTHS);
    std::vector<int> max_move_losses = parse_int_list(DEFAULT_ERRORS);
    std::vector<Display_filter> display_filters = parse_display_filters(DEFAULT_DISPLAY_FILTERS);
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

void validate_non_negative_list(const char* name, const std::vector<int>& values) {
    for (int value: values) {
        if (value < 0) {
            throw std::runtime_error(std::string(name) + " must not contain negative values");
        }
    }
}

bool is_display_allowed(int score_black, const Display_filter& filter) {
    if (filter.black_limit < HW2 && score_black > filter.black_limit) {
        return false;
    }
    if (filter.white_limit < HW2 && score_black < -filter.white_limit) {
        return false;
    }
    return true;
}

int book_value_to_black_score(const Book_elem& elem, int player) {
    return player == BLACK ? elem.value : -elem.value;
}

std::vector<Candidate> enumerate_registered_candidates(Board board, int max_move_loss) {
    std::vector<Candidate> registered;
    uint64_t legal = board.get_legal();
    int best_value = -INF;
    Flip flip;

    for (uint_fast8_t cell = first_bit(&legal); legal; cell = next_bit(&legal)) {
        calc_flip(&flip, &board, cell);
        Board child = board.move_copy(&flip);
        Board lookup = child.copy();
        int sign = -1;
        if (lookup.get_legal() == 0ULL) {
            sign = 1;
            lookup.pass();
        }
        if (book.contain(&lookup)) {
            Book_elem elem = book.get(&lookup);
            int value = sign * elem.value;
            best_value = std::max(best_value, value);
            registered.push_back(Candidate{static_cast<int>(cell), value, child});
        }
    }

    std::vector<Candidate> kept;
    for (const Candidate& candidate: registered) {
        if (candidate.mover_value >= best_value - max_move_loss) {
            kept.emplace_back(candidate);
        }
    }
    return kept;
}

bool lookup_child_after_move(Board board, int player, int policy, Child_lookup* lookup_out) {
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
    *lookup_out = Child_lookup{
        child,
        child_player,
        book_value_to_black_score(elem, lookup_player),
    };
    return true;
}

std::string candidate_summary(Board board, int max_move_loss) {
    std::vector<Candidate> kept = enumerate_registered_candidates(board, max_move_loss);
    std::vector<Candidate> all = enumerate_registered_candidates(board, HW2);
    std::ostringstream out;
    out << "candidates:";
    if (all.empty()) {
        out << " <none>";
    }
    for (const Candidate& candidate: all) {
        bool is_kept = false;
        for (const Candidate& kept_candidate: kept) {
            if (kept_candidate.policy == candidate.policy) {
                is_kept = true;
                break;
            }
        }
        out << " " << idx_to_coord(candidate.policy) << "@" << candidate.mover_value
            << (is_kept ? ":keep" : ":drop");
    }
    return out.str();
}

class Umigame_oracle {
private:
    struct Cache_key {
        Board board;
        int player;
        Search_case search_case;

        bool operator==(const Cache_key& other) const {
            return board == other.board
                && player == other.player
                && search_case.depth == other.search_case.depth
                && search_case.max_move_loss == other.search_case.max_move_loss;
        }
    };

    struct Cache_hash {
        size_t operator()(const Cache_key& key) const {
            size_t seed = Book_hash()(key.board);
            hash_combine(&seed, static_cast<size_t>(key.player));
            hash_combine(&seed, static_cast<size_t>(key.search_case.depth));
            hash_combine(&seed, static_cast<size_t>(key.search_case.max_move_loss));
            return seed;
        }
    };

    std::unordered_map<Cache_key, Umigame_result, Cache_hash> cache;

public:
    Umigame_result solve(Board board, int player, Search_case search_case) {
        if (!book.contain(&board) || board.n_discs() >= search_case.depth + 4) {
            return make_result(1, 1);
        }

        Cache_key key{representative_board(board), player, search_case};
        auto found = cache.find(key);
        if (found != cache.end()) {
            return found->second;
        }

        if (board.get_legal() == 0ULL) {
            player ^= 1;
            board.pass();
        }

        std::vector<Candidate> candidates = enumerate_registered_candidates(board, search_case.max_move_loss);
        if (candidates.empty()) {
            Umigame_result result = make_result(1, 1);
            cache[key] = result;
            return result;
        }

        Umigame_result result;
        if (player == BLACK) {
            result.b = INF;
            result.w = 0;
            for (const Candidate& candidate: candidates) {
                Umigame_result child = solve(candidate.child, player ^ 1, search_case);
                result.b = std::min(result.b, child.b);
                result.w += child.w;
            }
        } else {
            result.b = 0;
            result.w = INF;
            for (const Candidate& candidate: candidates) {
                Umigame_result child = solve(candidate.child, player ^ 1, search_case);
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
                << " errors_per_move=" << failure.search_case.max_move_loss;
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
        << "       [--errors 0,2,4] [--display inf:inf,0:0,4:2,2:4]\n"
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
        } else if (arg == "--errors") {
            options.max_move_losses = parse_int_list(require_value(arg));
        } else if (arg == "--display") {
            options.display_filters = parse_display_filters(require_value(arg));
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
    validate_non_negative_list("depths", options.depths);
    validate_non_negative_list("errors", options.max_move_losses);
    if (options.shard_index < 0 || options.shard_index >= options.shard_count) {
        throw std::runtime_error("shard-index must be in [0, shard-count)");
    }
    return options;
}

std::vector<Search_case> make_search_cases(const Options& options) {
    std::vector<Search_case> search_cases;
    for (int depth: options.depths) {
        for (int max_move_loss: options.max_move_losses) {
            search_cases.push_back(Search_case{depth, max_move_loss});
        }
    }
    return search_cases;
}

int max_traversal_loss(const Options& options) {
    int result = 0;
    for (int loss: options.max_move_losses) {
        result = std::max(result, loss);
    }
    return result;
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
    const std::vector<Display_filter>& display_filters,
    Umigame_oracle* oracle,
    Failure_reporter* failures,
    Counters* counters
) {
    ++counters->checked_nodes;

    for (Search_case search_case: search_cases) {
        Umigame_condition condition(search_case.max_move_loss);

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
                candidate_summary(position.board, search_case.max_move_loss)
            );
            return failures->limit_reached();
        }

        std::vector<Candidate> candidates = enumerate_registered_candidates(
            position.board,
            search_case.max_move_loss
        );
        for (const Candidate& candidate: candidates) {
            Child_lookup lookup;
            if (!lookup_child_after_move(position.board, position.player, candidate.policy, &lookup)) {
                continue;
            }

            Board child_actual_board = lookup.child.copy();
            Umigame_result child_actual = calculate_umigame(
                &child_actual_board,
                lookup.child_player,
                search_case.depth,
                condition
            );
            Umigame_result child_expected = oracle->solve(
                lookup.child,
                lookup.child_player,
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
                    "move_value=" + std::to_string(candidate.mover_value)
                );
                return failures->limit_reached();
            }

            for (const Display_filter& filter: display_filters) {
                if (is_display_allowed(lookup.score_black, filter)) {
                    ++counters->shown_by_display_filters;
                } else {
                    ++counters->hidden_by_display_filters;
                }
            }
        }
    }

    return false;
}

void enqueue_children(
    const Position& position,
    int traversal_loss,
    std::unordered_set<Board, Board_hash>* seen,
    std::vector<Position>* queue
) {
    for (const Candidate& candidate: enumerate_registered_candidates(position.board, traversal_loss)) {
        Child_lookup lookup;
        if (!lookup_child_after_move(position.board, position.player, candidate.policy, &lookup)) {
            continue;
        }
        Board key = representative_board(lookup.child);
        if (seen->insert(key).second) {
            queue->push_back(Position{
                lookup.child,
                lookup.child_player,
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
    std::cout << "shown_by_display_filters=" << counters.shown_by_display_filters << "\n";
    std::cout << "hidden_by_display_filters=" << counters.hidden_by_display_filters << "\n";
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
    int traversal_loss = max_traversal_loss(options);

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
                options.display_filters,
                &oracle,
                &failures,
                &counters
            );
            if (stopped_by_fail_limit) {
                break;
            }
        }

        enqueue_children(position, traversal_loss, &seen, &queue);
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
