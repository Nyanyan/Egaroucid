/* Rebuild every egbk3 leaf with the current engine, without changing its graph.
   GPL-3.0-or-later. See README.md for build, checkpoint and validation details. */
#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>
#include <thread>
#include "../../engine/engine_all.hpp"

namespace fs = std::filesystem;
using Bytes = std::vector<unsigned char>;
constexpr size_t HEADER = 14, RECORD = 25;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Bytes read_bytes(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "Cannot open " + path.string());
    Bytes result(std::istreambuf_iterator<char>(file), {});
    require(!file.bad(), "Read failed: " + path.string());
    return result;
}

uint64_t fingerprint(const Bytes& bytes) {
    uint64_t result = 14695981039346656037ULL;
    for (unsigned char c : bytes) result = (result ^ c) * 1099511628211ULL;
    return result;
}

template<class T> T load_le(const Bytes& bytes, size_t offset) {
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(T); ++i) value |= uint64_t(bytes[offset + i]) << (8 * i);
    return static_cast<T>(value);
}

struct Entry {
    Board board;
    uint64_t remaining;
    int level;
};

struct Options {
    std::string mode = "audit";
    fs::path input, output, resources, checkpoint, report;
    int threads = 24, hash_level = 25, level = 0;
    size_t limit = 0;
};

Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; i += 2) {
        require(i + 1 < argc, "Every option needs a value");
        std::string key = argv[i], value = argv[i + 1];
        if (key == "--mode") result.mode = value;
        else if (key == "--input") result.input = value;
        else if (key == "--output") result.output = value;
        else if (key == "--resources") result.resources = value;
        else if (key == "--checkpoint") result.checkpoint = value;
        else if (key == "--report") result.report = value;
        else if (key == "--threads") result.threads = std::stoi(value);
        else if (key == "--hash-level") result.hash_level = std::stoi(value);
        else if (key == "--level") result.level = std::stoi(value);
        else if (key == "--limit") result.limit = std::stoull(value);
        else throw std::runtime_error("Unknown option " + key);
    }
    require(!result.input.empty(), "--input is required");
    require(result.mode == "audit" || result.mode == "repair" || result.mode == "verify", "Unknown mode");
    require(result.threads >= 1 && result.threads <= 128, "Invalid thread count");
    require(result.hash_level >= 25 && result.hash_level <= 27, "Hash level must be 25..27");
    require(result.level >= 0 && result.level <= 60, "Level must be 0 (preserve) or 1..60");
    if (result.mode != "audit") require(!result.output.empty(), "--output is required");
    if (result.mode == "repair") {
        require(!result.resources.empty() && !result.checkpoint.empty(), "--resources and --checkpoint are required");
        require(fs::absolute(result.input).lexically_normal() != fs::absolute(result.output).lexically_normal(), "Output must differ from input");
    }
    return result;
}

void init_board() {
    bit_init(); mobility_init(); flip_init(); last_flip_init();
}

struct Book_file {
    Bytes bytes;
    std::vector<Entry> entries;
    std::unordered_set<Board, Book_hash> keys;
    std::map<int, size_t> levels;
    size_t searches = 0, invalid_values = 0, invalid_moves = 0, linked_leaves = 0, missing_leaves = 0;

    explicit Book_file(const fs::path& path, int override_level) : bytes(read_bytes(path)) {
        require(bytes.size() >= HEADER && std::memcmp(bytes.data(), "DICUORAGE\x03", 10) == 0, "Not an egbk3 file");
        const uint32_t count = load_le<uint32_t>(bytes, 10);
        require(bytes.size() == HEADER + uint64_t(count) * RECORD, "Incorrect book length");
        keys.reserve(count); entries.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            size_t p = HEADER + i * RECORD;
            Board b(load_le<uint64_t>(bytes, p), load_le<uint64_t>(bytes, p + 8));
            require((b.player & b.opponent) == 0, "Overlapping board stones");
            require(b == representative_board(b), "Input must use representative boards");
            require(keys.insert(b).second, "Duplicate board");
            int level = static_cast<int8_t>(bytes[p + 24]);
            if (level < 1 || level > 60) level = static_cast<int8_t>(bytes[p + 17]);
            if (level < 1 || level > 60) level = LEAF_CALCULATE_LEVEL;
            if (override_level) level = override_level;
            entries.push_back({b, 0, level});
        }
        for (size_t i = 0; i < entries.size(); ++i) {
            Entry& entry = entries[i];
            uint64_t legal = entry.board.get_legal();
            uint64_t remaining = legal;
            for (int cell = 0; cell < 64; ++cell) if ((legal >> cell) & 1ULL) {
                Flip flip; calc_flip(&flip, &entry.board, cell);
                Board child = entry.board.move_copy(&flip);
                bool registered;
                if (child.get_legal()) registered = keys.count(representative_board(child)) != 0;
                else if (!child.is_end()) {
                    child.pass(); registered = keys.count(representative_board(child)) != 0;
                } else {
                    registered = keys.count(representative_board(child)) != 0;
                    child.pass(); registered |= keys.count(representative_board(child)) != 0;
                }
                if (registered) remaining &= ~(1ULL << cell);
            }
            entry.remaining = remaining;
            size_t p = HEADER + i * RECORD;
            int move = static_cast<int8_t>(bytes[p + 23]);
            int value = static_cast<int8_t>(bytes[p + 22]);
            if (remaining) {
                ++searches; ++levels[entry.level];
                if (!is_valid_policy(move)) ++missing_leaves;
                else {
                    if (!(legal & (1ULL << move))) ++invalid_moves;
                    else if (!(remaining & (1ULL << move))) ++linked_leaves;
                    if (!is_valid_score(value)) ++invalid_values;
                }
            }
        }
    }
};

bool valid_leaf(const Entry& entry, int value, int move, int level) {
    if (!entry.remaining) return value == SCORE_UNDEFINED && move == MOVE_NOMOVE && level == LEVEL_UNDEFINED;
    return is_valid_score(value) && is_valid_policy(move) &&
           (entry.remaining & (1ULL << move)) && level == entry.level;
}

void set_leaf(Bytes& bytes, size_t index, int value, int move, int level) {
    size_t p = HEADER + index * RECORD + 22;
    bytes[p] = static_cast<unsigned char>(value);
    bytes[p + 1] = static_cast<unsigned char>(move);
    bytes[p + 2] = static_cast<unsigned char>(level);
}

void verify(const Book_file& original, const Bytes& output) {
    require(output.size() == original.bytes.size(), "Output length changed");
    require(std::equal(output.begin(), output.begin() + HEADER, original.bytes.begin()), "Header changed");
    for (size_t i = 0; i < original.entries.size(); ++i) {
        size_t p = HEADER + i * RECORD;
        require(std::equal(output.begin() + p, output.begin() + p + 22, original.bytes.begin() + p), "Board, value, level or n_lines changed");
        require(valid_leaf(original.entries[i], static_cast<int8_t>(output[p+22]),
                           static_cast<int8_t>(output[p+23]), static_cast<int8_t>(output[p+24])),
                "Invalid leaf at record " + std::to_string(i));
    }
}

void write_report(const Options& opt, const Book_file& original, size_t completed, bool complete, double elapsed) {
    if (opt.report.empty()) return;
    std::ofstream out(opt.report);
    require(bool(out), "Cannot create report");
    out << "{\n  \"boards\": " << original.entries.size()
        << ",\n  \"searches\": " << original.searches
        << ",\n  \"no_remaining_moves\": " << original.entries.size() - original.searches
        << ",\n  \"invalid_values_before\": " << original.invalid_values
        << ",\n  \"illegal_moves_before\": " << original.invalid_moves
        << ",\n  \"linked_leaves_before\": " << original.linked_leaves
        << ",\n  \"missing_leaves_before\": " << original.missing_leaves
        << ",\n  \"completed_searches\": " << completed
        << ",\n  \"complete\": " << (complete ? "true" : "false")
        << ",\n  \"elapsed_seconds\": " << elapsed << ",\n  \"levels\": {";
    bool first = true;
    for (const auto& item : original.levels) {
        out << (first ? "" : ", ") << '"' << item.first << "\": " << item.second;
        first = false;
    }
    out << "}\n}\n";
    require(bool(out), "Report write failed");
}

int main(int argc, char** argv) {
    try {
        Options opt = options(argc, argv);
        init_board();
        std::cout << "Auditing every board and unregistered legal move..." << std::endl;
        Book_file original(opt.input, opt.level);
        std::cout << "boards=" << original.entries.size() << " searches=" << original.searches
                  << " no_remaining=" << original.entries.size() - original.searches
                  << " invalid_values=" << original.invalid_values << " illegal_moves=" << original.invalid_moves
                  << " linked_leaves=" << original.linked_leaves << " missing=" << original.missing_leaves << std::endl;
        for (auto item : original.levels) std::cout << "level " << item.first << ": " << item.second << std::endl;
        if (opt.mode == "audit") { write_report(opt, original, 0, false, 0); return 0; }
        if (opt.mode == "verify") {
            verify(original, read_bytes(opt.output));
            write_report(opt, original, original.searches, true, 0);
            std::cout << "PASS: every leaf valid, all non-leaf bytes unchanged" << std::endl;
            return 0;
        }
        require(!fs::exists(opt.output), "Output already exists; choose a new output path");
        std::string resource_base = opt.resources.string() + "/";
        std::ostringstream identity;
        identity << "egaroucid-leaf-repair-v1 " << std::hex << fingerprint(original.bytes) << ' '
                 << fingerprint(read_bytes(opt.resources / "eval.egev2")) << ' '
                 << fingerprint(read_bytes(opt.resources / "eval_move_ordering_end.egev")) << ' '
                 << fingerprint(read_bytes(opt.resources / "hash" / ("hash" + std::to_string(opt.hash_level) + ".eghs")))
                 << ' ' << fingerprint(read_bytes(fs::absolute(argv[0])))
                 << std::dec << ' ' << opt.level << ' ' << opt.hash_level;
        Bytes output = original.bytes;
        std::vector<bool> done(original.entries.size(), false);
        size_t resumed = 0;
        if (fs::exists(opt.checkpoint)) {
            Bytes saved = read_bytes(opt.checkpoint);
            auto last = std::find(saved.rbegin(), saved.rend(), '\n');
            require(last != saved.rend(), "Checkpoint has no complete header");
            size_t valid_size = saved.size() - static_cast<size_t>(last - saved.rbegin());
            std::istringstream in(std::string(saved.begin(), saved.begin() + valid_size));
            std::string line; std::getline(in, line);
            require(line == identity.str(), "Checkpoint input, engine, evaluation or settings changed");
            if (valid_size != saved.size()) fs::resize_file(opt.checkpoint, valid_size);
            while (std::getline(in, line)) {
                std::istringstream row(line); size_t index; int value, move, level; uint64_t nodes, milliseconds;
                require(bool(row >> index >> value >> move >> level >> nodes >> milliseconds), "Malformed checkpoint row");
                std::string extra; require(!(row >> extra), "Unexpected checkpoint fields");
                require(index < done.size() && !done[index], "Duplicate or out-of-range checkpoint record");
                require(original.entries[index].remaining && valid_leaf(original.entries[index], value, move, level), "Invalid checkpoint leaf");
                set_leaf(output, index, value, move, level); done[index] = true; ++resumed;
            }
        } else {
            std::ofstream file(opt.checkpoint, std::ios::binary);
            require(bool(file << identity.str() << '\n'), "Cannot create checkpoint");
        }
        std::ofstream checkpoint(opt.checkpoint, std::ios::binary | std::ios::app);
        require(bool(checkpoint), "Cannot append checkpoint");
        std::vector<size_t> tasks;
        for (size_t i = 0; i < original.entries.size(); ++i) {
            if (!original.entries[i].remaining) set_leaf(output, i, SCORE_UNDEFINED, MOVE_NOMOVE, LEVEL_UNDEFINED);
            else if (!done[i]) tasks.push_back(i);
        }
        // Fixed shuffle spreads levels and board depths through the progress sample.
        std::mt19937 random(645); std::shuffle(tasks.begin(), tasks.end(), random);
        if (opt.limit && tasks.size() > opt.limit) tasks.resize(opt.limit);
        std::cout << "resumed=" << resumed << " this_run=" << tasks.size() << " workers=" << opt.threads << std::endl;
        endsearch_init(); move_ordering_init(); stability_init();
#if USE_MPC_PRE_CALCULATION
        mpc_init();
#endif
        // Workers run independent roots; no nested pool or Book lookup is needed.
        require(transposition_table.resize(opt.hash_level), "Cannot allocate transposition table");
        Bytes hash = read_bytes(opt.resources / "hash" / ("hash" + std::to_string(opt.hash_level) + ".eghs"));
        require(hash.size() == sizeof(hash_rand_player) + sizeof(hash_rand_opponent), "Incorrect hash file length");
        std::memcpy(hash_rand_player, hash.data(), sizeof(hash_rand_player));
        std::memcpy(hash_rand_opponent, hash.data() + sizeof(hash_rand_player), sizeof(hash_rand_opponent));
        global_hash_level = opt.hash_level;
        transposition_table.init();
        require(evaluate_init(resource_base + "eval.egev2", resource_base + "eval_move_ordering_end.egev", false), "Cannot load evaluations");
        std::atomic<size_t> next{0}, finished{0}, active{0};
        std::atomic<uint64_t> nodes{0};
        std::atomic<bool> failed{false};
        std::mutex mutex;
        std::string error;
        auto start = std::chrono::steady_clock::now();
        auto seconds = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count(); };
        auto worker = [&]() {
            try {
                while (!failed.load()) {
                    size_t task = next.fetch_add(1);
                    if (task >= tasks.size()) break;
                    size_t index = tasks[task]; const Entry& entry = original.entries[index];
                    int depth; bool mid; uint_fast8_t mpc;
                    get_level(entry.level, entry.board.n_discs() - 4, &mid, &depth, &mpc);
                    bool searching = true; ++active;
                    Search_result result = tree_search_legal(entry.board, -64, 64, depth, mpc, false,
                        entry.remaining, false, TIME_LIMIT_INF, THREAD_ID_NONE, &searching);
                    --active;
                    require(valid_leaf(entry, result.value, result.policy, entry.level), "Search returned an invalid leaf");
                    {
                        std::lock_guard<std::mutex> lock(mutex);
                        checkpoint << index << ' ' << result.value << ' ' << int(result.policy) << ' '
                                   << entry.level << ' ' << result.nodes << ' ' << result.time << '\n';
                        checkpoint.flush();
                        require(bool(checkpoint), "Checkpoint write failed");
                        set_leaf(output, index, result.value, result.policy, entry.level);
                        nodes += result.nodes; ++finished;
                    }
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(mutex); error = e.what(); failed = true;
            }
        };
        std::vector<std::thread> workers;
        for (int i = 0; i < opt.threads; ++i) workers.emplace_back(worker);
        while (finished.load() < tasks.size() && !failed.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(5));
            size_t n = finished.load(); double elapsed = seconds();
            double eta = n ? elapsed * (original.searches - resumed - n) / n : -1;
            std::cout << "progress " << resumed + n << '/' << original.searches
                      << " (" << std::fixed << std::setprecision(2) << 100.0 * (resumed + n) / std::max<size_t>(1, original.searches)
                      << "%) elapsed_s=" << elapsed << " ETA_s=" << eta << " active=" << active.load()
                      << " nodes=" << nodes.load() << std::endl;
        }
        for (auto& thread : workers) thread.join();
        require(!failed.load(), error);
        bool complete = resumed + finished.load() == original.searches;
        if (complete) {
            verify(original, output);
            fs::path temporary = opt.output; temporary += ".tmp";
            std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
            require(bool(file.write(reinterpret_cast<const char*>(output.data()), output.size())), "Output write failed");
            file.close(); require(bool(file), "Output close failed");
            verify(original, read_bytes(temporary)); fs::rename(temporary, opt.output);
        }
        write_report(opt, original, resumed + finished.load(), complete, seconds());
        std::cout << (complete ? "COMPLETE: verified repaired book written" : "PARTIAL: checkpoint saved; rerun without --limit")
                  << " elapsed_s=" << seconds() << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
