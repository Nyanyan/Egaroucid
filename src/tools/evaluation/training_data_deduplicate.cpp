// Exact duplicate/label-disagreement analysis for Egaroucid's 19-byte
// board_data records.
//
// A position is identified by the lexicographically smallest (player,
// opponent) pair among the eight D4 board symmetries.  Both bitboards receive
// the same transform: player/opponent are deliberately never exchanged.
//
// The program uses an external merge sort so that a union of data IDs is read
// exactly once even when an ID is selected by several phases.

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kRawRecordBytes = 19;
constexpr int kScoreMin = -64;
constexpr int kScoreMax = 64;
constexpr std::size_t kScoreBins = kScoreMax - kScoreMin + 1;

#pragma pack(push, 1)
struct SortRecord {
    std::uint64_t player;
    std::uint64_t opponent;
    std::uint16_t data_id;
    std::int8_t score;
    std::uint8_t phase;
};
#pragma pack(pop)

static_assert(sizeof(SortRecord) == 20);

struct RecordLess {
    bool operator()(const SortRecord& a, const SortRecord& b) const {
        return std::tie(a.phase, a.player, a.opponent, a.data_id, a.score) <
               std::tie(b.phase, b.player, b.opponent, b.data_id, b.score);
    }
};

struct PositionKey {
    std::uint8_t phase{};
    std::uint64_t player{};
    std::uint64_t opponent{};
};

bool same_key(const SortRecord& record, const PositionKey& key) {
    return record.phase == key.phase && record.player == key.player &&
           record.opponent == key.opponent;
}

struct PhaseStats {
    std::uint64_t records{};
    std::uint64_t unique_positions{};
    std::uint64_t conflicting_keys{};
    std::uint64_t conflicting_records{};
    std::uint64_t cross_id_keys{};
    std::uint64_t cross_id_records{};
    std::uint64_t cross_id_conflicting_keys{};
    std::array<std::uint64_t, kScoreBins> raw_score_hist{};
    // Key is twice the position's standard median label, so half-integer
    // medians are represented exactly. Range is [-128, +128].
    std::array<std::uint64_t, 2 * kScoreBins - 1> unique_median_hist{};
};

struct PhaseIdStats {
    std::uint64_t records{};
    std::uint64_t unique_positions{};
    std::uint64_t conflicting_keys{};
    std::uint64_t conflicting_records{};
    std::array<std::uint64_t, kScoreBins> raw_score_hist{};
    // Twice the median keeps half-integer labels exact.  This median is
    // calculated from occurrences within this data ID, rather than reusing
    // the possibly different cross-ID median of the canonical board.
    std::array<std::uint64_t, 2 * kScoreBins - 1> unique_median_hist{};
};

struct InputStats {
    std::uint64_t files{};
    std::uint64_t bytes{};
    std::uint64_t raw_records{};
    std::map<int, std::uint64_t> accepted_by_phase;
};

struct Options {
    fs::path board_root;
    fs::path output_prefix;
    fs::path temp_root;
    std::size_t chunk_records = 10'000'000;
    bool keep_temp = false;
    bool write_details = true;
    bool self_test = false;
    std::map<int, std::set<int>> phase_ids;
};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

std::string usage() {
    return
        "usage:\n"
        "  training_data_deduplicate --board-root DIR --output-prefix PATH\n"
        "      [--temp-dir DIR] [--chunk-records N] [--keep-temp] [--no-details]\n"
        "      PHASE:ID,ID,... [PHASE:ID,ID,...]\n"
        "  training_data_deduplicate --self-test\n\n"
        "Example:\n"
        "  training_data_deduplicate --board-root E:/egaroucid_data/train_data/board_data\n"
        "      --output-prefix D:/work/dedup 30:20,21,259 35:18,19,259\n\n"
        "Input records are 19 bytes: uint64 player, uint64 opponent, int8\n"
        "player_color, int8 policy, int8 score. player_color and policy are\n"
        "not part of the position key. The unique-position distribution uses\n"
        "one vote per canonical board and the standard median of all labels\n"
        "attached to that board (mean of the two middle labels for even N).\n";
}

long long parse_integer(const std::string& text, const std::string& what) {
    if (text.empty()) {
        fail("empty " + what);
    }
    std::size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used, 10);
    } catch (const std::exception&) {
        fail("invalid " + what + ": " + text);
    }
    if (used != text.size()) {
        fail("invalid " + what + ": " + text);
    }
    return value;
}

void parse_phase_spec(const std::string& spec,
                      std::map<int, std::set<int>>& phase_ids) {
    const auto colon = spec.find(':');
    if (colon == std::string::npos || colon == 0 || colon + 1 == spec.size()) {
        fail("invalid phase specification (expected PHASE:ID,ID,...): " + spec);
    }
    const auto parsed_phase = parse_integer(spec.substr(0, colon), "phase");
    if (parsed_phase < 0 || parsed_phase > 60) {
        fail("phase outside [0,60]: " + std::to_string(parsed_phase));
    }
    const int phase = static_cast<int>(parsed_phase);
    std::size_t begin = colon + 1;
    while (begin <= spec.size()) {
        const auto comma = spec.find(',', begin);
        const auto token = spec.substr(begin, comma == std::string::npos
                                                  ? std::string::npos
                                                  : comma - begin);
        const auto parsed_id = parse_integer(token, "data ID");
        if (parsed_id < 0 || parsed_id > std::numeric_limits<std::uint16_t>::max()) {
            fail("data ID outside uint16 range: " + std::to_string(parsed_id));
        }
        phase_ids[phase].insert(static_cast<int>(parsed_id));
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (++i >= argc) {
                fail("missing value for " + name);
            }
            return argv[i];
        };
        if (arg == "--board-root") {
            options.board_root = require_value(arg);
        } else if (arg == "--output-prefix") {
            options.output_prefix = require_value(arg);
        } else if (arg == "--temp-dir") {
            options.temp_root = require_value(arg);
        } else if (arg == "--chunk-records") {
            const auto n = parse_integer(require_value(arg), "chunk record count");
            if (n <= 0) {
                fail("--chunk-records must be positive");
            }
            options.chunk_records = static_cast<std::size_t>(n);
        } else if (arg == "--keep-temp") {
            options.keep_temp = true;
        } else if (arg == "--no-details") {
            options.write_details = false;
        } else if (arg == "--self-test") {
            options.self_test = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << usage();
            std::exit(0);
        } else if (!arg.empty() && arg[0] == '-') {
            fail("unknown option: " + arg);
        } else {
            parse_phase_spec(arg, options.phase_ids);
        }
    }
    if (options.self_test) {
        return options;
    }
    if (options.board_root.empty() || options.output_prefix.empty() ||
        options.phase_ids.empty()) {
        fail(usage());
    }
    if (options.temp_root.empty()) {
        options.temp_root = options.output_prefix.parent_path();
        if (options.temp_root.empty()) {
            options.temp_root = fs::current_path();
        }
    }
    return options;
}

class SymmetryTables {
public:
    SymmetryTables() {
        for (int symmetry = 0; symmetry < 8; ++symmetry) {
            for (int byte_index = 0; byte_index < 8; ++byte_index) {
                for (int byte_value = 0; byte_value < 256; ++byte_value) {
                    std::uint64_t transformed = 0;
                    for (int bit = 0; bit < 8; ++bit) {
                        if ((byte_value & (1 << bit)) == 0) {
                            continue;
                        }
                        const int source = byte_index * 8 + bit;
                        const int x = source & 7;
                        const int y = source >> 3;
                        const auto [tx, ty] = transform_coordinate(symmetry, x, y);
                        transformed |= std::uint64_t{1} << (ty * 8 + tx);
                    }
                    table_[symmetry][byte_index][byte_value] = transformed;
                }
            }
        }
    }

    std::uint64_t transform(std::uint64_t board, int symmetry) const {
        std::uint64_t result = 0;
        for (int byte_index = 0; byte_index < 8; ++byte_index) {
            result |= table_[symmetry][byte_index]
                            [static_cast<unsigned>((board >> (8 * byte_index)) & 0xffU)];
        }
        return result;
    }

    std::pair<std::uint64_t, std::uint64_t> canonical(
        std::uint64_t player, std::uint64_t opponent) const {
        auto best = std::make_pair(std::numeric_limits<std::uint64_t>::max(),
                                   std::numeric_limits<std::uint64_t>::max());
        for (int symmetry = 0; symmetry < 8; ++symmetry) {
            const auto candidate =
                std::make_pair(transform(player, symmetry),
                               transform(opponent, symmetry));
            if (candidate < best) {
                best = candidate;
            }
        }
        return best;
    }

private:
    static std::pair<int, int> transform_coordinate(int symmetry, int x, int y) {
        switch (symmetry) {
        case 0: return {x, y};                  // identity
        case 1: return {7 - y, x};              // rotate 90 degrees
        case 2: return {7 - x, 7 - y};          // rotate 180 degrees
        case 3: return {y, 7 - x};              // rotate 270 degrees
        case 4: return {7 - x, y};              // reflect left/right
        case 5: return {x, 7 - y};              // reflect top/bottom
        case 6: return {y, x};                  // reflect main diagonal
        case 7: return {7 - y, 7 - x};          // reflect anti-diagonal
        default: fail("internal invalid symmetry");
        }
    }

    std::array<std::array<std::array<std::uint64_t, 256>, 8>, 8> table_{};
};

void symmetry_self_test() {
    const SymmetryTables symmetries;
    const std::array<std::pair<std::uint64_t, std::uint64_t>, 5> tests{{
        {0x0000000810000000ULL, 0x0000001008000000ULL},
        {0x8000000000000001ULL, 0x0000000001000080ULL},
        {0x0123456789abcdefULL, 0xfedcba9876543210ULL},
        {0x0001020408102040ULL, 0x8040201008040201ULL},
        {0x0000000000000001ULL, 0x0000000000000100ULL},
    }};
    for (const auto& [player, opponent] : tests) {
        const auto expected = symmetries.canonical(player, opponent);
        for (int symmetry = 0; symmetry < 8; ++symmetry) {
            const auto actual = symmetries.canonical(
                symmetries.transform(player, symmetry),
                symmetries.transform(opponent, symmetry));
            if (actual != expected) {
                fail("symmetry self-test failed");
            }
        }
    }
    std::cout << "symmetry self-test passed\n";
}

std::string numeric_file_key(const fs::path& path) {
    const std::string stem = path.stem().string();
    const bool numeric = !stem.empty() &&
                         std::all_of(stem.begin(), stem.end(),
                                     [](unsigned char c) { return c >= '0' && c <= '9'; });
    if (!numeric) {
        return "1:" + stem;
    }
    std::ostringstream out;
    out << "0:" << std::setw(24) << std::setfill('0') << stem;
    return out.str();
}

std::vector<fs::path> list_data_files(const fs::path& directory) {
    if (!fs::exists(directory)) {
        fail("missing board-data directory: " + directory.string());
    }
    if (!fs::is_directory(directory)) {
        fail("board-data path is not a directory: " + directory.string());
    }
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".dat") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        const auto ak = numeric_file_key(a);
        const auto bk = numeric_file_key(b);
        return ak == bk ? a.filename().string() < b.filename().string() : ak < bk;
    });
    return files;
}

fs::path make_run_directory(const fs::path& temp_root) {
    fs::create_directories(temp_root);
    const auto tick = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    for (int attempt = 0; attempt < 1000; ++attempt) {
        const fs::path candidate = temp_root /
            ("training_data_deduplicate_" + std::to_string(tick) + "_" +
             std::to_string(attempt));
        std::error_code error;
        if (fs::create_directory(candidate, error)) {
            return candidate;
        }
        if (error) {
            fail("cannot create temporary run directory " + candidate.string() +
                 ": " + error.message());
        }
    }
    fail("cannot create a unique temporary run directory under " + temp_root.string());
}

fs::path output_path(const fs::path& prefix, std::string_view suffix) {
    return fs::path(prefix.string() + std::string(suffix));
}

void ensure_output_parent(const fs::path& prefix) {
    if (!prefix.parent_path().empty()) {
        fs::create_directories(prefix.parent_path());
    }
}

class ChunkWriter {
public:
    ChunkWriter(std::size_t capacity, fs::path run_directory)
        : capacity_(capacity), run_directory_(std::move(run_directory)) {
        records_.reserve(capacity_);
    }

    void add(const SortRecord& record) {
        records_.push_back(record);
        if (records_.size() >= capacity_) {
            flush();
        }
    }

    void finish() { flush(); }

    const std::vector<fs::path>& paths() const { return paths_; }

private:
    void flush() {
        if (records_.empty()) {
            return;
        }
        std::sort(records_.begin(), records_.end(), RecordLess{});
        const fs::path path = run_directory_ /
            ("chunk_" + std::to_string(paths_.size()) + ".bin");
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        if (!output) {
            fail("cannot create chunk: " + path.string());
        }
        output.write(reinterpret_cast<const char*>(records_.data()),
                     static_cast<std::streamsize>(records_.size() * sizeof(SortRecord)));
        if (!output) {
            fail("failed while writing chunk: " + path.string());
        }
        output.close();
        if (!output) {
            fail("failed while closing chunk: " + path.string());
        }
        paths_.push_back(path);
        std::cerr << "wrote sorted chunk " << paths_.size() << " ("
                  << records_.size() << " records)\n";
        records_.clear();
    }

    std::size_t capacity_;
    fs::path run_directory_;
    std::vector<SortRecord> records_;
    std::vector<fs::path> paths_;
};

class ChunkReader {
public:
    explicit ChunkReader(const fs::path& path)
        : input_(path, std::ios::binary), buffer_(65'536) {
        if (!input_) {
            fail("cannot open chunk: " + path.string());
        }
    }

    bool next(SortRecord& output) {
        if (cursor_ == valid_) {
            input_.read(reinterpret_cast<char*>(buffer_.data()),
                        static_cast<std::streamsize>(buffer_.size() * sizeof(SortRecord)));
            const auto bytes = input_.gcount();
            if (bytes == 0) {
                return false;
            }
            if (bytes % static_cast<std::streamsize>(sizeof(SortRecord)) != 0) {
                fail("truncated temporary chunk");
            }
            valid_ = static_cast<std::size_t>(bytes) / sizeof(SortRecord);
            cursor_ = 0;
        }
        output = buffer_[cursor_++];
        return true;
    }

private:
    std::ifstream input_;
    std::vector<SortRecord> buffer_;
    std::size_t cursor_{};
    std::size_t valid_{};
};

std::uint64_t read_u64(const char* input) {
    std::uint64_t value;
    std::memcpy(&value, input, sizeof(value));
    return value;
}

void scan_inputs(const Options& options,
                 const SymmetryTables& symmetries,
                 ChunkWriter& chunks,
                 std::map<int, InputStats>& inputs,
                 std::map<int, PhaseStats>& phases) {
    std::map<int, std::set<int>> id_phases;
    for (const auto& [phase, ids] : options.phase_ids) {
        phases.try_emplace(phase);
        for (const int id : ids) {
            id_phases[id].insert(phase);
        }
    }

    constexpr std::size_t kRecordsPerRead = 1'048'576;
    std::vector<char> buffer(kRawRecordBytes * kRecordsPerRead);
    std::uint64_t accepted_total = 0;

    for (const auto& [id, selected_phases] : id_phases) {
        InputStats& input_stats = inputs[id];
        const fs::path directory = options.board_root / ("records" + std::to_string(id));
        const auto files = list_data_files(directory);
        input_stats.files = files.size();
        std::cerr << "scanning ID " << id << ": " << files.size() << " files";
        if (files.empty()) {
            std::cerr << " (empty)";
        }
        std::cerr << "\n";

        for (const auto& path : files) {
            const auto bytes = fs::file_size(path);
            if (bytes % kRawRecordBytes != 0) {
                fail("file size is not divisible by 19: " + path.string());
            }
            input_stats.bytes += bytes;
            input_stats.raw_records += bytes / kRawRecordBytes;

            std::ifstream input(path, std::ios::binary);
            if (!input) {
                fail("cannot open board-data file: " + path.string());
            }
            std::uint64_t file_records_read = 0;
            while (input) {
                input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                const auto got = input.gcount();
                if (got == 0) {
                    break;
                }
                if (got % static_cast<std::streamsize>(kRawRecordBytes) != 0) {
                    fail("short non-record read from: " + path.string());
                }
                const std::size_t count =
                    static_cast<std::size_t>(got) / kRawRecordBytes;
                file_records_read += count;
                for (std::size_t index = 0; index < count; ++index) {
                    const char* raw = buffer.data() + index * kRawRecordBytes;
                    const auto player = read_u64(raw);
                    const auto opponent = read_u64(raw + 8);
                    if ((player & opponent) != 0) {
                        fail("overlapping player/opponent bitboards in " + path.string());
                    }
                    const int occupied = std::popcount(player | opponent);
                    const int phase = occupied - 4;
                    if (!selected_phases.contains(phase)) {
                        continue;
                    }
                    const int score = static_cast<std::int8_t>(raw[18]);
                    if (score < kScoreMin || score > kScoreMax) {
                        fail("score outside [-64,64] in " + path.string());
                    }
                    const auto [canonical_player, canonical_opponent] =
                        symmetries.canonical(player, opponent);
                    chunks.add(SortRecord{
                        canonical_player,
                        canonical_opponent,
                        static_cast<std::uint16_t>(id),
                        static_cast<std::int8_t>(score),
                        static_cast<std::uint8_t>(phase),
                    });
                    ++input_stats.accepted_by_phase[phase];
                    PhaseStats& phase_stats = phases[phase];
                    ++phase_stats.records;
                    ++phase_stats.raw_score_hist[score - kScoreMin];
                    ++accepted_total;
                }
            }
            if (!input.eof()) {
                fail("I/O error while reading: " + path.string());
            }
            if (file_records_read != bytes / kRawRecordBytes) {
                fail("file changed size while being read: " + path.string());
            }
        }
        std::cerr << "  accepted";
        for (const int phase : selected_phases) {
            std::cerr << " phase " << phase << "="
                      << input_stats.accepted_by_phase[phase];
        }
        std::cerr << "\n";
    }
    chunks.finish();
    std::cerr << "accepted total: " << accepted_total << " records\n";
}

int standard_median_times_two(const std::array<std::uint64_t, kScoreBins>& counts,
                              std::uint64_t total) {
    if (total == 0) {
        fail("internal empty score group");
    }
    const std::uint64_t lower_rank = (total - 1) / 2;
    const std::uint64_t upper_rank = total / 2;
    std::uint64_t cumulative = 0;
    int lower = kScoreMin;
    int upper = kScoreMin;
    bool got_lower = false;
    for (int score = kScoreMin; score <= kScoreMax; ++score) {
        cumulative += counts[score - kScoreMin];
        if (!got_lower && cumulative > lower_rank) {
            lower = score;
            got_lower = true;
        }
        if (cumulative > upper_rank) {
            upper = score;
            break;
        }
    }
    return lower + upper;
}

std::string hex64(std::uint64_t value) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return out.str();
}

std::string join_ids(const std::vector<std::uint16_t>& ids) {
    std::ostringstream out;
    for (std::size_t index = 0; index < ids.size(); ++index) {
        if (index != 0) {
            out << ';';
        }
        out << ids[index];
    }
    return out.str();
}

std::string score_hist_string(
    const std::array<std::uint64_t, kScoreBins>& counts);

struct IdDetail {
    std::uint16_t id{};
    std::uint64_t records{};
    std::array<std::uint64_t, kScoreBins> score_counts{};
};

std::vector<std::uint16_t> detail_ids(const std::vector<IdDetail>& details) {
    std::vector<std::uint16_t> result;
    result.reserve(details.size());
    for (const auto& detail : details) {
        result.push_back(detail.id);
    }
    return result;
}

std::string id_count_string(const std::vector<IdDetail>& details) {
    std::ostringstream out;
    for (std::size_t index = 0; index < details.size(); ++index) {
        if (index != 0) {
            out << ';';
        }
        out << details[index].id << ':' << details[index].records;
    }
    return out.str();
}

std::string id_score_hist_string(const std::vector<IdDetail>& details) {
    std::ostringstream out;
    for (std::size_t index = 0; index < details.size(); ++index) {
        if (index != 0) {
            out << '|';
        }
        out << details[index].id << '('
            << score_hist_string(details[index].score_counts) << ')';
    }
    return out.str();
}

std::string score_hist_string(
    const std::array<std::uint64_t, kScoreBins>& counts) {
    std::ostringstream out;
    bool first = true;
    for (int score = kScoreMin; score <= kScoreMax; ++score) {
        const auto count = counts[score - kScoreMin];
        if (count == 0) {
            continue;
        }
        if (!first) {
            out << ';';
        }
        first = false;
        out << score << ':' << count;
    }
    return out.str();
}

struct HeapItem {
    SortRecord record;
    std::size_t reader_index{};
};

struct HeapGreater {
    bool operator()(const HeapItem& a, const HeapItem& b) const {
        const RecordLess less;
        if (less(b.record, a.record)) {
            return true;
        }
        if (less(a.record, b.record)) {
            return false;
        }
        return a.reader_index > b.reader_index;
    }
};

void merge_chunks(const std::vector<fs::path>& paths,
                  const fs::path& output_prefix,
                  std::map<int, PhaseStats>& phases,
                  std::map<std::pair<int, int>, PhaseIdStats>& phase_ids,
                  bool write_details) {
    std::ofstream cross_output(output_path(output_prefix, "_cross_id.csv"),
                               std::ios::trunc);
    std::ofstream conflict_output(output_path(output_prefix, "_conflicting_labels.csv"),
                                  std::ios::trunc);
    std::ofstream cross_json(output_path(output_prefix, "_cross_id.jsonl"),
                             std::ios::trunc);
    std::ofstream conflict_json(output_path(output_prefix, "_conflicting_labels.jsonl"),
                                std::ios::trunc);
    if (!cross_output || !conflict_output || !cross_json || !conflict_json) {
        fail("cannot create detailed CSV outputs");
    }
    const std::string detail_header =
        "phase,canonical_player,canonical_opponent,total_records,data_id_count,"
        "data_ids,data_id_counts,distinct_score_count,min_score,max_score,"
        "median_label,score_counts,data_id_score_counts\n";
    cross_output << detail_header;
    conflict_output << detail_header;

    std::vector<ChunkReader> readers;
    readers.reserve(paths.size());
    for (const auto& path : paths) {
        readers.emplace_back(path);
    }
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapGreater> heap;
    for (std::size_t index = 0; index < readers.size(); ++index) {
        SortRecord record{};
        if (readers[index].next(record)) {
            heap.push({record, index});
        }
    }

    bool have_group = false;
    PositionKey key{};
    std::array<std::uint64_t, kScoreBins> score_counts{};
    std::vector<IdDetail> id_details;
    std::uint64_t group_records = 0;
    std::uint64_t merged_records = 0;

    auto finish_group = [&]() {
        if (!have_group) {
            return;
        }
        PhaseStats& stats = phases[key.phase];
        ++stats.unique_positions;
        const int median2 = standard_median_times_two(score_counts, group_records);
        ++stats.unique_median_hist[median2 - 2 * kScoreMin];

        int distinct_scores = 0;
        int min_score = kScoreMax;
        int max_score = kScoreMin;
        for (int score = kScoreMin; score <= kScoreMax; ++score) {
            if (score_counts[score - kScoreMin] != 0) {
                ++distinct_scores;
                min_score = std::min(min_score, score);
                max_score = std::max(max_score, score);
            }
        }
        const bool conflicting = distinct_scores > 1;
        const bool cross_id = id_details.size() > 1;

        for (const IdDetail& detail : id_details) {
            PhaseIdStats& id_stats = phase_ids[
                {static_cast<int>(key.phase), static_cast<int>(detail.id)}];
            id_stats.records += detail.records;
            ++id_stats.unique_positions;
            for (int score = kScoreMin; score <= kScoreMax; ++score) {
                id_stats.raw_score_hist[score - kScoreMin] +=
                    detail.score_counts[score - kScoreMin];
            }
            const int id_median2 = standard_median_times_two(
                detail.score_counts, detail.records);
            ++id_stats.unique_median_hist[id_median2 - 2 * kScoreMin];
            int id_distinct_scores = 0;
            for (const std::uint64_t count : detail.score_counts) {
                id_distinct_scores += count != 0;
            }
            if (id_distinct_scores > 1) {
                ++id_stats.conflicting_keys;
                id_stats.conflicting_records += detail.records;
            }
        }
        if (conflicting) {
            ++stats.conflicting_keys;
            stats.conflicting_records += group_records;
        }
        if (cross_id) {
            ++stats.cross_id_keys;
            stats.cross_id_records += group_records;
        }
        if (conflicting && cross_id) {
            ++stats.cross_id_conflicting_keys;
        }

        if (write_details && (conflicting || cross_id)) {
            const auto ids = detail_ids(id_details);
            std::ostringstream row;
            row << static_cast<int>(key.phase) << ',' << hex64(key.player) << ','
                << hex64(key.opponent) << ',' << group_records << ','
                << id_details.size() << ',' << join_ids(ids) << ','
                << id_count_string(id_details) << ',' << distinct_scores << ','
                << min_score << ',' << max_score << ',' << std::fixed
                << std::setprecision(1)
                << (static_cast<double>(median2) / 2.0) << ','
                << score_hist_string(score_counts) << ','
                << id_score_hist_string(id_details) << '\n';
            std::ostringstream json;
            json << "{\"phase\":" << static_cast<int>(key.phase)
                 << ",\"canonical_player\":\"" << hex64(key.player)
                 << "\",\"canonical_opponent\":\"" << hex64(key.opponent)
                 << "\",\"total_records\":" << group_records
                 << ",\"distinct_score_count\":" << distinct_scores
                 << ",\"min_score\":" << min_score
                 << ",\"max_score\":" << max_score
                 << ",\"median_label\":" << std::fixed << std::setprecision(1)
                 << (static_cast<double>(median2) / 2.0)
                 << ",\"score_counts\":{\"encoded\":\""
                 << score_hist_string(score_counts) << "\"},\"data_ids\":[";
            for (std::size_t index = 0; index < id_details.size(); ++index) {
                if (index != 0) json << ',';
                json << "{\"id\":" << id_details[index].id
                     << ",\"records\":" << id_details[index].records
                     << ",\"score_counts\":{\"encoded\":\""
                     << score_hist_string(id_details[index].score_counts)
                     << "\"}}";
            }
            json << "]}\n";
            if (cross_id) {
                cross_output << row.str();
                cross_json << json.str();
            }
            if (conflicting) {
                conflict_output << row.str();
                conflict_json << json.str();
            }
        }
    };

    while (!heap.empty()) {
        const HeapItem item = heap.top();
        heap.pop();
        const SortRecord& record = item.record;
        if (!have_group || !same_key(record, key)) {
            finish_group();
            key = PositionKey{record.phase, record.player, record.opponent};
            score_counts.fill(0);
            id_details.clear();
            group_records = 0;
            have_group = true;
        }
        ++group_records;
        ++score_counts[static_cast<int>(record.score) - kScoreMin];
        if (id_details.empty() || id_details.back().id != record.data_id) {
            id_details.push_back(IdDetail{});
            id_details.back().id = record.data_id;
        }
        ++id_details.back().records;
        ++id_details.back().score_counts[static_cast<int>(record.score) - kScoreMin];
        ++merged_records;
        if (merged_records % 50'000'000 == 0) {
            std::cerr << "merged " << merged_records << " records\n";
        }

        SortRecord next{};
        if (readers[item.reader_index].next(next)) {
            heap.push({next, item.reader_index});
        }
    }
    finish_group();
    std::cerr << "merged total: " << merged_records << " records\n";

    cross_output.flush();
    conflict_output.flush();
    cross_json.flush();
    conflict_json.flush();
    if (!cross_output || !conflict_output || !cross_json || !conflict_json) {
        fail("failed while writing duplicate-detail outputs");
    }

    std::uint64_t expected = 0;
    for (const auto& [phase, stats] : phases) {
        (void)phase;
        expected += stats.records;
    }
    if (merged_records != expected) {
        fail("merge count mismatch: expected " + std::to_string(expected) +
             ", got " + std::to_string(merged_records));
    }
}

struct Distribution {
    std::uint64_t count{};
    double mean{};
    double stddev{};
    double positive_ratio{};
    double zero_ratio{};
    double negative_ratio{};
    double mean_abs{};
    double median{};
    double p10{};
    double p25{};
    double p50{};
    double p75{};
    double p90{};
    double p95{};
    double p99{};
    std::array<std::uint64_t, 9> bands{};
    std::uint64_t unclassified{};
};

double nearest_rank_quantile(const std::map<int, std::uint64_t>& hist,
                             std::uint64_t total,
                             double probability) {
    if (total == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const auto rank = static_cast<std::uint64_t>(
        std::ceil(probability * static_cast<double>(total)));
    const std::uint64_t wanted = std::max<std::uint64_t>(1, rank);
    std::uint64_t cumulative = 0;
    for (const auto& [score2, count] : hist) {
        cumulative += count;
        if (cumulative >= wanted) {
            return static_cast<double>(score2) / 2.0;
        }
    }
    return static_cast<double>(hist.rbegin()->first) / 2.0;
}

double standard_histogram_median(const std::map<int, std::uint64_t>& hist,
                                 std::uint64_t total) {
    if (total == 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::uint64_t lower_rank = (total - 1) / 2;
    const std::uint64_t upper_rank = total / 2;
    std::uint64_t cumulative = 0;
    int lower2 = hist.begin()->first;
    int upper2 = hist.begin()->first;
    bool got_lower = false;
    for (const auto& [score2, count] : hist) {
        cumulative += count;
        if (!got_lower && cumulative > lower_rank) {
            lower2 = score2;
            got_lower = true;
        }
        if (cumulative > upper_rank) {
            upper2 = score2;
            break;
        }
    }
    return static_cast<double>(lower2 + upper2) / 4.0;
}

Distribution calculate_distribution(const std::map<int, std::uint64_t>& hist) {
    Distribution result;
    long double sum = 0;
    long double sum_squared = 0;
    long double sum_abs = 0;
    std::uint64_t positive = 0;
    std::uint64_t negative = 0;
    std::uint64_t zero = 0;
    for (const auto& [score2, count] : hist) {
        const long double score = static_cast<long double>(score2) / 2.0L;
        result.count += count;
        sum += score * count;
        sum_squared += score * score * count;
        sum_abs += std::abs(score) * count;
        if (score2 > 0) positive += count;
        else if (score2 < 0) negative += count;
        else zero += count;

        int band = -1;
        if (score <= -21.0L) band = 0;
        else if (score >= -20.0L && score <= -11.0L) band = 1;
        else if (score >= -10.0L && score <= -5.0L) band = 2;
        else if (score >= -4.0L && score <= -1.0L) band = 3;
        else if (score == 0.0L) band = 4;
        else if (score >= 1.0L && score <= 4.0L) band = 5;
        else if (score >= 5.0L && score <= 10.0L) band = 6;
        else if (score >= 11.0L && score <= 20.0L) band = 7;
        else if (score >= 21.0L) band = 8;
        if (band >= 0) result.bands[band] += count;
        else result.unclassified += count;  // possible only for half-integer medians at boundaries
    }
    if (result.count == 0) {
        const double nan = std::numeric_limits<double>::quiet_NaN();
        result.mean = result.stddev = result.positive_ratio = result.zero_ratio =
            result.negative_ratio = result.mean_abs = result.median = result.p10 =
            result.p25 = result.p50 = result.p75 = result.p90 = result.p95 =
            result.p99 = nan;
        return result;
    }
    const long double n = static_cast<long double>(result.count);
    const long double mean = sum / n;
    result.mean = static_cast<double>(mean);
    result.stddev = static_cast<double>(
        std::sqrt(std::max<long double>(0, sum_squared / n - mean * mean)));
    result.positive_ratio = static_cast<double>(positive) / result.count;
    result.zero_ratio = static_cast<double>(zero) / result.count;
    result.negative_ratio = static_cast<double>(negative) / result.count;
    result.mean_abs = static_cast<double>(sum_abs / n);
    result.p10 = nearest_rank_quantile(hist, result.count, 0.10);
    result.p25 = nearest_rank_quantile(hist, result.count, 0.25);
    result.p50 = nearest_rank_quantile(hist, result.count, 0.50);
    result.median = standard_histogram_median(hist, result.count);
    result.p75 = nearest_rank_quantile(hist, result.count, 0.75);
    result.p90 = nearest_rank_quantile(hist, result.count, 0.90);
    result.p95 = nearest_rank_quantile(hist, result.count, 0.95);
    result.p99 = nearest_rank_quantile(hist, result.count, 0.99);
    return result;
}

std::map<int, std::uint64_t> raw_histogram(const PhaseStats& stats) {
    std::map<int, std::uint64_t> result;
    for (int score = kScoreMin; score <= kScoreMax; ++score) {
        const auto count = stats.raw_score_hist[score - kScoreMin];
        if (count != 0) result[score * 2] = count;
    }
    return result;
}

std::map<int, std::uint64_t> unique_histogram(const PhaseStats& stats) {
    std::map<int, std::uint64_t> result;
    for (int score2 = 2 * kScoreMin; score2 <= 2 * kScoreMax; ++score2) {
        const auto count = stats.unique_median_hist[score2 - 2 * kScoreMin];
        if (count != 0) result[score2] = count;
    }
    return result;
}

std::map<int, std::uint64_t> raw_histogram(const PhaseIdStats& stats) {
    std::map<int, std::uint64_t> result;
    for (int score = kScoreMin; score <= kScoreMax; ++score) {
        const auto count = stats.raw_score_hist[score - kScoreMin];
        if (count != 0) result[score * 2] = count;
    }
    return result;
}

std::map<int, std::uint64_t> unique_histogram(const PhaseIdStats& stats) {
    std::map<int, std::uint64_t> result;
    for (int score2 = 2 * kScoreMin; score2 <= 2 * kScoreMax; ++score2) {
        const auto count = stats.unique_median_hist[score2 - 2 * kScoreMin];
        if (count != 0) result[score2] = count;
    }
    return result;
}

std::string json_escape(const std::string& text) {
    std::ostringstream out;
    for (const unsigned char c : text) {
        switch (c) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (c < 0x20) {
                out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                    << static_cast<int>(c) << std::dec;
            } else {
                out << c;
            }
        }
    }
    return out.str();
}

void write_outputs(const Options& options,
                   const std::map<int, InputStats>& inputs,
                   const std::map<int, PhaseStats>& phases,
                   const std::map<std::pair<int, int>, PhaseIdStats>& phase_ids,
                   std::size_t chunk_count) {
    {
        std::ofstream out(output_path(options.output_prefix, "_summary.csv"),
                          std::ios::trunc);
        out << "phase,selected_id_count,records,unique_positions,duplicate_occurrences,"
               "duplicate_rate,conflicting_keys,conflicting_records,cross_id_keys,"
               "cross_id_records,cross_id_conflicting_keys,unique_label_aggregation\n";
        out << std::setprecision(17);
        for (const auto& [phase, stats] : phases) {
            const auto duplicates = stats.records - stats.unique_positions;
            const double duplicate_rate = stats.records == 0
                ? 0.0 : static_cast<double>(duplicates) / stats.records;
            out << phase << ',' << options.phase_ids.at(phase).size() << ','
                << stats.records << ',' << stats.unique_positions << ',' << duplicates
                << ',' << duplicate_rate << ',' << stats.conflicting_keys << ','
                << stats.conflicting_records << ',' << stats.cross_id_keys << ','
                << stats.cross_id_records << ',' << stats.cross_id_conflicting_keys
                << ",standard_median\n";
        }
    }

    {
        std::ofstream out(output_path(options.output_prefix,
                                     "_distribution_summary.csv"),
                          std::ios::trunc);
        out << "phase,population,aggregation,count,mean,stddev,positive_ratio,zero_ratio,"
               "negative_ratio,mean_abs,median,p10,p25,p50,p75,p90,p95,p99,"
               "count_-64_-21,count_-20_-11,count_-10_-5,count_-4_-1,count_0,"
               "count_1_4,count_5_10,count_11_20,count_21_64,unclassified_half_boundary\n";
        out << std::setprecision(17);
        for (const auto& [phase, stats] : phases) {
            for (const auto& [population, aggregation, hist] :
                 std::array<std::tuple<std::string, std::string,
                                       std::map<int, std::uint64_t>>, 2>{{
                     {"records_with_duplicates", "none", raw_histogram(stats)},
                     {"unique_canonical_positions", "standard_median",
                      unique_histogram(stats)},
                 }}) {
                const Distribution d = calculate_distribution(hist);
                out << phase << ',' << population << ',' << aggregation << ',' << d.count
                    << ',' << d.mean << ',' << d.stddev << ',' << d.positive_ratio << ','
                    << d.zero_ratio << ',' << d.negative_ratio << ',' << d.mean_abs << ','
                    << d.median << ',' << d.p10 << ',' << d.p25 << ',' << d.p50 << ','
                    << d.p75 << ',' << d.p90 << ',' << d.p95 << ',' << d.p99;
                for (const auto count : d.bands) out << ',' << count;
                out << ',' << d.unclassified << '\n';
            }
        }
    }

    {
        std::ofstream out(output_path(options.output_prefix,
                                     "_phase_id_distribution_summary.csv"),
                          std::ios::trunc);
        out << "phase,data_id,population,aggregation,count,mean,stddev,positive_ratio,"
               "zero_ratio,negative_ratio,mean_abs,median,p10,p25,p50,p75,p90,p95,p99,"
               "count_-64_-21,count_-20_-11,count_-10_-5,count_-4_-1,count_0,"
               "count_1_4,count_5_10,count_11_20,count_21_64,unclassified_half_boundary\n";
        out << std::setprecision(17);
        for (const auto& [phase_id, stats] : phase_ids) {
            const auto [phase, data_id] = phase_id;
            for (const auto& [population, aggregation, hist] :
                 std::array<std::tuple<std::string, std::string,
                                       std::map<int, std::uint64_t>>, 2>{{
                     {"records_with_duplicates", "none", raw_histogram(stats)},
                     {"unique_canonical_positions_within_data_id", "standard_median",
                      unique_histogram(stats)},
                 }}) {
                const Distribution d = calculate_distribution(hist);
                out << phase << ',' << data_id << ',' << population << ',' << aggregation
                    << ',' << d.count << ',' << d.mean << ',' << d.stddev << ','
                    << d.positive_ratio << ',' << d.zero_ratio << ',' << d.negative_ratio
                    << ',' << d.mean_abs << ',' << d.median << ',' << d.p10 << ','
                    << d.p25 << ',' << d.p50 << ',' << d.p75 << ',' << d.p90 << ','
                    << d.p95 << ',' << d.p99;
                for (const auto count : d.bands) out << ',' << count;
                out << ',' << d.unclassified << '\n';
            }
        }
    }

    {
        std::ofstream out(output_path(options.output_prefix,
                                     "_phase_id_unique_summary.csv"),
                          std::ios::trunc);
        out << "phase,data_id,records,unique_positions_within_data_id,"
               "duplicate_occurrences_within_data_id,duplicate_rate_within_data_id,"
               "conflicting_keys_within_data_id,conflicting_records_within_data_id,"
               "unique_label_aggregation\n";
        out << std::setprecision(17);
        for (const auto& [phase_id, stats] : phase_ids) {
            const auto [phase, data_id] = phase_id;
            const std::uint64_t duplicates = stats.records - stats.unique_positions;
            const double duplicate_rate = stats.records == 0
                ? 0.0 : static_cast<double>(duplicates) / stats.records;
            out << phase << ',' << data_id << ',' << stats.records << ','
                << stats.unique_positions << ',' << duplicates << ',' << duplicate_rate
                << ',' << stats.conflicting_keys << ',' << stats.conflicting_records
                << ",standard_median_within_data_id\n";
        }
    }

    {
        std::ofstream out(output_path(options.output_prefix, "_score_histogram.csv"),
                          std::ios::trunc);
        out << "phase,population,aggregation,score,count,ratio\n";
        out << std::setprecision(17);
        for (const auto& [phase, stats] : phases) {
            for (const auto& [population, aggregation, hist] :
                 std::array<std::tuple<std::string, std::string,
                                       std::map<int, std::uint64_t>>, 2>{{
                     {"records_with_duplicates", "none", raw_histogram(stats)},
                     {"unique_canonical_positions", "standard_median",
                      unique_histogram(stats)},
                 }}) {
                std::uint64_t total = 0;
                for (const auto& [score2, count] : hist) {
                    (void)score2;
                    total += count;
                }
                for (const auto& [score2, count] : hist) {
                    out << phase << ',' << population << ',' << aggregation << ','
                        << (static_cast<double>(score2) / 2.0) << ',' << count << ','
                        << (total == 0 ? 0.0 : static_cast<double>(count) / total)
                        << '\n';
                }
            }
        }
    }

    {
        std::ofstream out(output_path(options.output_prefix, "_phase_id_counts.csv"),
                          std::ios::trunc);
        out << "phase,data_id,selected,accepted_records\n";
        for (const auto& [phase, ids] : options.phase_ids) {
            for (const int id : ids) {
                std::uint64_t accepted = 0;
                const auto input_it = inputs.find(id);
                if (input_it != inputs.end()) {
                    const auto count_it = input_it->second.accepted_by_phase.find(phase);
                    if (count_it != input_it->second.accepted_by_phase.end()) {
                        accepted = count_it->second;
                    }
                }
                out << phase << ',' << id << ",1," << accepted << '\n';
            }
        }
    }

    {
        std::ofstream out(output_path(options.output_prefix, "_input_manifest.csv"),
                          std::ios::trunc);
        out << "data_id,directory,file_count,bytes,raw_records,selected_phases\n";
        for (const auto& [id, stats] : inputs) {
            std::ostringstream selected;
            bool first = true;
            for (const auto& [phase, ids] : options.phase_ids) {
                if (ids.contains(id)) {
                    if (!first) selected << ';';
                    first = false;
                    selected << phase;
                }
            }
            out << id << ',' << (options.board_root / ("records" + std::to_string(id))).string()
                << ',' << stats.files << ',' << stats.bytes << ',' << stats.raw_records
                << ',' << selected.str() << '\n';
        }
    }

    {
        std::ofstream out(output_path(options.output_prefix, "_summary.json"),
                          std::ios::trunc);
        out << std::setprecision(17);
        out << "{\n"
               "  \"format_version\": 1,\n"
               "  \"board_root\": \"" << json_escape(options.board_root.string()) << "\",\n"
               "  \"position_key\": \"lexicographic minimum (player, opponent) under eight D4 symmetries; player/opponent never exchanged\",\n"
               "  \"phase_definition\": \"popcount(player | opponent) - 4\",\n"
               "  \"unique_label_aggregation\": \"standard median of every record label for a canonical position; even count uses arithmetic mean of middle two\",\n"
               "  \"distribution_median_definition\": \"standard median; even count uses arithmetic mean of middle two\",\n"
               "  \"quantile_definition\": \"nearest rank\",\n"
               "  \"stddev_definition\": \"population standard deviation (ddof=0)\",\n"
               "  \"chunk_records\": " << options.chunk_records << ",\n"
               "  \"chunk_count\": " << chunk_count << ",\n"
               "  \"phases\": [\n";
        bool first = true;
        for (const auto& [phase, stats] : phases) {
            if (!first) out << ",\n";
            first = false;
            const auto duplicates = stats.records - stats.unique_positions;
            const double rate = stats.records == 0
                ? 0.0 : static_cast<double>(duplicates) / stats.records;
            out << "    {\"phase\": " << phase
                << ", \"selected_id_count\": " << options.phase_ids.at(phase).size()
                << ", \"records\": " << stats.records
                << ", \"unique_positions\": " << stats.unique_positions
                << ", \"duplicate_occurrences\": " << duplicates
                << ", \"duplicate_rate\": " << rate
                << ", \"conflicting_keys\": " << stats.conflicting_keys
                << ", \"conflicting_records\": " << stats.conflicting_records
                << ", \"cross_id_keys\": " << stats.cross_id_keys
                << ", \"cross_id_records\": " << stats.cross_id_records
                << ", \"cross_id_conflicting_keys\": "
                << stats.cross_id_conflicting_keys << '}';
        }
        out << "\n  ]\n}\n";
    }
}

void cleanup_chunks(const std::vector<fs::path>& paths,
                    const fs::path& run_directory) {
    for (const auto& path : paths) {
        std::error_code error;
        fs::remove(path, error);
        if (error) {
            std::cerr << "warning: cannot remove temporary chunk " << path
                      << ": " << error.message() << '\n';
        }
    }
    std::error_code error;
    fs::remove(run_directory, error);  // succeeds only if our run dir is empty
    if (error) {
        std::cerr << "warning: cannot remove temporary run directory "
                  << run_directory << ": " << error.message() << '\n';
    }
}

int run(const Options& options) {
    if constexpr (std::endian::native != std::endian::little) {
        fail("the native 19-byte board_data format requires a little-endian host");
    }
    ensure_output_parent(options.output_prefix);
    const fs::path run_directory = make_run_directory(options.temp_root);
    std::cerr << "temporary run directory: " << run_directory << '\n';
    std::cerr << "canonicalization: 8 D4 symmetries on player/opponent together; "
                 "no player/opponent exchange\n";

    const SymmetryTables symmetries;
    ChunkWriter chunks(options.chunk_records, run_directory);
    std::map<int, InputStats> inputs;
    std::map<int, PhaseStats> phases;
    std::map<std::pair<int, int>, PhaseIdStats> phase_ids;
    scan_inputs(options, symmetries, chunks, inputs, phases);
    merge_chunks(chunks.paths(), options.output_prefix, phases, phase_ids,
                 options.write_details);
    write_outputs(options, inputs, phases, phase_ids, chunks.paths().size());
    if (options.keep_temp) {
        std::cerr << "kept temporary chunks in: " << run_directory << '\n';
    } else {
        cleanup_chunks(chunks.paths(), run_directory);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (options.self_test) {
            symmetry_self_test();
            return 0;
        }
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
