/*
    Egaroucid Project

    @file check_line_book_edax.cpp
        Check if specific lines are in Edax-formatted book
    @date 2025
    @author Takuto Yamana
    @author GitHub Copilot
    @license GPL-3.0-or-later
*/

#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <algorithm>

// Constants
constexpr int HW = 8;
constexpr int HW2 = 64;
constexpr int SCORE_UNDEFINED = -65;
constexpr int MOVE_UNDEFINED = -1;
constexpr int LEVEL_UNDEFINED = -1;

// Direction offsets for 8 directions
constexpr int DR[] = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr int DC[] = {-1, 0, 1, -1, 1, -1, 0, 1};

// Link structure
struct Book_link {
    int8_t move;
    int8_t value;
    
    Book_link() : move(MOVE_UNDEFINED), value(SCORE_UNDEFINED) {}
    Book_link(int8_t m, int8_t v) : move(m), value(v) {}
};

// Leaf structure
struct Leaf {
    int8_t value;
    int8_t move;
    int8_t level;
    
    Leaf() : value(SCORE_UNDEFINED), move(MOVE_UNDEFINED), level(LEVEL_UNDEFINED) {}
};

// Book element
struct Book_elem {
    int8_t value;
    int8_t level;
    Leaf leaf;
    uint32_t n_lines;
    bool seen;
    std::vector<Book_link> links;
    int trans_idx; // transformation index used to get representative board
    
    Book_elem() : value(SCORE_UNDEFINED), level(LEVEL_UNDEFINED), n_lines(0), seen(false), trans_idx(0) {}
};

// Board structure
struct Board {
    uint64_t player;
    uint64_t opponent;
    
    Board() : player(0), opponent(0) {}
    
    void reset() {
        player = 0x0000000810000000ULL;
        opponent = 0x0000001008000000ULL;
    }
    
    inline int n_discs() const {
        return __builtin_popcountll(player | opponent);
    }
    
    inline bool operator==(const Board& other) const {
        return player == other.player && opponent == other.opponent;
    }
    
    // Generate legal moves
    uint64_t get_legal() const {
        uint64_t legal = 0;
        uint64_t blank = ~(player | opponent);
        
        for (int dir = 0; dir < 8; ++dir) {
            uint64_t rev = 0;
            uint64_t mask = opponent;
            
            if (dir == 0) { // Upper left
                mask &= 0x007E7E7E7E7E7E00ULL;
                rev = mask & (player >> 9);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev >> 9);
                legal |= blank & (rev >> 9);
            } else if (dir == 1) { // Upper
                mask &= 0x00FFFFFFFFFFFF00ULL;
                rev = mask & (player >> 8);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev >> 8);
                legal |= blank & (rev >> 8);
            } else if (dir == 2) { // Upper right
                mask &= 0x007E7E7E7E7E7E00ULL;
                rev = mask & (player >> 7);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev >> 7);
                legal |= blank & (rev >> 7);
            } else if (dir == 3) { // Left
                mask &= 0x7E7E7E7E7E7E7E7EULL;
                rev = mask & (player >> 1);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev >> 1);
                legal |= blank & (rev >> 1);
            } else if (dir == 4) { // Right
                mask &= 0x7E7E7E7E7E7E7E7EULL;
                rev = mask & (player << 1);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev << 1);
                legal |= blank & (rev << 1);
            } else if (dir == 5) { // Lower left
                mask &= 0x007E7E7E7E7E7E00ULL;
                rev = mask & (player << 7);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev << 7);
                legal |= blank & (rev << 7);
            } else if (dir == 6) { // Lower
                mask &= 0x00FFFFFFFFFFFF00ULL;
                rev = mask & (player << 8);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev << 8);
                legal |= blank & (rev << 8);
            } else if (dir == 7) { // Lower right
                mask &= 0x007E7E7E7E7E7E00ULL;
                rev = mask & (player << 9);
                for (int i = 0; i < 5; ++i) rev |= mask & (rev << 9);
                legal |= blank & (rev << 9);
            }
        }
        
        return legal;
    }
    
    // Make a move
    void move(int cell) {
        uint64_t place = 1ULL << cell;
        uint64_t flipped = 0;
        
        for (int dir = 0; dir < 8; ++dir) {
            uint64_t rev = 0;
            int shift = DR[dir] * 8 + DC[dir];
            uint64_t pos = place;
            
            if (shift > 0) {
                pos <<= shift;
            } else if (shift < 0) {
                pos >>= -shift;
            }
            
            while (pos && (pos & opponent)) {
                rev |= pos;
                if (shift > 0) {
                    pos <<= shift;
                } else if (shift < 0) {
                    pos >>= -shift;
                }
            }
            
            if (pos & player) {
                flipped |= rev;
            }
        }
        
        // Update board: place stone and flip
        uint64_t next_player = opponent ^ flipped;
        uint64_t next_opponent = (player | place | flipped);
        player = next_player;
        opponent = next_opponent;
    }
    
    void pass() {
        uint64_t tmp = player;
        player = opponent;
        opponent = tmp;
    }
};

// Board transformations for symmetry (following util.hpp implementation)
Board board_vertical_mirror(const Board& b) {
    Board res;
    res.player = __builtin_bswap64(b.player);
    res.opponent = __builtin_bswap64(b.opponent);
    return res;
}

Board board_horizontal_mirror(const Board& b) {
    Board res;
    for (int i = 0; i < HW; ++i) {
        for (int j = 0; j < HW; ++j) {
            int idx = i * HW + j;
            int mirror_idx = i * HW + (HW - 1 - j);
            if (b.player & (1ULL << idx)) res.player |= (1ULL << mirror_idx);
            if (b.opponent & (1ULL << idx)) res.opponent |= (1ULL << mirror_idx);
        }
    }
    return res;
}

Board board_black_line_mirror(const Board& b) {
    Board res;
    for (int i = 0; i < HW; ++i) {
        for (int j = 0; j < HW; ++j) {
            int idx = i * HW + j;
            int mirror_idx = j * HW + i;
            if (b.player & (1ULL << idx)) res.player |= (1ULL << mirror_idx);
            if (b.opponent & (1ULL << idx)) res.opponent |= (1ULL << mirror_idx);
        }
    }
    return res;
}

bool compare_representative_board(Board* res, const Board& cmp) {
    if (res->player > cmp.player || (res->player == cmp.player && res->opponent > cmp.opponent)) {
        res->player = cmp.player;
        res->opponent = cmp.opponent;
        return true;
    }
    return false;
}

// Get representative board (canonical form) - following util.hpp implementation
Board representative_board(const Board& b) {
    Board res = b;
    Board bt = b;   bt = board_black_line_mirror(bt);          compare_representative_board(&res, bt);
    Board bv =      board_vertical_mirror(b);                  compare_representative_board(&res, bv);
    Board btv =     board_vertical_mirror(bt);                 compare_representative_board(&res, btv);
    Board b_h =     board_horizontal_mirror(b);                compare_representative_board(&res, b_h);
    Board bt_h =    board_horizontal_mirror(bt);               compare_representative_board(&res, bt_h);
    Board b_hv =    board_vertical_mirror(b_h);                compare_representative_board(&res, b_hv);
    Board bt_hv =   board_vertical_mirror(bt_h);               compare_representative_board(&res, bt_hv);
    return res;
}

// Get representative board with transformation index - following util.hpp implementation
Board representative_board(const Board& b, int* idx) {
    Board res = b;                                                                  *idx = 0; // default
    Board bt = b;   bt = board_black_line_mirror(bt);          if (compare_representative_board(&res, bt))     *idx = 2; // black line
    Board bv =      board_vertical_mirror(b);                  if (compare_representative_board(&res, bv))     *idx = 1; // vertical
    Board btv =     board_vertical_mirror(bt);                 if (compare_representative_board(&res, btv))    *idx = 3; // black line + vertical
    Board b_h =     board_horizontal_mirror(b);                if (compare_representative_board(&res, b_h))    *idx = 6; // horizontal
    Board bt_h =    board_horizontal_mirror(bt);               if (compare_representative_board(&res, bt_h))   *idx = 4; // black line + horizontal
    Board b_hv =    board_vertical_mirror(b_h);                if (compare_representative_board(&res, b_hv))   *idx = 7; // horizontal + vertical
    Board bt_hv =   board_vertical_mirror(bt_h);               if (compare_representative_board(&res, bt_hv))  *idx = 5; // black line + horizontal + vertical
    return res;
}

// Convert coordinate to representative board - copied from util.hpp
int convert_coord_to_representative_board(int cell, int idx) {
    if (cell < 0 || cell >= HW2) return cell;
    int res;
    int y = cell / HW;
    int x = cell % HW;
    switch (idx) {
        case 0:
            res = cell;
            break;
        case 1:
            res = (HW - 1 - y) * HW + x; // vertical
            break;
        case 2:
            res = x * HW + y; // black line (transpose)
            break;
        case 3:
            res = (HW - 1 - x) * HW + y; // black line + vertical ( = rotate 90 counterclockwise)
            break;
        case 4:
            res = x * HW + (HW - 1 - y); // black line + horizontal ( = rotate 90 clockwise)
            break;
        case 5:
            res = (HW - 1 - x) * HW + (HW - 1 - y); // black line + horizontal + vertical ( = white line)
            break;
        case 6:
            res = y * HW + (HW - 1 - x); // horizontal
            break;
        case 7:
            res = (HW - 1 - y) * HW + (HW - 1 - x); // horizontal + vertical ( = rotate180)
            break;
        default:
            res = cell;
            std::cerr << "converting coord error" << std::endl;
            break;
    }
    return res;
}

// Convert coordinate from representative board - copied from util.hpp
int convert_coord_from_representative_board(int cell, int idx) {
    if (cell < 0 || cell >= HW2) return cell;
    int res;
    int y = cell / HW;
    int x = cell % HW;
    switch (idx) {
        case 0:
            res = cell;
            break;
        case 1:
            res = (HW - 1 - y) * HW + x; // vertical
            break;
        case 2:
            res = x * HW + y; // black line (transpose, self-inverse)
            break;
        case 3:
            res = x * HW + (HW - 1 - y); // black line + vertical ( = rotate 90 counterclockwise)
            break;
        case 4:
            res = (HW - 1 - x) * HW + y; // black line + horizontal ( = rotate 90 clockwise)
            break;
        case 5:
            res = (HW - 1 - x) * HW + (HW - 1 - y); // black line + horizontal + vertical ( = white line, self-inverse)
            break;
        case 6:
            res = y * HW + (HW - 1 - x); // horizontal
            break;
        case 7:
            res = (HW - 1 - y) * HW + (HW - 1 - x); // horizontal + vertical ( = rotate180)
            break;
        default:
            res = cell;
            std::cerr << "converting coord error" << std::endl;
            break;
    }
    return res;
}

// Hash function for Board
struct Book_hash {
    size_t operator()(const Board& b) const {
        return std::hash<uint64_t>{}(b.player) ^ (std::hash<uint64_t>{}(b.opponent) << 1);
    }
};

// Convert move string (e.g., "f5") to cell index
int move_string_to_cell(std::string str) {
    if (str.length() != 2) return -1;
    
    // Convert to lowercase
    for (char& c : str) {
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    }
    
    int col = str[0] - 'a';
    int row = str[1] - '1';
    
    if (col < 0 || col >= HW || row < 0 || row >= HW) return -1;
    
    return row * HW + col;
}

// Convert cell index to move string
std::string cell_to_move_string(int cell) {
    if (cell < 0 || cell >= HW2) return "??";
    
    int row = cell / HW;
    int col = cell % HW;
    
    std::string result;
    result += ('a' + col);
    result += ('1' + row);
    return result;
}

// Print board state
void print_board(const Board& b, const std::string& title) {
    std::cout << "  " << title << ":" << std::endl;
    std::cout << "    a b c d e f g h" << std::endl;
    for (int i = 0; i < HW; ++i) {
        std::cout << "  " << (i + 1) << " ";
        for (int j = 0; j < HW; ++j) {
            int cell = i * HW + j;
            if (b.player & (1ULL << cell)) {
                std::cout << "X ";
            } else if (b.opponent & (1ULL << cell)) {
                std::cout << "O ";
            } else {
                std::cout << ". ";
            }
        }
        std::cout << std::endl;
    }
    std::cout << "  Player: 0x" << std::hex << b.player << std::dec << std::endl;
    std::cout << "  Opponent: 0x" << std::hex << b.opponent << std::dec << std::endl;
}

// Book class
class Book {
private:
    std::unordered_map<Board, Book_elem, Book_hash> book;
    
public:
    bool import_file_edax(const std::string& file, bool show_log) {
        std::ifstream ifs(file, std::ios::binary);
        if (!ifs) {
            std::cerr << "[ERROR] cannot open " << file << std::endl;
            return false;
        }
        
        // Header (38 bytes total)
        char header[38];
        ifs.read(header, 38);
        if (!ifs || ifs.gcount() < 38) {
            std::cerr << "[ERROR] file broken (header)" << std::endl;
            return false;
        }
        
        // Number of boards
        int n_boards;
        ifs.read((char*)&n_boards, 4);
        if (!ifs) {
            std::cerr << "[ERROR] file broken (n_boards)" << std::endl;
            return false;
        }
        
        if (show_log) {
            std::cerr << "loading Edax book with " << n_boards << " boards" << std::endl;
        }
        
        int percent = -1;
        for (int i = 0; i < n_boards; ++i) {
            int n_percent = (int)((double)i / n_boards * 100);
            if (n_percent > percent && show_log) {
                percent = n_percent;
                std::cerr << "loading book " << percent << "%" << std::endl;
            }
            
            uint64_t player, opponent;
            int32_t wdl[3];
            uint32_t n_lines;
            int16_t value;
            int16_t additional[2];
            char link, level, leaf_value, leaf_move;
            
            // Read board player
            ifs.read((char*)&player, 8);
            if (!ifs) break;
            
            // Read board opponent
            ifs.read((char*)&opponent, 8);
            if (!ifs) break;
            
            // Read additional data (w/d/l)
            ifs.read((char*)wdl, 12);
            if (!ifs) break;
            
            // Read n_lines
            ifs.read((char*)&n_lines, 4);
            if (!ifs) break;
            
            // Read value
            ifs.read((char*)&value, 2);
            if (!ifs) break;
            
            // Read additional data
            ifs.read((char*)additional, 4);
            if (!ifs) break;
            
            // Read link count
            ifs.read(&link, 1);
            if (!ifs) break;
            
            // Read level
            ifs.read(&level, 1);
            if (!ifs) break;
            
            // For each link
            std::vector<Book_link> links;
            for (int j = 0; j < (int)link; ++j) {
                char link_value, link_move;
                ifs.read(&link_value, 1);
                if (!ifs) break;
                ifs.read(&link_move, 1);
                if (!ifs) break;
                links.emplace_back(link_move, link_value);
            }
            if (!ifs) break;
            
            // Read leaf value
            ifs.read(&leaf_value, 1);
            if (!ifs) break;
            
            // Read leaf move
            ifs.read(&leaf_move, 1);
            if (!ifs) break;
            
            // Validate and store
            if (value >= -HW2 && value <= HW2 && (player & opponent) == 0) {
                Board board;
                board.player = player;
                board.opponent = opponent;
                
                int trans_idx;
                Board repr = representative_board(board, &trans_idx);
                
                Book_elem elem;
                elem.value = (int8_t)value;
                elem.level = level;
                elem.n_lines = n_lines;
                elem.trans_idx = trans_idx; // Store transformation index
                
                // Transform leaf move to representative board coordinates
                elem.leaf.value = leaf_value;
                if (leaf_move >= 0 && leaf_move < HW2) {
                    elem.leaf.move = convert_coord_to_representative_board(leaf_move, trans_idx);
                } else {
                    elem.leaf.move = leaf_move;
                }
                elem.leaf.level = level;
                
                // Transform link moves to representative board coordinates
                for (const auto& link : links) {
                    Book_link transformed_link;
                    transformed_link.value = link.value;
                    if (link.move >= 0 && link.move < HW2) {
                        transformed_link.move = convert_coord_to_representative_board(link.move, trans_idx);
                    } else {
                        transformed_link.move = link.move;
                    }
                    elem.links.push_back(transformed_link);
                }
                
                book[repr] = elem;
            }
        }
        
        if (show_log) {
            std::cerr << "loaded " << book.size() << " positions" << std::endl;
        }
        
        return true;
    }
    
    bool contain(const Board* b) const {
        Board repr = representative_board(*b);
        return book.find(repr) != book.end();
    }
    
    Book_elem get(const Board& b) const {
        int current_trans_idx;
        Board repr = representative_board(b, &current_trans_idx);
        auto it = book.find(repr);
        if (it != book.end()) {
            Book_elem elem = it->second;
            // Store the transformation index for current board
            elem.trans_idx = current_trans_idx;
            return elem;
        }
        return Book_elem();
    }
    
    size_t get_n_book() const {
        return book.size();
    }
};

// Parse a line string (e.g., "f5d6c3") into individual moves
std::vector<std::string> parse_line(const std::string& line_str) {
    std::vector<std::string> moves;
    for (size_t i = 0; i + 1 < line_str.length(); i += 2) {
        std::string move = line_str.substr(i, 2);
        moves.push_back(move);
    }
    return moves;
}

// Check a single line
bool check_single_line(Book& test_book, const std::string& line_str) {
    std::vector<std::string> moves = parse_line(line_str);
    
    if (moves.empty()) {
        std::cerr << "[WARNING] Empty line: " << line_str << std::endl;
        return false;
    }

    Board board;
    board.reset();
    
    std::cout << "Checking line: " << line_str << " (";
    for (size_t i = 0; i < moves.size(); ++i) {
        if (i > 0) std::cout << " ";
        std::cout << moves[i];
    }
    std::cout << ")" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Initial board
    std::cout << "Initial board (move 0):" << std::endl;
    // print_board(board, "  Current board");
    if (test_book.contain(&board)) {
        // int trans_idx;
        // Board repr = representative_board(board, &trans_idx);
        // print_board(repr, "  Representative board (trans=" + std::to_string(trans_idx) + ")");
        Book_elem elem = test_book.get(board);
        std::cout << "  [FOUND] Value: " << (int)elem.value 
                  << ", Level: " << (int)elem.level 
                  << ", N_lines: " << elem.n_lines << std::endl;
        
        // Check if next move exists in links/leaf
        int next_cell = -1;
        if (!moves.empty()) {
            next_cell = move_string_to_cell(moves[0]);
        }
        
        // Display leaf in original orientation
        int orig_leaf_move = elem.leaf.move;
        if (elem.leaf.move >= 0 && elem.leaf.move < HW2) {
            orig_leaf_move = convert_coord_from_representative_board(elem.leaf.move, elem.trans_idx);
        }
        std::cout << "  Leaf: " << cell_to_move_string(orig_leaf_move);
        // if (elem.leaf.move >= 0 && elem.leaf.move < HW2) {
        //     std::cout << " [repr: " << cell_to_move_string(elem.leaf.move) << "]";
        // }
        std::cout << " (value: " << (int)elem.leaf.value 
                  << ", level: " << (int)elem.leaf.level << ")" << std::endl;
        if (elem.leaf.move != MOVE_UNDEFINED && next_cell == orig_leaf_move) {
            std::cout << "  *** NEXT MOVE IS LEAF ***" << std::endl;
        }
        
        // Display links in original orientation
        if (!elem.links.empty()) {
            std::cout << "  Links (" << elem.links.size() << "):";
            bool next_found_in_links = false;
            for (size_t j = 0; j < elem.links.size(); ++j) {
                if (j > 0) std::cout << ",";
                int stored_link_move = elem.links[j].move;
                int orig_link_move = stored_link_move;
                if (stored_link_move >= 0 && stored_link_move < HW2) {
                    orig_link_move = convert_coord_from_representative_board(stored_link_move, elem.trans_idx);
                }
                std::cout << " " << cell_to_move_string(orig_link_move);
                // if (stored_link_move >= 0 && stored_link_move < HW2) {
                //     std::cout << "[repr:" << cell_to_move_string(stored_link_move) << "]";
                // }
                std::cout << "(" << (int)elem.links[j].value << ")";
                if (next_cell == orig_link_move) {
                    std::cout << "***";
                    next_found_in_links = true;
                }
            }
            std::cout << std::endl;
            if (next_found_in_links) {
                std::cout << "  *** NEXT MOVE FOUND IN LINKS ***" << std::endl;
            }
        }
    } else {
        std::cout << "  [POSITION NOT FOUND]" << std::endl;
    }
    std::cout << std::endl;

    // Apply each move and check
    bool all_found = true;
    for (size_t i = 0; i < moves.size(); ++i) {
        std::string move_str = moves[i];
        
        // Convert move string to cell index
        int cell = move_string_to_cell(move_str);
        if (cell == -1) {
            std::cerr << "[ERROR] Invalid move format: " << move_str << std::endl;
            return false;
        }

        uint64_t legal = board.get_legal();
        if (!(legal & (1ULL << cell))) {
            std::cerr << "[ERROR] Illegal move at position " << (i + 1) << ": " << move_str << std::endl;
            std::cerr << "Legal moves: ";
            for (int j = 0; j < HW2; ++j) {
                if (legal & (1ULL << j)) {
                    std::cerr << cell_to_move_string(j) << " ";
                }
            }
            std::cerr << std::endl;
            return false;
        }

        // Make the move
        board.move(cell);

        // Check if in book
        std::string transcript_so_far;
        for (size_t k = 0; k <= i; ++k) {
            transcript_so_far += moves[k];
        }
        std::cout << "After move " << (i + 1) << " (" << transcript_so_far << "):" << std::endl;
        // print_board(board, "  Current board");
        if (test_book.contain(&board)) {
            // int trans_idx;
            // Board repr = representative_board(board, &trans_idx);
            // print_board(repr, "  Representative board (trans=" + std::to_string(trans_idx) + ")");
            Book_elem elem = test_book.get(board);
            std::cout << "  Value: " << (int)elem.value 
                      << ", Level: " << (int)elem.level 
                      << ", N_lines: " << elem.n_lines << std::endl;
            
            // Check if next move exists in links/leaf
            int next_cell = -1;
            if (i + 1 < moves.size()) {
                next_cell = move_string_to_cell(moves[i + 1]);
            }
            
            // Display leaf in original orientation
            int orig_leaf_move = elem.leaf.move;
            if (elem.leaf.move >= 0 && elem.leaf.move < HW2) {
                orig_leaf_move = convert_coord_from_representative_board(elem.leaf.move, elem.trans_idx);
            }
            std::cout << "  Leaf: " << cell_to_move_string(orig_leaf_move);
            // if (elem.leaf.move >= 0 && elem.leaf.move < HW2) {
            //     std::cout << " [repr: " << cell_to_move_string(elem.leaf.move) << "]";
            // }
            std::cout << " (value: " << (int)elem.leaf.value 
                      << ", level: " << (int)elem.leaf.level << ")" << std::endl;
            if (elem.leaf.move != MOVE_UNDEFINED && next_cell == orig_leaf_move) {
                std::cout << "  *** NEXT MOVE IS LEAF ***" << std::endl;
            }
            
            // Display links in original orientation
            if (!elem.links.empty()) {
                std::cout << "  Links (" << elem.links.size() << "):";
                bool next_found_in_links = false;
                for (size_t j = 0; j < elem.links.size(); ++j) {
                    if (j > 0) std::cout << ",";
                    int stored_link_move = elem.links[j].move; // This is in representative board coordinates
                    int orig_link_move = stored_link_move;
                    if (stored_link_move >= 0 && stored_link_move < HW2) {
                        orig_link_move = convert_coord_from_representative_board(stored_link_move, elem.trans_idx);
                    }
                    std::cout << " " << cell_to_move_string(orig_link_move);
                    // if (stored_link_move >= 0 && stored_link_move < HW2) {
                    //     std::cout << "[repr:" << cell_to_move_string(stored_link_move) << "]";
                    // }
                    std::cout << "(" << (int)elem.links[j].value << ")";
                    if (next_cell == orig_link_move) {
                        std::cout << "***";
                        next_found_in_links = true;
                    }
                }
                std::cout << std::endl;
                if (next_found_in_links) {
                    std::cout << "  *** NEXT MOVE FOUND IN LINKS ***" << std::endl;
                }
            }
        } else {
            std::cout << "  [POSITION NOT FOUND]" << std::endl;
            all_found = false;
        }
        std::cout << std::endl;
    }

    return all_found;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <edax_book_file.dat> <line1> [line2] [line3] ..." << std::endl;
        std::cerr << "Example: " << argv[0] << " book.dat f5d6c3d3 f5f6e6f4" << std::endl;
        std::cerr << "  Each line should be a sequence of moves like 'f5d6c3d3'" << std::endl;
        return 1;
    }

    // Load Edax book
    std::string book_file = argv[1];
    std::cerr << "Loading Edax book: " << book_file << std::endl;
    
    Book test_book;
    if (!test_book.import_file_edax(book_file, true)) {
        std::cerr << "[ERROR] Failed to load book file: " << book_file << std::endl;
        return 1;
    }
    
    std::cerr << "Book loaded successfully. Total boards: " << test_book.get_n_book() << std::endl;
    std::cerr << std::endl;

    // Check each line
    int total_lines = argc - 2;
    int found_lines = 0;
    
    for (int i = 2; i < argc; ++i) {
        std::string line_str = argv[i];
        
        std::cout << std::string(60, '=') << std::endl;
        std::cout << "Line " << (i - 1) << " / " << total_lines << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        bool result = check_single_line(test_book, line_str);
        
        if (result) {
            std::cout << "Result: All positions in this line are FOUND." << std::endl;
            found_lines++;
        } else {
            std::cout << "Result: Some positions are NOT FOUND in this line." << std::endl;
        }
        
        std::cout << std::endl;
    }

    // Final summary
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "FINAL SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Total lines checked: " << total_lines << std::endl;
    std::cout << "Lines completely found: " << found_lines << std::endl;
    std::cout << "Lines with missing positions: " << (total_lines - found_lines) << std::endl;

    return 0;
}