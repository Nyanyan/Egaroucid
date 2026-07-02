/*
    Egaroucid Project

    @file input.hpp
        Input Functions
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include "const/gui_common.hpp"
#if SIV3D_PLATFORM(WINDOWS)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <imm.h>
#endif

constexpr int INPUT_STR_MAX_SIZE = 100000;


constexpr int TEXT_INPUT_FORMAT_NONE = -1;
constexpr int TEXT_INPUT_FORMAT_GGF = 0;
constexpr int TEXT_INPUT_FORMAT_OTHELLO_QUEST = 1;
constexpr int TEXT_INPUT_FORMAT_TRANSCRIPT = 2;
constexpr int TEXT_INPUT_FORMAT_TRANSCRIPT_FROM_THIS_POSITION = 3;
constexpr int TEXT_INPUT_FORMAT_BOARD = 4;
constexpr int TEXT_INPUT_FORMAT_GENERAL_BOARD_TRANSCRIPT = 5;

namespace gui_textarea_ime {

struct Deferred_ime_candidate_window_state {
    bool requested{ false };
    Vec2 pos{ 0.0, 0.0 };
    double editing_line_y{ 0.0 };
};

inline Deferred_ime_candidate_window_state deferred_state;
inline bool escape_suppressed_until_release{ false };

[[nodiscard]]
inline bool has_visible_ime_candidates() {
#if SIV3D_PLATFORM(WINDOWS)
    return (not Platform::Windows::TextInput::GetCandidateState().candidates.isEmpty());
#else
    return false;
#endif
}

inline void close_ime_candidate_window() {
#if SIV3D_PLATFORM(WINDOWS)
    const auto hwnd = static_cast<HWND>(Platform::Windows::Window::GetHWND());
    if (not hwnd) {
        return;
    }
    if (const auto himc = ImmGetContext(hwnd)) {
        ImmNotifyIME(himc, NI_CLOSECANDIDATE, 0, 0);
        ImmReleaseContext(hwnd, himc);
    }
#endif
}

[[nodiscard]]
inline bool consume_escape_for_ime_candidate_window() {
    if (escape_suppressed_until_release) {
        if (not KeyEscape.pressed()) {
            escape_suppressed_until_release = false;
        }
        return true;
    }

#if SIV3D_PLATFORM(WINDOWS)
    if (has_visible_ime_candidates() && KeyEscape.down()) {
        close_ime_candidate_window();
        deferred_state.requested = false;
        escape_suppressed_until_release = true;
        return true;
    }
#endif

    return false;
}

[[nodiscard]]
inline bool escape_down_for_scene_change() {
    return (not consume_escape_for_ime_candidate_window()) && KeyEscape.down();
}

[[nodiscard]]
inline bool escape_pressed_for_scene_change() {
    return (not consume_escape_for_ime_candidate_window()) && KeyEscape.pressed();
}

inline Vec2 calculate_editing_text_pos(
    const TextAreaEditState& text,
    const Vec2& pos,
    const SizeF& size
) {
    // Match SimpleGUI::TextArea internal text render region (Siv3D 0.6.x).
    constexpr double TEXTAREA_SCROLLBAR_WIDTH = 3.0;
    const RectF region = SimpleGUI::TextAreaRegion(pos, size);
    const RectF text_render_region = region.stretched(-2.0, -(6.0 + TEXTAREA_SCROLLBAR_WIDTH), -2.0, -8.0);
    const Vec2 default_pos = text_render_region.pos.movedBy(0.0, text.scrollY);

    if ((text.cursorPos == 0) || text.glyphs.isEmpty()) {
        return default_pos;
    }

    const size_t target_index = (text.cursorPos - 1);
    if (target_index >= text.glyphs.size()) {
        return default_pos;
    }

    for (const auto& clip_info : text.clipInfos) {
        if (clip_info.index != target_index) {
            continue;
        }

        const Glyph& glyph = text.glyphs[target_index];
        const bool is_line_feed = (glyph.codePoint == U'\n');
        const double caret_x = (clip_info.pos.x + (is_line_feed ? 0.0 : clip_info.clipRect.w));
        const double caret_y = (clip_info.pos.y - glyph.getOffset().y);
        return Vec2{ caret_x, caret_y };
    }

    return default_pos;
}

inline Vec2 calculate_editing_text_pos(
    const TextEditState& text,
    const Vec2& pos,
    double width
) {
    const size_t cursor_pos = Min<size_t>(text.cursorPos, text.text.size());
    const String text_before_cursor = text.text.substr(0, cursor_pos);
    const double text_x = pos.x + 8.0 + SimpleGUI::GetFont()(text_before_cursor).region().w;
    const double caret_x = Clamp(text_x, pos.x + 8.0, pos.x + Max(8.0, width - 8.0));
    return Vec2{ caret_x, pos.y };
}

inline void request_textarea_ime_candidate_window(
    const TextAreaEditState& text,
    const Vec2& pos,
    const SizeF& size
) {
    const Vec2 editing_text_pos = calculate_editing_text_pos(text, pos, size);
    const double candidate_y = (editing_text_pos.y + SimpleGUI::GetFont().height() + 2.0);
    deferred_state.requested = true;
    deferred_state.pos = Vec2{ editing_text_pos.x, candidate_y };
    deferred_state.editing_line_y = editing_text_pos.y;
}

inline void request_textbox_ime_candidate_window(
    const TextEditState& text,
    const Vec2& pos,
    double width
) {
    const Vec2 editing_text_pos = calculate_editing_text_pos(text, pos, width);
    const double candidate_y = editing_text_pos.y + SimpleGUI::GetFont().height() + 14.0;
    deferred_state.requested = true;
    deferred_state.pos = Vec2{ editing_text_pos.x, candidate_y };
    deferred_state.editing_line_y = editing_text_pos.y;
}

[[nodiscard]]
inline String replace_textbox_line_breaks_with_spaces(const String& source) {
    String replaced;
    replaced.reserve(source.size());
    for (size_t i = 0; i < source.size(); ++i) {
        const char32 ch = source[i];
        if (ch == U'\r') {
            replaced.push_back(U' ');
            if (((i + 1) < source.size()) && (source[i + 1] == U'\n')) {
                ++i;
            }
        } else if (ch == U'\n') {
            replaced.push_back(U' ');
        } else {
            replaced.push_back(ch);
        }
    }
    return replaced;
}

inline bool sanitize_textbox_line_breaks(TextEditState& text) {
    const size_t cursor_pos = Min<size_t>(text.cursorPos, text.text.size());
    const String before_cursor = replace_textbox_line_breaks_with_spaces(text.text.substr(0, cursor_pos));
    const String after_cursor = replace_textbox_line_breaks_with_spaces(text.text.substr(cursor_pos));
    const String sanitized = before_cursor + after_cursor;
    if (sanitized == text.text) {
        return false;
    }
    text.text = sanitized;
    text.cursorPos = before_cursor.size();
    return true;
}

inline bool insert_textbox_space_at_cursor(TextEditState& text, const Optional<size_t>& maxChars) {
    if (maxChars && (*maxChars <= text.text.size())) {
        return false;
    }

    const size_t cursor_pos = Min<size_t>(text.cursorPos, text.text.size());
    text.text = text.text.substr(0, cursor_pos) + U" " + text.text.substr(cursor_pos);
    text.cursorPos = cursor_pos + 1;
    return true;
}

inline Vec2 fit_ime_candidate_window_pos(const Vec2& pos, const double editing_line_y) {
#if SIV3D_PLATFORM(WINDOWS)
    constexpr double MARGIN = 2.0;
    const RectF bounds{ 0.0, 0.0, static_cast<double>(WINDOW_SIZE_X), static_cast<double>(WINDOW_SIZE_Y) };
    Vec2 adjusted = pos;
    RectF region = SimpleGUI::IMECandidateWindowRegion(adjusted);

    if ((region.w <= 0.0) && (region.h <= 0.0)) {
        return adjusted;
    }

    if ((bounds.w - MARGIN) < (region.x + region.w)) {
        adjusted.x -= ((region.x + region.w) - (bounds.w - MARGIN));
        adjusted.x = Max(MARGIN, adjusted.x);
        region = SimpleGUI::IMECandidateWindowRegion(adjusted);
    }

    if ((bounds.h - MARGIN) < (region.y + region.h)) {
        adjusted.y = Max(MARGIN, editing_line_y - region.h - MARGIN);
        region = SimpleGUI::IMECandidateWindowRegion(adjusted);
    }

    if ((bounds.h - MARGIN) < (region.y + region.h)) {
        adjusted.y = Max(MARGIN, bounds.h - region.h - MARGIN);
    }

    return adjusted;
#else
    (void)editing_line_y;
    return pos;
#endif
}

inline void flush_deferred_ime_candidate_window() {
    (void)consume_escape_for_ime_candidate_window();
    if (deferred_state.requested) {
#if SIV3D_PLATFORM(WINDOWS)
        SimpleGUI::IMECandidateWindow(
            fit_ime_candidate_window_pos(deferred_state.pos, deferred_state.editing_line_y)
        );
#endif
        deferred_state.requested = false;
    }
}

} // namespace gui_textarea_ime

inline bool text_area_with_ime_candidate_window(
    TextAreaEditState& text,
    const Vec2& pos,
    const SizeF& size = SizeF{ 200, 100 },
    size_t maxChars = SimpleGUI::PreferredTextAreaMaxChars,
    bool enabled = true
) {
    const bool changed = SimpleGUI::TextArea(text, pos, size, maxChars, enabled);
    if (enabled && text.active) {
        gui_textarea_ime::request_textarea_ime_candidate_window(text, pos, size);
    }
    return changed;
}

inline void set_path_textarea_text(TextAreaEditState& text, const String& value) {
    text.text = value;
    text.cursorPos = text.text.size();
    text.rebuildGlyphs();
}

inline bool remove_path_textarea_line_breaks(TextAreaEditState& text) {
    const size_t cursor_pos = Min<size_t>(text.cursorPos, text.text.size());
    const String before_cursor = text.text.substr(0, cursor_pos).replaced(U"\r", U"").replaced(U"\n", U"");
    const String after_cursor = text.text.substr(cursor_pos).replaced(U"\r", U"").replaced(U"\n", U"");
    const String sanitized = before_cursor + after_cursor;
    if (sanitized == text.text) {
        return false;
    }
    text.text = sanitized;
    text.cursorPos = before_cursor.size();
    text.rebuildGlyphs();
    return true;
}

inline bool path_text_area_with_ime_candidate_window(
    TextAreaEditState& text,
    const Vec2& pos,
    const SizeF& size = SizeF{ 600, 60 },
    size_t maxChars = SimpleGUI::PreferredTextAreaMaxChars,
    bool enabled = true
) {
    const bool was_active = text.active;
    bool changed = SimpleGUI::TextArea(text, pos, size, maxChars, enabled);
    changed = remove_path_textarea_line_breaks(text) || changed;

    if (enabled && was_active && KeyEnter.down()) {
        text.active = true;
    }

    if (enabled && text.active) {
        gui_textarea_ime::request_textarea_ime_candidate_window(text, pos, size);
    }
    return changed;
}

inline bool path_text_area_return_pressed(const TextAreaEditState& text) {
    return text.active && KeyEnter.down() && (not gui_textarea_ime::has_visible_ime_candidates());
}

inline bool text_box_with_ime_candidate_window(
    TextEditState& text,
    const Vec2& pos,
    double width = 200.0,
    const Optional<size_t>& maxChars = unspecified,
    bool enabled = true,
    bool replaceEnterWithSpace = true
) {
    const bool was_active = text.active;
    bool changed = SimpleGUI::TextBox(text, pos, width, maxChars, enabled);
    const bool line_break_replaced = gui_textarea_ime::sanitize_textbox_line_breaks(text);
    changed = changed || line_break_replaced;

    if (enabled && was_active && text.enterKey) {
        text.active = true;
        if (replaceEnterWithSpace && (not line_break_replaced)) {
            changed = gui_textarea_ime::insert_textbox_space_at_cursor(text, maxChars) || changed;
        }
    }

    if (enabled && text.active) {
        gui_textarea_ime::request_textbox_ime_candidate_window(text, pos, width);
    }
    return changed;
}

std::vector<History_elem> import_transcript_processing(std::vector<History_elem> n_history, History_elem strt_elem, std::string transcript, bool *failed) {
    *failed = false;
    Board h_bd = strt_elem.board;
    String transcript_str = Unicode::Widen(transcript).replace(U"\r", U"").replace(U"\n", U"").replace(U" ", U"").replace(U"\t", U"");
    if (transcript_str.size() % 2 != 0 && transcript_str.size() >= 120) {
        *failed = true;
        return n_history;
    }
    int y, x;
    uint64_t legal;
    Flip flip;
    History_elem history_elem;
    int player = strt_elem.player;
    bool passed = false;
    for (int i = 0; i < (int)transcript_str.size(); i += 2) {
        if (is_pass_like_str(transcript_str.narrow().substr(i, 2)) && passed) {
            continue;
        }
        x = (int)transcript_str[i] - (int)'a';
        if (x < 0 || HW <= x) {
            x = (int)transcript_str[i] - (int)'A';
            if (x < 0 || HW <= x) {
                *failed = true;
                break;
            }
        }
        y = (int)transcript_str[i + 1] - (int)'1';
        if (y < 0 || HW <= y) {
            *failed = true;
            break;
        }
        y = HW_M1 - y;
        x = HW_M1 - x;
        legal = h_bd.get_legal();
        if (1 & (legal >> (y * HW + x))) {
            calc_flip(&flip, &h_bd, y * HW + x);
            h_bd.move_board(&flip);
            player ^= 1;
            passed = false;
            if (h_bd.get_legal() == 0ULL) {
                h_bd.pass();
                player ^= 1;
                passed = true;
                if (h_bd.get_legal() == 0ULL) {
                    h_bd.pass();
                    player ^= 1;
                    if (i != transcript_str.size() - 2) {
                        *failed = true;
                        break;
                    }
                }
            }
        } else {
            *failed = true;
            break;
        }
        n_history.back().next_policy = y * HW + x;
        history_elem.set(h_bd, player, GRAPH_IGNORE_VALUE, -1, y * HW + x, -1, "");
        n_history.emplace_back(history_elem);
    }
    return n_history;
}


std::pair<Board, int> import_board_processing(std::string board_str, bool *failed) {
    *failed = false;
    String board_str_str = Unicode::Widen(board_str).replace(U"\r", U"").replace(U"\n", U"").replace(U" ", U"").replace(U"\t", U"");
    int bd_arr[HW2];
    Board bd;
    int player = -1;
    if (board_str_str.size() != HW2 + 1) {
        *failed = true;
    } else {
        for (int i = 0; i < HW2; ++i) {
            if (is_black_like_char(board_str_str[i])) {
                bd_arr[i] = BLACK;
            } else if (is_white_like_char(board_str_str[i])) {
                bd_arr[i] = WHITE;
            } else if (is_vacant_like_char(board_str_str[i])) {
                bd_arr[i] = VACANT;
            } else {
                *failed = true;
                break;
            }
        }
        if (is_black_like_char(board_str_str[HW2])) {
            player = BLACK;
        } else if (is_white_like_char(board_str_str[HW2])) {
            player = WHITE;
        } else {
            *failed = true;
        }
    }
    Board board;
    if (!(*failed)) {
        board.translate_from_arr(bd_arr, player);
        if (!board.is_end() && board.get_legal() == 0) {
            board.pass();
        }
    }
    return std::make_pair(board, player);
}


struct Game_import_t {
    std::vector<History_elem> history;
    String black_player_name;
    String white_player_name;
    int format{TEXT_INPUT_FORMAT_NONE};
};


Game_import_t import_ggf_processing(std::string ggf, bool *failed) {
    *failed = false;
    Game_import_t res;
    String ggf_str = Unicode::Widen(ggf).replace(U"\r", U"").replace(U"\n", U"").replace(U" ", U"").replace(U"\t", U"");
    if (ggf_str.size() > INPUT_STR_MAX_SIZE) {
        *failed = true;
        return res;
    }
    int board_start_idx = ggf_str.indexOf(U"BO[8");
    if (board_start_idx == std::string::npos) {
        *failed = true;
        return res;
    }
    board_start_idx += 4;
    if (ggf_str.size() < board_start_idx + 65) {
        *failed = true;
        return res;
    }
    std::string start_board_str = ggf_str.substr(board_start_idx, 65).narrow();
    std::cerr << "start board " << start_board_str << std::endl;
    std::pair<Board, int> board_player = import_board_processing(start_board_str, failed);
    if (*failed) {
        return res;
    }
    History_elem start_board;
    start_board.board = board_player.first;
    start_board.player = board_player.second;
    res.history.emplace_back(start_board);
    std::string transcript;
    int offset = board_start_idx + 65;
    while (true) {
        int coord_start_idx1 = ggf_str.indexOf(U"B[", offset);
        int coord_start_idx2 = ggf_str.indexOf(U"W[", offset);
        if (coord_start_idx1 == std::string::npos && coord_start_idx2 == std::string::npos) {
            break;
        }
        int coord_start_idx;
        if (coord_start_idx1 == std::string::npos) {
            coord_start_idx = coord_start_idx2;
        } else if (coord_start_idx2 == std::string::npos) {
            coord_start_idx = coord_start_idx1;
        } else {
            coord_start_idx = std::min(coord_start_idx1, coord_start_idx2);
        }
        coord_start_idx += 2;
        std::string coord = ggf_str.substr(coord_start_idx, 2).narrow();
        transcript += coord;
        offset = coord_start_idx + 2;
    }
    std::cerr << "import " << start_board_str << " " << transcript << std::endl;
    res.history = import_transcript_processing(res.history, start_board, transcript, failed);
    int player_idx_start = ggf_str.indexOf(U"PB[");
    if (player_idx_start != std::string::npos) {
        player_idx_start += 3;
        int player_idx_end = ggf_str.indexOf(U"]", player_idx_start);
        res.black_player_name = ggf_str.substr(player_idx_start, player_idx_end - player_idx_start);
    }
    player_idx_start = ggf_str.indexOf(U"PW[");
    if (player_idx_start != std::string::npos) {
        player_idx_start += 3;
        int player_idx_end = ggf_str.indexOf(U"]", player_idx_start);
        res.white_player_name = ggf_str.substr(player_idx_start, player_idx_end - player_idx_start);
    }
    return res;
}


Game_import_t import_othello_quest_processing(std::string s, bool *failed) {
    *failed = false;
    Game_import_t res;
    String str = Unicode::Widen(s).replace(U"\r", U"").replace(U"\n", U"").replace(U" ", U"").replace(U"\t", U"");
    if (str.size() > INPUT_STR_MAX_SIZE) {
        *failed = true;
        return res;
    }
    History_elem start_board;
    start_board.board.reset();
    start_board.player = BLACK;
    res.history.emplace_back(start_board); // always initial board for Othello Quest format
    std::string transcript;
    int offset = 0;
    while (true) {
        int coord_start_idx = str.indexOf(U"\"m\":\"", offset);
        if (coord_start_idx == std::string::npos) {
            break;
        }
        coord_start_idx += 5;
        std::string coord = str.substr(coord_start_idx, 2).narrow();
        if (coord != "-\"") {
            transcript += coord;
        }
        offset = coord_start_idx + 2;
    }
    std::cerr << "import " << transcript << std::endl;
    if (transcript.size() == 0) {
        *failed = true;
        return res;
    }
    res.history = import_transcript_processing(res.history, start_board, transcript, failed);
    int player_idx_offset = str.indexOf(U"\"players\":[{\"id\":\"");
    int player_idx_start = str.indexOf(U"\"name\":\"", player_idx_offset);
    if (player_idx_start != std::string::npos) {
        player_idx_start += 8;
        int player_idx_end = str.indexOf(U"\"", player_idx_start);
        res.black_player_name = str.substr(player_idx_start, player_idx_end - player_idx_start);
        player_idx_offset = player_idx_end;
    }
    player_idx_start = str.indexOf(U"\"name\":\"", player_idx_offset);
    if (player_idx_start != std::string::npos) {
        player_idx_start += 8;
        int player_idx_end = str.indexOf(U"\"", player_idx_start);
        res.white_player_name = str.substr(player_idx_start, player_idx_end - player_idx_start);
    }
    return res;
}

Game_import_t import_general_board_transcript_processing(std::string s, bool *failed) {
    *failed = true;
    Game_import_t res;
    String str = Unicode::Widen(s).replace(U"\r", U"").replace(U"\n", U"").replace(U" ", U"").replace(U"\t", U"");
    int len_str = str.size();
    if (len_str > INPUT_STR_MAX_SIZE) {
        *failed = true;
        return res;
    }
    std::string board_str = "---------------------------OX------XO---------------------------X";
    int transcript_str_start_idx = 0;
    for (int i = 0; i < len_str - 65; ++i) {
        bool is_board_format = true;
        for (int j = 0; j < 65; ++j) {
            if (!is_black_like_char(str[i + j]) && !is_white_like_char(str[i + j]) && !is_vacant_like_char(str[i + j])) {
                is_board_format = false;
                break;
            }
        }
        if (is_board_format) {
            board_str = str.substr(i, 65).narrow();
            transcript_str_start_idx = i + 65;
            std::cerr << "board_str " << board_str << std::endl;
            *failed = false;
            break;
        }
    }
    History_elem history_elem;
    // if (transcript_str_start_idx != 0) {
    //     history_elem.board.reset();
    //     history_elem.player = BLACK;
    //     res.history.emplace_back(history_elem);
    // }
    history_elem.board.from_str(board_str);
    history_elem.player = BLACK;
    if (is_white_like_char(board_str[64])) {
        history_elem.player = WHITE;
    }
    if (history_elem.board.get_legal() == 0) {
        history_elem.board.pass();
        history_elem.player ^= 1;
        if (history_elem.board.get_legal() == 0) {
            history_elem.board.pass();
            history_elem.player ^= 1;
        }
    }
    res.history.emplace_back(history_elem);
    Flip flip;
    for (int i = transcript_str_start_idx; i < len_str - 1; ++i) {
        if (is_coord_like_chars(str[i], str[i + 1])) {
            int coord = get_coord_from_chars(str[i], str[i + 1]);
            uint64_t coord_bit = 1ULL << coord;
            uint64_t legal = history_elem.board.get_legal();
            if (legal & coord_bit) {
                calc_flip(&flip, &history_elem.board, coord);
                history_elem.board.move_board(&flip);
                history_elem.player ^= 1;
                history_elem.policy = coord;
                res.history.back().next_policy = coord;
                if (history_elem.board.get_legal() == 0) {
                    history_elem.board.pass();
                    history_elem.player ^= 1;
                    if (history_elem.board.get_legal() == 0) {
                        history_elem.board.pass();
                        history_elem.player ^= 1;
                    }
                }
                res.history.emplace_back(history_elem);
                *failed = false;
            }
        }
    }
    return res;
}

Game_import_t import_any_format_processing(std::string s, std::vector<History_elem> history_until_now, History_elem start_history_elem, bool *failed) {
    bool f;

    // GGF
    {
        Game_import_t res_ggf = import_ggf_processing(s, &f);
        if (!f) {
            *failed = false;
            std::cerr << "imported as GGF" << std::endl;
            res_ggf.format = TEXT_INPUT_FORMAT_GGF;
            return res_ggf;
        }
    }

    // Othello Quest
    {
        Game_import_t res_othello_quest = import_othello_quest_processing(s, &f);
        if (!f) {
            *failed = false;
            std::cerr << "imported as Othello Quest" << std::endl;
            res_othello_quest.format = TEXT_INPUT_FORMAT_OTHELLO_QUEST;
            return res_othello_quest;
        }
    }

    // Transcript (from start)
    {
        History_elem history_elem;
        Board h_bd;
        history_elem.reset();
        Game_import_t res_transcript;
        res_transcript.history.emplace_back(history_elem);
        res_transcript.history = import_transcript_processing(res_transcript.history, history_elem, s, &f);
        if (!f) {
            *failed = false;
            std::cerr << "imported as Transcript" << std::endl;
            res_transcript.format = TEXT_INPUT_FORMAT_TRANSCRIPT;
            return res_transcript;
        }
    }

    // Transcript (from this board)
    {
        Game_import_t res_transcript_from_board;
        res_transcript_from_board.history = import_transcript_processing(history_until_now, start_history_elem, s, &f);
        if (!f) {
            *failed = false;
            std::cerr << "imported as Transcript from this Board" << std::endl;
            res_transcript_from_board.format = TEXT_INPUT_FORMAT_TRANSCRIPT_FROM_THIS_POSITION;
            return res_transcript_from_board;
        }
    }

    // Board
    {
        Game_import_t res_board;
        std::pair<Board, int> board_player = import_board_processing(s, &f);
        if (!f) {
            History_elem history_elem;
            // history_elem.reset();
            // res_board.history.emplace_back(history_elem);
            history_elem.board = board_player.first;
            history_elem.player = board_player.second;
            res_board.history.emplace_back(history_elem);
            *failed = false;
            std::cerr << "imported as Board String" << std::endl;
            res_board.format = TEXT_INPUT_FORMAT_BOARD;
            return res_board;
        }
    }

    // General Board + Transcript
    {
        Game_import_t res_board_transcript = import_general_board_transcript_processing(s, &f);
        if (!f) {
            *failed = false;
            std::cerr << "imported as General Board & Transcript" << std::endl;
            res_board_transcript.format = TEXT_INPUT_FORMAT_GENERAL_BOARD_TRANSCRIPT;
            return res_board_transcript;
        }
    }

    *failed = true;
    Game_import_t res_empty;
    return res_empty;
}

Game_import_t import_any_format_processing(std::string s, bool *failed) {
    std::vector<History_elem> history_until_now;
    History_elem start_history_elem;
    start_history_elem.board.player = 0;
    start_history_elem.board.opponent = 0;
    start_history_elem.player = BLACK;
    return import_any_format_processing(s, history_until_now, start_history_elem, failed);
}
