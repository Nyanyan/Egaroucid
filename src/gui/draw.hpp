/*
    Egaroucid Project

    @file draw.hpp
        Drawing board / information
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"

void draw_board(Fonts fonts, Colors colors, History_elem history_elem, bool monochrome) {
    String coord_x = U"abcdefgh";
    Color dark_gray_color = colors.dark_gray;
    if (monochrome) {
        dark_gray_color = colors.black;
    }
    for (int i = 0; i < HW; ++i) {
        fonts.font_bold(i + 1).draw(15, Arg::center(BOARD_SX - BOARD_COORD_SIZE, BOARD_SY + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2), dark_gray_color);
        fonts.font_bold(coord_x[i]).draw(15, Arg::center(BOARD_SX + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2, BOARD_SY - BOARD_COORD_SIZE - 2), dark_gray_color);
    }
    for (int i = 0; i < HW_M1; ++i) {
        Line(BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY, BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY + BOARD_CELL_SIZE * HW).draw(BOARD_CELL_FRAME_WIDTH, dark_gray_color);
        Line(BOARD_SX, BOARD_SY + BOARD_CELL_SIZE * (i + 1), BOARD_SX + BOARD_CELL_SIZE * HW, BOARD_SY + BOARD_CELL_SIZE * (i + 1)).draw(BOARD_CELL_FRAME_WIDTH, dark_gray_color);
    }
    Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(dark_gray_color);
    RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, colors.white);
    Flip flip;
    int board_arr[HW2];
    history_elem.board.translate_to_arr(board_arr, history_elem.player);
    for (int cell = 0; cell < HW2; ++cell) {
        int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
        int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
        if (board_arr[cell] == BLACK) {
            Circle(x, y, DISC_SIZE).draw(colors.black);
        } else if (board_arr[cell] == WHITE) {
            if (monochrome) {
                Circle(x, y, DISC_SIZE).draw(colors.white).drawFrame(BOARD_DISC_FRAME_WIDTH, 0, colors.black);
            } else {
                Circle(x, y, DISC_SIZE).draw(colors.white);
            }
        }
    }
}

void draw_board(Fonts fonts, Colors colors, History_elem history_elem) {
    draw_board(fonts, colors, history_elem, false);
}

void draw_info(Colors colors, History_elem history_elem, Fonts fonts, Menu_elements menu_elements, bool pausing_in_pass, std::string principal_variation) {
    RoundRect round_rect{ INFO_SX, INFO_SY, INFO_WIDTH, INFO_HEIGHT, INFO_RECT_RADIUS };
    round_rect.drawFrame(INFO_RECT_THICKNESS, colors.white);
    // 1st line
    int dy = 6;
    String moves_line;
    if (history_elem.board.get_legal()) {
        moves_line = language.get("info", "ply") + Format(history_elem.board.n_discs() - 3) + language.get("info", "moves");
        bool black_to_move = history_elem.player == BLACK;
        if (black_to_move ^ pausing_in_pass) {
            moves_line += U" " + language.get("info", "black");
        } else {
            moves_line += U" " + language.get("info", "white");
        }
        bool ai_to_move = (menu_elements.ai_put_black && history_elem.player == BLACK) || (menu_elements.ai_put_white && history_elem.player == WHITE);
        if (ai_to_move ^ pausing_in_pass) {
            moves_line += U" (" + language.get("info", "ai") + U")";
        } else {
            moves_line += U" (" + language.get("info", "human") + U")";
        }
    } else {
        moves_line = language.get("info", "game_end");
    }
    fonts.font(moves_line).draw(15, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
    dy += 23;
    // 2nd line
    String opening_info = language.get("info", "opening_name") + U": ";
    if (menu_elements.show_opening_name) {
        opening_info += Unicode::FromUTF8(history_elem.opening_name);
    }
    fonts.font(opening_info).draw(12, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
    if (menu_elements.show_ai_focus) {
        dy += 24;
    } else {
        dy += 27;
    }
    // 3rd line
    int black_discs, white_discs;
    if (history_elem.player == BLACK) {
        black_discs = history_elem.board.count_player();
        white_discs = history_elem.board.count_opponent();
    } else {
        black_discs = history_elem.board.count_opponent();
        white_discs = history_elem.board.count_player();
    }
    Line(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy, INFO_SX + INFO_WIDTH / 2, INFO_SY + dy + INFO_DISC_RADIUS * 2).draw(2, colors.dark_gray);
    if (menu_elements.show_ai_focus) {
        Rect(INFO_SX + 10, INFO_SY + dy - 2, AI_FOCUS_INFO_COLOR_RECT_WIDTH, INFO_DISC_RADIUS * 2 + 4).draw(colors.black_advantage);
        Rect(INFO_SX + INFO_WIDTH - 10 - AI_FOCUS_INFO_COLOR_RECT_WIDTH, INFO_SY + dy - 2, AI_FOCUS_INFO_COLOR_RECT_WIDTH, INFO_DISC_RADIUS * 2 + 4).draw(colors.white_advantage);
        fonts.font(black_discs).draw(20, Arg::center(INFO_SX + 138, INFO_SY + dy + INFO_DISC_RADIUS));
        Circle(INFO_SX + 100, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.black);
        Circle(INFO_SX + INFO_WIDTH - 100, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.white);
        fonts.font(language.get("info", "black_advantage")).draw(12, Arg::center(INFO_SX + 10 + (AI_FOCUS_INFO_COLOR_RECT_WIDTH - INFO_DISC_RADIUS * 2 - 6) / 2, INFO_SY + dy + INFO_DISC_RADIUS), colors.black);
        fonts.font(language.get("info", "white_advantage")).draw(12, Arg::center(INFO_SX + INFO_WIDTH - 10 - (AI_FOCUS_INFO_COLOR_RECT_WIDTH - INFO_DISC_RADIUS * 2 - 6) / 2, INFO_SY + dy + INFO_DISC_RADIUS), colors.black);
        fonts.font(white_discs).draw(20, Arg::center(INFO_SX + INFO_WIDTH - 138, INFO_SY + dy + INFO_DISC_RADIUS));
        dy += 28;
    } else {
        Circle(INFO_SX + 70, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.black);
        Circle(INFO_SX + INFO_WIDTH - 70, INFO_SY + dy + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.white);
        fonts.font(black_discs).draw(20, Arg::center(INFO_SX + 110, INFO_SY + dy + INFO_DISC_RADIUS));
        fonts.font(white_discs).draw(20, Arg::center(INFO_SX + INFO_WIDTH - 110, INFO_SY + dy + INFO_DISC_RADIUS));
        dy += 30;
    }
    // 4th line
    if (menu_elements.show_ai_focus) {
        const double linewidth = 3;
        const double width = AI_FOCUS_INFO_COLOR_RECT_WIDTH - linewidth;
        const double height = 20;
        const double lleft = INFO_SX + 10 + linewidth / 2;
        const double rright = INFO_SX + INFO_WIDTH - 10 - AI_FOCUS_INFO_COLOR_RECT_WIDTH + linewidth / 2 + width;
        const double up = INFO_SY + dy - 2 + linewidth / 2;
        Line{ lleft, up, lleft + width / 2, up }.draw(LineStyle::SquareDot, linewidth, colors.blue);
        Line{ lleft, up + height, lleft + width / 2, up + height }.draw(LineStyle::SquareDot, linewidth, colors.blue);
        Line{ lleft, up, lleft, up + height }.draw(LineStyle::SquareDot, linewidth, colors.blue);
        Line{ lleft + width, up, lleft + width, up + height }.draw(linewidth, colors.blue);
        Line{ lleft + width / 2, up, lleft + width, up }.draw(linewidth, colors.blue);
        Line{ lleft + width / 2, up + height, lleft + width, up + height }.draw(linewidth, colors.blue);
        fonts.font(language.get("info", "good_point")).draw(12, Arg::center(lleft + width / 2, up + height / 2));
        Line{ rright, up, rright - width / 2, up }.draw(LineStyle::SquareDot, linewidth, colors.red);
        Line{ rright, up + height, rright - width / 2, up + height }.draw(LineStyle::SquareDot, linewidth, colors.red);
        Line{ rright, up, rright, up + height }.draw(LineStyle::SquareDot, linewidth, colors.red);
        Line{ rright - width, up, rright - width, up + height }.draw(linewidth, colors.red);
        Line{ rright - width / 2, up, rright - width, up }.draw(linewidth, colors.red);
        Line{ rright - width / 2, up + height, rright - width, up + height }.draw(linewidth, colors.red);
        fonts.font(language.get("info", "bad_point")).draw(12, Arg::center(rright - width / 2, up + height / 2));
        dy += 23;
    } else {
        String level_info = language.get("common", "level") + U" " + Format(menu_elements.level) + U" (";
        if (menu_elements.level <= LIGHT_LEVEL) {
            level_info += language.get("info", "light");
        } else if (menu_elements.level <= STANDARD_MAX_LEVEL) {
            level_info += language.get("info", "standard");
        } else if (menu_elements.level <= PRAGMATIC_MAX_LEVEL) {
            level_info += language.get("info", "pragmatic");
        } else if (menu_elements.level <= ACCURATE_MAX_LEVEL) {
            level_info += language.get("info", "accurate");
        } else {
            level_info += language.get("info", "danger");
        }
        level_info += U")";
        fonts.font(level_info).draw(12, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
        dy += 18;
    }
    // 5th line
    String pv_info = language.get("info", "principal_variation") + U": ";
    String pv_info2 = U"";
    bool use_second_line = false;
    if (menu_elements.show_principal_variation) {
        if (principal_variation.size() > 15 * 2) { // 15 moves and more?
            int center = principal_variation.size() / 2 / 2 * 2;
            if (center > 18 * 2) {
                center = 18 * 2;
            }
            std::string pv1 = principal_variation.substr(0, center);
            std::string pv2 = principal_variation.substr(center);
            pv_info += Unicode::Widen(pv1);
            pv_info2 = Unicode::Widen(pv2);
            use_second_line = true;
        } else {
            pv_info += Unicode::Widen(principal_variation);
        }
    }
    if (use_second_line) {
        fonts.font(pv_info).draw(11, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy));
        fonts.font(pv_info2).draw(11, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + dy + 12));
    } else {
        fonts.font(pv_info).draw(13, Arg::center(INFO_SX + INFO_WIDTH / 2, ((INFO_SY + dy) + (INFO_SY + INFO_HEIGHT - INFO_RECT_THICKNESS / 2)) / 2));
    }
}