/*
    Egaroucid Project

    @file draw.hpp
        Drawing board / information
    @date 2021-2023
    @author Takuto Yamana
    @license GPL-3.0 license
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"

void draw_board(Fonts fonts, Colors colors, History_elem history_elem) {
    String coord_x = U"abcdefgh";
    for (int i = 0; i < HW; ++i) {
        fonts.font_bold(i + 1).draw(15, Arg::center(BOARD_SX - BOARD_COORD_SIZE, BOARD_SY + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2), colors.dark_gray);
        fonts.font_bold(coord_x[i]).draw(15, Arg::center(BOARD_SX + BOARD_CELL_SIZE * i + BOARD_CELL_SIZE / 2, BOARD_SY - BOARD_COORD_SIZE - 2), colors.dark_gray);
    }
    for (int i = 0; i < HW_M1; ++i) {
        Line(BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY, BOARD_SX + BOARD_CELL_SIZE * (i + 1), BOARD_SY + BOARD_CELL_SIZE * HW).draw(BOARD_CELL_FRAME_WIDTH, colors.dark_gray);
        Line(BOARD_SX, BOARD_SY + BOARD_CELL_SIZE * (i + 1), BOARD_SX + BOARD_CELL_SIZE * HW, BOARD_SY + BOARD_CELL_SIZE * (i + 1)).draw(BOARD_CELL_FRAME_WIDTH, colors.dark_gray);
    }
    Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
    Circle(BOARD_SX + 2 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
    Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 2 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
    Circle(BOARD_SX + 6 * BOARD_CELL_SIZE, BOARD_SY + 6 * BOARD_CELL_SIZE, BOARD_DOT_SIZE).draw(colors.dark_gray);
    RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, colors.white);
    Flip flip;
    int board_arr[HW2];
    history_elem.board.translate_to_arr(board_arr, history_elem.player);
    for (int cell = 0; cell < HW2; ++cell) {
        int x = BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
        int y = BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2;
        if (board_arr[cell] == BLACK) {
            Circle(x, y, DISC_SIZE).draw(colors.black);
        }
        else if (board_arr[cell] == WHITE) {
            Circle(x, y, DISC_SIZE).draw(colors.white);
        }
    }
}

void draw_info(Colors colors, History_elem history_elem, Fonts fonts, Menu_elements menu_elements, bool pausing_in_pass) {
    RoundRect round_rect{ INFO_SX, INFO_SY, INFO_WIDTH, INFO_HEIGHT, INFO_RECT_RADIUS };
    round_rect.drawFrame(INFO_RECT_THICKNESS, colors.white);
    if (history_elem.board.get_legal()) {
        fonts.font(Format(history_elem.board.n_discs() - 3) + language.get("info", "moves")).draw(13, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + 5));
        String ai_human_str;
        bool ai_to_move = (menu_elements.ai_put_black && history_elem.player == BLACK) || (menu_elements.ai_put_white && history_elem.player == WHITE);
        bool black_to_move = history_elem.player == BLACK;
        if (ai_to_move ^ pausing_in_pass)
            ai_human_str = language.get("info", "ai");
        else
            ai_human_str = language.get("info", "human");
        if (black_to_move ^ pausing_in_pass) {
            fonts.font(language.get("info", "black") + U" " + ai_human_str).draw(18, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + 22));
        }
        else {
            fonts.font(language.get("info", "white") + U" " + ai_human_str).draw(18, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + 22));
        }
    }
    else {
        fonts.font(language.get("info", "game_end")).draw(20, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + 22));
    }
    Circle(INFO_SX + 70, INFO_SY + 60 + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.black);
    Circle(INFO_SX + INFO_WIDTH - 70, INFO_SY + 60 + INFO_DISC_RADIUS, INFO_DISC_RADIUS).draw(colors.white);
    int black_discs, white_discs;
    if (history_elem.player == BLACK) {
        black_discs = history_elem.board.count_player();
        white_discs = history_elem.board.count_opponent();
    }
    else {
        black_discs = history_elem.board.count_opponent();
        white_discs = history_elem.board.count_player();
    }
    fonts.font(black_discs).draw(20, Arg::leftCenter(INFO_SX + 100, INFO_SY + 60 + INFO_DISC_RADIUS));
    fonts.font(white_discs).draw(20, Arg::rightCenter(INFO_SX + INFO_WIDTH - 100, INFO_SY + 60 + INFO_DISC_RADIUS));
    Line(INFO_SX + INFO_WIDTH / 2, INFO_SY + 60, INFO_SX + INFO_WIDTH / 2, INFO_SY + 60 + INFO_DISC_RADIUS * 2).draw(2, colors.dark_gray);
    fonts.font(language.get("info", "opening_name") + U": " + Unicode::FromUTF8(history_elem.opening_name)).draw(12, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + 95));
    String level_info = language.get("common", "level") + U" " + Format(menu_elements.level) + U" (";
    if (menu_elements.level <= LIGHT_LEVEL) {
        level_info += language.get("info", "light");
    }
    else if (menu_elements.level <= STANDARD_MAX_LEVEL) {
        level_info += language.get("info", "standard");
    }
    else if (menu_elements.level <= PRAGMATIC_MAX_LEVEL) {
        level_info += language.get("info", "pragmatic");
    }
    else if (menu_elements.level <= ACCURATE_MAX_LEVEL) {
        level_info += language.get("info", "accurate");
    }
    else {
        level_info += language.get("info", "danger");
    }
    level_info += U")";
    fonts.font(level_info).draw(13, Arg::topCenter(INFO_SX + INFO_WIDTH / 2, INFO_SY + 115));
}