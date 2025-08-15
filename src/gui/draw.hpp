/*
    Egaroucid Project

    @file draw.hpp
        Drawing board / information
    @date 2021-2025
    @author Takuto Yamana
    @license GPL-3.0-or-later
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
        String level_info = language.get("common", "level") + U" " + Format(menu_elements.level);
        fonts.font(level_info).draw(12, Arg::center(INFO_SX + INFO_WIDTH / 2, up + height / 2));
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



struct ExplorerDrawResult {
    bool folderClicked = false;
    String clickedFolder;
    bool importClicked = false;
    int importIndex = -1;
    bool deleteClicked = false;
    int deleteIndex = -1;
};

template <class FontsT, class ColorsT, class ResourcesT>
inline ExplorerDrawResult DrawExplorerList(
    const std::vector<String>& folders_display,
    const std::vector<Game_abstract>& games,
    std::vector<Button>& import_buttons,
    std::vector<ImageButton>& delete_buttons,
    Scroll_manager& scroll_manager,
    bool showGames,
    int itemHeight,
    FontsT& fonts,
    ColorsT& colors,
    ResourcesT& resources
) {
    ExplorerDrawResult res;
    int sy = IMPORT_GAME_SY;
    int strt_idx_int = scroll_manager.get_strt_idx_int();
    if (strt_idx_int > 0) {
        fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, sy }, colors.white);
    }
    sy += 8;
    int total_rows = (int)folders_display.size() + (showGames ? (int)games.size() : 0);
    for (int row = strt_idx_int; row < std::min(total_rows, strt_idx_int + IMPORT_GAME_N_GAMES_ON_WINDOW); ++row) {
        Rect rect;
        rect.y = sy;
        rect.x = IMPORT_GAME_SX;
        rect.w = IMPORT_GAME_WIDTH;
        rect.h = itemHeight;
        if (row % 2) {
            rect.draw(colors.dark_green).drawFrame(1.0, colors.white);
        } else {
            rect.draw(colors.green).drawFrame(1.0, colors.white);
        }

        if (row < (int)folders_display.size()) {
            String fname = folders_display[row];
            double folder_icon_scale = (double)(rect.h - 2 * 10) / (double)resources.folder.height();
            resources.folder.scaled(folder_icon_scale).draw(Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + itemHeight / 2));
            fonts.font(fname).draw(15, Arg::leftCenter(IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10 + 30, sy + itemHeight / 2), colors.white);
            if (Rect(IMPORT_GAME_SX, sy, IMPORT_GAME_WIDTH, itemHeight).leftClicked()) {
                res.folderClicked = true;
                res.clickedFolder = fname;
                return res;
            }
    } else if (showGames) {
            int i = row - (int)folders_display.size();
            int winner = -1;
            if (games[i].black_score != GAME_DISCS_UNDEFINED && games[i].white_score != GAME_DISCS_UNDEFINED) {
                if (games[i].black_score > games[i].white_score) {
                    winner = IMPORT_GAME_WINNER_BLACK;
                } else if (games[i].black_score < games[i].white_score) {
                    winner = IMPORT_GAME_WINNER_WHITE;
                } else {
                    winner = IMPORT_GAME_WINNER_DRAW;
                }
            }
            delete_buttons[i].move(IMPORT_GAME_SX + 1, sy + 1);
            delete_buttons[i].draw();
            if (delete_buttons[i].clicked()) {
                res.deleteClicked = true;
                res.deleteIndex = i;
                return res;
            }
            String date = games[i].date.substr(0, 10).replace(U"_", U"/");
            fonts.font(date).draw(15, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, sy + 2, colors.white);
            Rect black_player_rect;
            black_player_rect.w = IMPORT_GAME_PLAYER_WIDTH;
            black_player_rect.h = IMPORT_GAME_PLAYER_HEIGHT;
            black_player_rect.y = sy + 1;
            black_player_rect.x = IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + IMPORT_GAME_DATE_WIDTH;
            if (winner == IMPORT_GAME_WINNER_BLACK) {
                black_player_rect.draw(colors.darkred);
            } else if (winner == IMPORT_GAME_WINNER_WHITE) {
                black_player_rect.draw(colors.darkblue);
            } else if (winner == IMPORT_GAME_WINNER_DRAW) {
                black_player_rect.draw(colors.chocolate);
            }
            int upper_center_y = black_player_rect.y + black_player_rect.h / 2;
            for (int font_size = 15; font_size >= 12; --font_size) {
                if (fonts.font(games[i].black_player).region(font_size, Vec2{0, 0}).w <= IMPORT_GAME_PLAYER_WIDTH - 4) {
                    fonts.font(games[i].black_player).draw(font_size, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH - 2, upper_center_y), colors.white);
                    break;
                } else if (font_size == 12) {
                    String player = games[i].black_player;
                    while (fonts.font(player).region(font_size, Vec2{0, 0}).w > IMPORT_GAME_PLAYER_WIDTH - 4) {
                        for (int i2 = 0; i2 < 4; ++i2) {
                            player.pop_back();
                        }
                        player += U"...";
                    }
                    fonts.font(player).draw(font_size, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH - 2, upper_center_y), colors.white);
                }
            }
            String black_score = U"??";
            String white_score = U"??";
            if (games[i].black_score != GAME_DISCS_UNDEFINED && games[i].white_score != GAME_DISCS_UNDEFINED) {
                black_score = Format(games[i].black_score);
                white_score = Format(games[i].white_score);
            }
            double hyphen_w = fonts.font(U"-").region(15, Vec2{0, 0}).w;
            fonts.font(black_score).draw(15, Arg::rightCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2 - hyphen_w / 2 - 1, upper_center_y), colors.white);
            fonts.font(U"-").draw(15, Arg::center(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2, upper_center_y), colors.white);
            fonts.font(white_score).draw(15, Arg::leftCenter(black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH / 2 + hyphen_w / 2 + 1, upper_center_y), colors.white);
            Rect white_player_rect;
            white_player_rect.w = IMPORT_GAME_PLAYER_WIDTH;
            white_player_rect.h = IMPORT_GAME_PLAYER_HEIGHT;
            white_player_rect.y = sy + 1;
            white_player_rect.x = black_player_rect.x + IMPORT_GAME_PLAYER_WIDTH + IMPORT_GAME_SCORE_WIDTH;
            if (winner == IMPORT_GAME_WINNER_BLACK) {
                white_player_rect.draw(colors.darkblue);
            } else if (winner == IMPORT_GAME_WINNER_WHITE) {
                white_player_rect.draw(colors.darkred);
            } else if (winner == IMPORT_GAME_WINNER_DRAW) {
                white_player_rect.draw(colors.chocolate);
            }
            for (int font_size = 15; font_size >= 12; --font_size) {
                if (fonts.font(games[i].white_player).region(font_size, Vec2{0, 0}).w <= IMPORT_GAME_PLAYER_WIDTH - 4) {
                    fonts.font(games[i].white_player).draw(font_size, Arg::leftCenter(white_player_rect.x + 2, upper_center_y), colors.white);
                    break;
                } else if (font_size == 12) {
                    String player = games[i].white_player;
                    while (fonts.font(player).region(font_size, Vec2{0, 0}).w > IMPORT_GAME_PLAYER_WIDTH - 4) {
                        for (int i2 = 0; i2 < 4; ++i2) {
                            player.pop_back();
                        }
                        player += U"...";
                    }
                    fonts.font(player).draw(font_size, Arg::leftCenter(white_player_rect.x + 2, upper_center_y), colors.white);
                }
            }
            fonts.font(games[i].memo).draw(12, IMPORT_GAME_SX + IMPORT_GAME_LEFT_MARGIN + 10, black_player_rect.y + black_player_rect.h, colors.white);
            import_buttons[i].move(IMPORT_GAME_BUTTON_SX, sy + IMPORT_GAME_BUTTON_SY);
            import_buttons[i].draw();
            if (import_buttons[i].clicked()) {
                res.importClicked = true;
                res.importIndex = i;
                return res;
            }
        }
        sy += itemHeight;
    }
    int total_rows2 = (int)folders_display.size() + (showGames ? (int)games.size() : 0);
    if (strt_idx_int + IMPORT_GAME_N_GAMES_ON_WINDOW < total_rows2) {
        fonts.font(U"︙").draw(15, Arg::bottomCenter = Vec2{ X_CENTER, 415}, colors.white);
    }
    scroll_manager.draw();
    scroll_manager.update();
    return res;
}