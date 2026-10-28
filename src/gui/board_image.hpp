/*
    Egaroucid Project
    @file board_image.hpp
        Edit board image
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <iostream>
#include <future>
#include "./../engine/engine_all.hpp"
#include "function/function_all.hpp"
#include "draw.hpp"
#include "screen_shot.hpp"

constexpr int BOARD_IMAGE_MARK_DELETED = -1;
constexpr int BOARD_IMAGE_NOT_CLICKED = -2;
constexpr int BOARD_IMAGE_RECT_SIZE = 20;
constexpr int BOARD_IMAGE_STAR_SIZE = 15;
constexpr int BOARD_IMAGE_FRAME_WIDTH = 2;

class Board_image : public App::Scene {
private:
    FrameButton back_button;
    Button save_image_button;
    FrameButton clear_button;
    Radio_button color_radio;
    Radio_button outer_frame_color_radio;
    Radio_button coordinate_radio;
    Board_image_edit edit;
    Board_image_arrow_input arrow_input;
    std::array<int, HW2> last_marked;
    std::future<int> eval_futures[HW2];
    bool taking_screen_shot = false;
    bool erasing_arrow = false;
    int selected_mark;

    static void draw_outline_action(const FrameButton& button) {
        button.rect.drawFrame(button.frame_width, 0, button.frame_color);
        button.font(button.str).drawAt(button.font_size, button.rect.x + button.rect.w / 2,
            button.rect.y + button.rect.h / 2, button.font_color);
        if (button.rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }
    }

    Board_image_position position() const {
        const auto& history = getData().history_elem;
        return {history.board.player, history.board.opponent, history.player};
    }

    void save_edit() {
        getData().graph_resources.board_image_edits.save(position(), edit);
        arrow_input.cancel();
    }

    static Vec2 cell_center(int cell) {
        return Vec2{BOARD_SX + (cell % HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2,
                    BOARD_SY + (cell / HW) * BOARD_CELL_SIZE + BOARD_CELL_SIZE / 2};
    }

    static int cursor_cell() {
        const Vec2 cursor = Cursor::PosF();
        return board_image_cell_at(cursor.x - BOARD_SX, cursor.y - BOARD_SY, BOARD_CELL_SIZE);
    }

    static Shape2D arrow_shape(const Vec2& from, const Vec2& to) {
        return Shape2D::Arrow(from, to, 6, Vec2{18, 16});
    }

    static void draw_arrow(const Vec2& from, const Vec2& to) {
        if (from.distanceFrom(to) < 1.0) {
            return;
        }
        const auto arrow = arrow_shape(from, to);
        arrow.draw(Palette::Black);
        arrow.drawFrame(2, Palette::White);
    }

    bool erase_arrow_at_cursor() {
        for (size_t i = edit.arrows.size(); i > 0; --i) {
            const auto& arrow = edit.arrows[i - 1];
            const auto polygon = arrow_shape(cell_center(arrow.from), cell_center(arrow.to)).asPolygon();
            if (polygon.intersects(Circle(Cursor::PosF(), 2))) {
                edit.arrows.erase(edit.arrows.begin() + (i - 1));
                return true;
            }
        }
        return false;
    }

    void update_board_input() {
        if (selected_mark == BOARD_IMAGE_ARROW) {
            if (MouseL.down()) {
                arrow_input.click(cursor_cell(), &edit);
            }
            return;
        }
        if (!MouseL.pressed()) {
            erasing_arrow = false;
        }
        if (selected_mark == BOARD_IMAGE_NOMARK && MouseL.down()) {
            erasing_arrow = erase_arrow_at_cursor();
        }
        for (int cell = 0; cell < HW2; ++cell) {
            const Rect region(BOARD_SX + (cell % HW) * BOARD_CELL_SIZE,
                              BOARD_SY + (cell / HW) * BOARD_CELL_SIZE,
                              BOARD_CELL_SIZE, BOARD_CELL_SIZE);
            if (!erasing_arrow && region.leftPressed()) {
                const bool legal = (getData().history_elem.board.get_legal() >> (HW2_M1 - cell)) & 1ULL;
                if (selected_mark == BOARD_IMAGE_VALUE && !legal) {
                    continue;
                }
                if (edit.marks[cell] == selected_mark && last_marked[cell] != selected_mark) {
                    edit.marks[cell] = BOARD_IMAGE_NOMARK;
                    edit.values[cell].reset();
                    last_marked[cell] = BOARD_IMAGE_MARK_DELETED;
                } else if (last_marked[cell] != BOARD_IMAGE_MARK_DELETED && last_marked[cell] != selected_mark) {
                    edit.marks[cell] = selected_mark;
                    edit.values[cell].reset();
                    last_marked[cell] = selected_mark;
                }
            } else {
                last_marked[cell] = BOARD_IMAGE_NOT_CLICKED;
            }
        }
    }

    void update_evaluations() {
        const uint64_t legal = getData().history_elem.board.get_legal();
        for (int cell = 0; cell < HW2; ++cell) {
            if (eval_futures[cell].valid() && eval_futures[cell].wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                const int value = eval_futures[cell].get();
                if (edit.marks[cell] == BOARD_IMAGE_VALUE) {
                    edit.values[cell] = value;
                }
            }
            if (edit.marks[cell] == BOARD_IMAGE_VALUE && !edit.values[cell] && !eval_futures[cell].valid() &&
                ((legal >> (HW2_M1 - cell)) & 1ULL)) {
                const Board board = getData().history_elem.board;
                const int level = getData().menu_elements.level;
                const int policy = HW2_M1 - cell;
                eval_futures[cell] = std::async(std::launch::async, [board, level, policy]() {
                    Board b = board;
                    Flip flip;
                    calc_flip(&flip, &b, policy);
                    b.move_board(&flip);
                    return -ai(b, level, true, 0, false, false).value;
                });
            }
        }
    }

    static void draw_flip(const Vec2& center, const Color& color, double thickness = 2) {
        constexpr double radius = 11;
        for (int half = 0; half < 2; ++half) {
            const double start = -Math::HalfPi + 0.22 + half * Math::Pi;
            const double angle = Math::Pi - 0.52;
            Circle(center, radius).drawArc(start, angle, thickness / 2, thickness / 2, color);
            const double end = start + angle;
            const Vec2 radial{std::sin(end), -std::cos(end)};
            const Vec2 tangent{std::cos(end), std::sin(end)};
            const Vec2 tip = center + radial * radius + tangent * 3;
            const Vec2 base = tip - tangent * (5 + thickness);
            Triangle(tip, base + radial * (2 + thickness), base - radial * (2 + thickness)).draw(color);
        }
    }

    void draw_mark_symbol(int mark, const Vec2& center, int disc, bool monochrome, std::optional<int> value = std::nullopt) const {
        if (mark == BOARD_IMAGE_NOMARK) {
            return;
        }
        const auto& colors = getData().colors;
        const Color contrast = disc == WHITE ? colors.black : colors.white;
        if (mark == BOARD_IMAGE_FLIP) {
            if (disc == VACANT && monochrome) {
                draw_flip(center, colors.black, 4);
            }
            draw_flip(center, contrast);
            return;
        }
        if (mark == BOARD_IMAGE_VALUE) {
            const String text = value ? Format(*value) : U"...";
            getData().fonts.font_bold(text).drawAt(22, center, monochrome ? colors.black : colors.white);
            return;
        }
        const auto [tool, style] = board_image_tool_from_mark(mark);
        const bool outline = style == BOARD_IMAGE_STYLE_OUTLINE;
        const Color color = outline ? contrast : (style == BOARD_IMAGE_STYLE_BLACK ? colors.black : colors.white);
        const bool halo = !outline && ((style == BOARD_IMAGE_STYLE_BLACK && disc == BLACK) ||
            (style == BOARD_IMAGE_STYLE_WHITE && (disc == WHITE || (disc == VACANT && monochrome))));
        const Color halo_color = style == BOARD_IMAGE_STYLE_BLACK ? colors.white : colors.black;
        const bool outline_halo = outline && disc == VACANT && monochrome;
        if (tool == BOARD_IMAGE_TOOL_RECT) {
            const RectF rect(Arg::center = center, BOARD_IMAGE_RECT_SIZE, BOARD_IMAGE_RECT_SIZE);
            if (outline) {
                if (outline_halo) rect.drawFrame(2, 2, colors.black);
                rect.drawFrame(1, 1, color);
            } else {
                rect.draw(color);
                if (halo) rect.drawFrame(1, 1, halo_color);
            }
        } else if (tool == BOARD_IMAGE_TOOL_STAR) {
            const auto star = Shape2D::Star(BOARD_IMAGE_STAR_SIZE, center);
            if (outline) {
                if (outline_halo) star.drawFrame(4, colors.black);
                star.drawFrame(BOARD_IMAGE_FRAME_WIDTH, color);
            } else {
                star.draw(color);
                if (halo) star.drawFrame(BOARD_IMAGE_FRAME_WIDTH, halo_color);
            }
        } else if (tool == BOARD_IMAGE_TOOL_CIRCLE) {
            const Circle circle(center, BOARD_IMAGE_RECT_SIZE / 2);
            if (outline) {
                if (outline_halo) circle.drawFrame(2, 2, colors.black);
                circle.drawFrame(1, 1, color);
            } else {
                circle.draw(color);
                if (halo) circle.drawFrame(1, 1, halo_color);
            }
        }
    }

    void select_mark(int mark) {
        if (selected_mark != mark) {
            arrow_input.cancel();
            last_marked.fill(BOARD_IMAGE_NOT_CLICKED);
            selected_mark = mark;
        }
    }

    void draw_palette_tile(int mark, const RectF& region) {
        const bool hovered = region.mouseOver();
        if (hovered) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }
        const bool selected = mark == selected_mark;
        const s3d::RoundRect tile(region, 6);
        tile.drawFrame(selected ? 2 : 1, 0, selected ? getData().colors.white : getData().colors.dark_gray);
        const Vec2 center = region.center();
        if (mark == BOARD_IMAGE_ARROW) {
            draw_arrow(center - Vec2{16, 0}, center + Vec2{16, 0});
        } else if (mark == BOARD_IMAGE_NOMARK) {
            // The same eraser icon is used for every language.
            const Vec2 a = center + Vec2{-10, 0}, b = center + Vec2{0, -10};
            const Vec2 c = center + Vec2{12, 2}, d = center + Vec2{2, 12};
            Quad(a, b, (b + c) / 2, (a + d) / 2).draw(getData().colors.white);
            Quad(a, b, c, d).drawFrame(2, getData().colors.white);
        } else {
            // Palette and board share the exact same mark renderer.
            draw_mark_symbol(mark, center, VACANT, false, 0);
        }
    }

    void draw_mark_palette() {
        constexpr int shapes[3][3] = {
            {BOARD_IMAGE_BRECT, BOARD_IMAGE_WRECT, BOARD_IMAGE_ORECT},
            {BOARD_IMAGE_BSTAR, BOARD_IMAGE_WSTAR, BOARD_IMAGE_OSTAR},
            {BOARD_IMAGE_BCIRCLE, BOARD_IMAGE_WCIRCLE, BOARD_IMAGE_OCIRCLE}
        };
        constexpr double tile_width = (300.0 - 2 * 8) / 3;
        std::array<std::pair<int, RectF>, 13> tiles;
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                tiles[row * 3 + column] = {shapes[row][column], RectF(480 + column * (tile_width + 8), 90 + row * 44, tile_width, 36)};
            }
        }
        constexpr int tools[] = {BOARD_IMAGE_FLIP, BOARD_IMAGE_VALUE, BOARD_IMAGE_ARROW, BOARD_IMAGE_NOMARK};
        for (int i = 0; i < 4; ++i) {
            tiles[9 + i] = {tools[i], RectF(480 + i * 77, 222, 69, 36)};
        }
        // Resolve selection first so only the new tile is highlighted this frame.
        for (const auto& [mark, region] : tiles) {
            if (region.leftClicked()) {
                select_mark(mark);
            }
        }
        for (const auto& [mark, region] : tiles) {
            draw_palette_tile(mark, region);
        }
        const char* hints[] = {"flip", "value", "arrow_hint", "nomark"};
        for (int i = 0; i < 4; ++i) {
            if (tiles[9 + i].second.mouseOver() && selected_mark != BOARD_IMAGE_ARROW) {
                getData().fonts.font(language.get("board_image", hints[i])).draw(13, 480, 266, getData().colors.white);
            }
        }
        if (selected_mark == BOARD_IMAGE_ARROW) {
            getData().fonts.font(language.get("board_image", "arrow_hint")).draw(13, 480, 266, getData().colors.white);
        }
    }

    void draw_image_options() {
        Line(480, 288, 780, 288).draw(1, ColorF(getData().colors.white, 0.2));
        getData().fonts.font(language.get("board_image", "coordinate")).draw(15, 480, 300, getData().colors.white);
        getData().fonts.font(language.get("board_image", "color")).draw(15, 480, 338, getData().colors.white);
        coordinate_radio.draw();
        color_radio.draw();
        const bool color_enabled = color_radio.checked == BOARD_IMAGE_COLOR_DEFAULT;
        getData().fonts.font(language.get("board_image", "outer_frame_color")).draw(15, 480, 376, ColorF(getData().colors.white, color_enabled ? 1.0 : 0.35));
        if (color_enabled) {
            outer_frame_color_radio.draw();
        } else {
            for (const auto& option : outer_frame_color_radio.elems) {
                if (option.checked) {
                    option.circle.draw(ColorF(getData().colors.cyan, 0.35));
                }
                option.font(option.str).draw(option.font_size, Arg::leftCenter = Vec2{option.x + radio_button_margin, option.y}, ColorF(getData().colors.white, 0.35));
            }
        }
    }

public:
    Board_image(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(false);
        back_button.init(480, 436, 90, 44, 8, 1, language.get("common", "back"), 17, getData().fonts.font, Color(0, 0, 0, 0), getData().colors.white, getData().colors.white);
        clear_button.init(578, 436, 90, 44, 8, 1, language.get("board_image", "clear"), 17, getData().fonts.font, Color(0, 0, 0, 0), getData().colors.white, getData().colors.white);
        save_image_button.init(676, 436, 104, 44, 8, language.get("board_image", "save_image"), 17, getData().fonts.font, getData().colors.white, getData().colors.black);
        Radio_button_element elem;
        coordinate_radio.init();
        elem.init(580, 310, getData().fonts.font, 14, language.get("board_image", "include_coordinate"), true);
        coordinate_radio.push(elem);
        elem.init(676, 310, getData().fonts.font, 14, language.get("board_image", "exclude_coordinate"), false);
        coordinate_radio.push(elem);
        color_radio.init();
        elem.init(580, 348, getData().fonts.font, 14, language.get("board_image", "default"), true);
        color_radio.push(elem);
        elem.init(676, 348, getData().fonts.font, 14, language.get("board_image", "monochrome"), false);
        color_radio.push(elem);
        outer_frame_color_radio.init();
        elem.init(580, 386, getData().fonts.font, 14, language.get("board_image", "white"), true);
        outer_frame_color_radio.push(elem);
        elem.init(676, 386, getData().fonts.font, 14, language.get("board_image", "dark_gray"), false);
        outer_frame_color_radio.push(elem);

        selected_mark = getData().settings.board_image_mark;
        coordinate_radio.set_checked(getData().settings.board_image_coordinate);
        color_radio.set_checked(getData().settings.board_image_color);
        outer_frame_color_radio.set_checked(getData().settings.board_image_outer_frame_color);
        getData().graph_resources.prune_board_image_edits();
        edit = getData().graph_resources.board_image_edits.load(position());
        last_marked.fill(BOARD_IMAGE_NOT_CLICKED);
    }

    void update() override {
        if (taking_screen_shot) {
            take_board_image_screen_shot(getData().window_state.window_scale,
                getData().user_settings.screenshot_saving_dir,
                get_transcript(getData().graph_resources, getData().history_elem),
                coordinate_radio.checked == BOARD_IMAGE_COORDINATE_INCLUDE);
            taking_screen_shot = false;
        }
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            save_edit();
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }
        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("in_out", "board_image")).draw(25, 480, 20, getData().colors.white);
        getData().fonts.font(language.get("board_image", "mark")).draw(15, 480, 64, getData().colors.white);
        draw_mark_palette();
        draw_image_options();
        getData().settings.board_image_mark = selected_mark;
        getData().settings.board_image_coordinate = coordinate_radio.checked;
        getData().settings.board_image_color = color_radio.checked;
        getData().settings.board_image_outer_frame_color = outer_frame_color_radio.checked;
        Line(480, 420, 780, 420).draw(1, ColorF(getData().colors.white, 0.35));
        draw_outline_action(clear_button);
        draw_outline_action(back_button);
        save_image_button.draw();
        if (clear_button.clicked()) {
            edit.clear();
            arrow_input.cancel();
            last_marked.fill(BOARD_IMAGE_NOT_CLICKED);
        }
        if (back_button.clicked() || KeyEscape.pressed()) {
            save_edit();
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
            return;
        }
        if (save_image_button.clicked()) {
            arrow_input.cancel();
            taking_screen_shot = true;
            ScreenCapture::RequestCurrentFrame();
        }
        update_board_input();
        update_evaluations();
        const bool monochrome = color_radio.checked == BOARD_IMAGE_COLOR_MONOCHROME;
        const bool include_coordinate = coordinate_radio.checked == BOARD_IMAGE_COORDINATE_INCLUDE;
        if (monochrome) {
            Rect(BOARD_SX - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE - 1,
                 BOARD_SY - BOARD_ROUND_FRAME_WIDTH - BOARD_COORD_SIZE - 1,
                 BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 9,
                 BOARD_CELL_SIZE * HW + BOARD_ROUND_FRAME_WIDTH * 2 + BOARD_COORD_SIZE + 9).draw(getData().colors.white);
        }
        draw_board(getData().fonts, getData().colors, getData().history_elem, monochrome, include_coordinate);
        const Color frame_color = monochrome ? getData().colors.black :
            (outer_frame_color_radio.checked == BOARD_IMAGE_OUTER_FRAME_WHITE ? getData().colors.white : getData().colors.dark_gray);
        s3d::RoundRect(BOARD_SX, BOARD_SY, BOARD_CELL_SIZE * HW, BOARD_CELL_SIZE * HW, BOARD_ROUND_DIAMETER).drawFrame(0, BOARD_ROUND_FRAME_WIDTH, frame_color);
        int board_arr[HW2];
        getData().history_elem.board.translate_to_arr(board_arr, getData().history_elem.player);
        for (int cell = 0; cell < HW2; ++cell) {
            draw_mark_symbol(edit.marks[cell], cell_center(cell), board_arr[cell], monochrome, edit.values[cell]);
        }
        for (const auto& arrow : edit.arrows) {
            draw_arrow(cell_center(arrow.from), cell_center(arrow.to));
        }
        const int preview_cell = cursor_cell();
        if (arrow_input.start != -1 && preview_cell != -1) {
            draw_arrow(cell_center(arrow_input.start), cell_center(preview_cell));
        }
    }

    void draw() const override {}
};
