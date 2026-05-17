/*
    Egaroucid Project

    @file ai_loss_graph_setting_scene.hpp
        AI loss graph setting scene
    @date 2021-2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include "function/function_all.hpp"

constexpr int AI_LOSS_GRAPH_SCENE_GRAPH_SX = 50;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH_WIDTH = WINDOW_SIZE_X - AI_LOSS_GRAPH_SCENE_GRAPH_SX * 2;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH_HEIGHT = 130;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH1_SY = 90;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH2_SY = 270;
constexpr int AI_LOSS_GRAPH_SCENE_LABEL_FONT_SIZE = 20;
constexpr int AI_LOSS_GRAPH_SCENE_AXIS_FONT_SIZE = 10;
constexpr double AI_LOSS_GRAPH_SCENE_POINT_RADIUS = 3.0;
constexpr int AI_LOSS_GRAPH_DEFAULT_MAX_LOSS = 2;
constexpr int AI_LOSS_GRAPH_DEFAULT_LOSS_PERCENTAGE = 30;

class AI_loss_graph_setting : public App::Scene {
private:
    Button default_button;
    Button ok_button;
    int dragging_graph_idx;
    int dragging_point_idx;

public:
    AI_loss_graph_setting(const InitData& init) : IScene{ init } {
        set_scene_ime_enabled(false);
        default_button.init(GO_BACK_BUTTON_BACK_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "use_default"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        ok_button.init(GO_BACK_BUTTON_GO_SX, GO_BACK_BUTTON_SY, GO_BACK_BUTTON_WIDTH, GO_BACK_BUTTON_HEIGHT, GO_BACK_BUTTON_RADIUS, language.get("common", "ok"), 25, getData().fonts.font, getData().colors.white, getData().colors.black);
        dragging_graph_idx = -1;
        dragging_point_idx = -1;
    }

    void update() override {
        if (System::GetUserActions() & UserAction::CloseButtonClicked) {
            changeScene(U"Close", SCENE_FADE_TIME);
            return;
        }

        Scene::SetBackground(getData().colors.green);
        getData().fonts.font(language.get("ai_settings", "accept_ai_loss") + U" / " + language.get("settings", "settings")).draw(24, Arg::topCenter(X_CENTER, 12), getData().colors.white);

        update_graph(0, &getData().menu_elements.max_loss_by_move, language.get("ai_settings", "max_loss"), 0, AI_MAX_LOSS_INF, getData().colors.yellow, false);
        update_graph(1, &getData().menu_elements.loss_percentage_by_move, language.get("ai_settings", "loss_percentage"), 0, AI_LOSS_PERCENTAGE_INF, getData().colors.cyan, true);
        sync_legacy_scalar_values();

        if (!MouseL.pressed()) {
            dragging_graph_idx = -1;
            dragging_point_idx = -1;
        }

        default_button.draw();
        if (default_button.clicked()) {
            getData().menu_elements.max_loss_by_move.fill(AI_LOSS_GRAPH_DEFAULT_MAX_LOSS);
            getData().menu_elements.loss_percentage_by_move.fill(AI_LOSS_GRAPH_DEFAULT_LOSS_PERCENTAGE);
            sync_legacy_scalar_values();
        }

        ok_button.draw();
        if (ok_button.clicked() || KeyEnter.down() || KeyEscape.down()) {
            getData().graph_resources.need_init = false;
            changeScene(U"Main_scene", SCENE_FADE_TIME);
            return;
        }
    }

    void draw() const override {
    }

private:
    RectF get_graph_rect(int graph_idx) const {
        const int sy = (graph_idx == 0) ? AI_LOSS_GRAPH_SCENE_GRAPH1_SY : AI_LOSS_GRAPH_SCENE_GRAPH2_SY;
        return RectF(AI_LOSS_GRAPH_SCENE_GRAPH_SX, sy, AI_LOSS_GRAPH_SCENE_GRAPH_WIDTH, AI_LOSS_GRAPH_SCENE_GRAPH_HEIGHT);
    }

    double idx_to_x(const RectF& rect, int idx) const {
        if (AI_LOSS_GRAPH_POINT_COUNT <= 1) {
            return rect.center().x;
        }
        const double ratio = static_cast<double>(idx) / static_cast<double>(AI_LOSS_GRAPH_POINT_COUNT - 1);
        return rect.x + rect.w * ratio;
    }

    int x_to_idx(const RectF& rect, double x) const {
        const double ratio = std::clamp((x - rect.x) / rect.w, 0.0, 1.0);
        const int idx = static_cast<int>(std::round(ratio * (AI_LOSS_GRAPH_POINT_COUNT - 1)));
        return std::clamp(idx, 0, AI_LOSS_GRAPH_POINT_COUNT - 1);
    }

    double value_to_y(const RectF& rect, int value, int min_value, int max_value, bool is_percentage_graph) const {
        if (!is_percentage_graph) {
            int snapped_value = snap_ai_max_loss_value(value);
            int value_idx = 0;
            for (int i = 0; i < static_cast<int>(AI_MAX_LOSS_SNAP_VALUES.size()); ++i) {
                if (AI_MAX_LOSS_SNAP_VALUES[i] == snapped_value) {
                    value_idx = i;
                    break;
                }
            }
            int index_from_top = static_cast<int>(AI_MAX_LOSS_SNAP_VALUES.size()) - 1 - value_idx;
            if (AI_MAX_LOSS_SNAP_VALUES.size() <= 1) {
                return rect.y + rect.h;
            }
            double ratio = static_cast<double>(index_from_top) / static_cast<double>(AI_MAX_LOSS_SNAP_VALUES.size() - 1);
            return rect.y + rect.h * ratio;
        }
        if (max_value <= min_value) {
            return rect.y + rect.h;
        }
        const double ratio = std::clamp(static_cast<double>(value - min_value) / static_cast<double>(max_value - min_value), 0.0, 1.0);
        return rect.y + rect.h * (1.0 - ratio);
    }

    int y_to_value(const RectF& rect, double y, int min_value, int max_value, bool is_percentage_graph) const {
        if (!is_percentage_graph) {
            if (AI_MAX_LOSS_SNAP_VALUES.size() <= 1) {
                return AI_MAX_LOSS_SNAP_VALUES[0];
            }
            double top_to_bottom_ratio = std::clamp((y - rect.y) / rect.h, 0.0, 1.0);
            int index_from_top = static_cast<int>(std::round(top_to_bottom_ratio * static_cast<double>(AI_MAX_LOSS_SNAP_VALUES.size() - 1)));
            int value_idx = static_cast<int>(AI_MAX_LOSS_SNAP_VALUES.size()) - 1 - index_from_top;
            value_idx = std::clamp(value_idx, 0, static_cast<int>(AI_MAX_LOSS_SNAP_VALUES.size()) - 1);
            return AI_MAX_LOSS_SNAP_VALUES[value_idx];
        }
        if (max_value <= min_value) {
            return min_value;
        }
        const double ratio = std::clamp((rect.y + rect.h - y) / rect.h, 0.0, 1.0);
        return static_cast<int>(std::round(min_value + ratio * (max_value - min_value)));
    }

    int find_point_idx(const RectF& rect, const AI_loss_graph_values& values, int min_value, int max_value, bool is_percentage_graph) const {
        for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
            const double px = idx_to_x(rect, i);
            const double py = value_to_y(rect, values[i], min_value, max_value, is_percentage_graph);
            if (Circle(px, py, AI_LOSS_GRAPH_SCENE_POINT_RADIUS + 5.0).mouseOver()) {
                return i;
            }
        }
        return -1;
    }

    void sync_legacy_scalar_values() {
        getData().menu_elements.max_loss = getData().menu_elements.max_loss_by_move[0];
        getData().menu_elements.loss_percentage = getData().menu_elements.loss_percentage_by_move[0];
    }

    void update_graph(int graph_idx, AI_loss_graph_values* values, const String& label, int min_value, int max_value, const Color& line_color, bool is_percentage_graph) {
        RectF rect = get_graph_rect(graph_idx);
        rect.draw(ColorF(getData().colors.dark_green, 0.75)).drawFrame(1, getData().colors.white);
        getData().fonts.font(label).draw(AI_LOSS_GRAPH_SCENE_LABEL_FONT_SIZE, Arg::topLeft(rect.x + 6, rect.y - 28), getData().colors.white);

        if (is_percentage_graph) {
            for (int i = 0; i <= 4; ++i) {
                const double y = rect.y + rect.h * i / 4.0;
                const int value = static_cast<int>(std::round(max_value - (max_value - min_value) * (i / 4.0)));
                Line(rect.x, y, rect.x + rect.w, y).draw(1.0, ColorF(getData().colors.white, (i == 4) ? 0.35 : 0.18));
                getData().fonts.font(Format(value)).draw(AI_LOSS_GRAPH_SCENE_AXIS_FONT_SIZE, Arg::rightCenter(rect.x - 6, y), getData().colors.white);
            }
        } else {
            const int n_levels = static_cast<int>(AI_MAX_LOSS_SNAP_VALUES.size());
            for (int i = 0; i < n_levels; ++i) {
                const int value_idx = n_levels - 1 - i;
                const int value = AI_MAX_LOSS_SNAP_VALUES[value_idx];
                const double y = rect.y + rect.h * i / static_cast<double>(n_levels - 1);
                Line(rect.x, y, rect.x + rect.w, y).draw(1.0, ColorF(getData().colors.white, (value == 0) ? 0.35 : 0.18));
                getData().fonts.font(Format(value)).draw(AI_LOSS_GRAPH_SCENE_AXIS_FONT_SIZE, Arg::rightCenter(rect.x - 6, y), getData().colors.white);
            }
        }
        for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
            const int start_move = i * AI_LOSS_GRAPH_INTERVAL + 1;
            const int end_move = std::min(HW2 - 4, (i + 1) * AI_LOSS_GRAPH_INTERVAL);
            const double x = idx_to_x(rect, i);
            Line(x, rect.y, x, rect.y + rect.h).draw(1.0, ColorF(getData().colors.white, 0.10));
            const String range_label = Format(start_move) + U"~" + Format(end_move);
            getData().fonts.font(range_label).draw(AI_LOSS_GRAPH_SCENE_AXIS_FONT_SIZE, Arg::topCenter(x, rect.y + rect.h + 2), getData().colors.white);
        }

        if (dragging_graph_idx == -1 && MouseL.down()) {
            int target_point_idx = find_point_idx(rect, *values, min_value, max_value, is_percentage_graph);
            if (target_point_idx == -1 && rect.mouseOver()) {
                target_point_idx = x_to_idx(rect, Cursor::PosF().x);
            }
            if (target_point_idx != -1) {
                dragging_graph_idx = graph_idx;
                dragging_point_idx = target_point_idx;
            }
        }

        if (dragging_graph_idx == graph_idx && dragging_point_idx != -1 && MouseL.pressed()) {
            int value = y_to_value(rect, Cursor::PosF().y, min_value, max_value, is_percentage_graph);
            value = std::clamp(value, min_value, max_value);
            if (is_percentage_graph) {
                value = snap_ai_loss_percentage_value(value);
            } else {
                value = snap_ai_max_loss_value(value);
            }
            (*values)[dragging_point_idx] = value;
            Cursor::RequestStyle(CursorStyle::ResizeUpDown);
        }

        LineString curve;
        for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
            curve << Vec2(idx_to_x(rect, i), value_to_y(rect, (*values)[i], min_value, max_value, is_percentage_graph));
        }
        curve.draw(2.0, line_color);

        for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
            const double x = idx_to_x(rect, i);
            const double y = value_to_y(rect, (*values)[i], min_value, max_value, is_percentage_graph);
            const bool is_active = (dragging_graph_idx == graph_idx && dragging_point_idx == i);
            const double radius = is_active ? 5.0 : 3.5;
            Circle(x, y, radius).draw(line_color);
            if (is_active) {
                Circle(x, y, radius + 2.0).drawFrame(2.0, getData().colors.white);
                const double info_x = std::clamp(x, rect.x + 20.0, rect.x + rect.w - 20.0);
                const bool place_below = (y < rect.y + rect.h * 0.5);
                if (place_below) {
                    getData().fonts.font(Format((*values)[i])).draw(14, Arg::topCenter(info_x, y + 10), getData().colors.white);
                } else {
                    getData().fonts.font(Format((*values)[i])).draw(14, Arg::bottomCenter(info_x, y - 10), getData().colors.white);
                }
            }
        }

        if (rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }
    }
};
