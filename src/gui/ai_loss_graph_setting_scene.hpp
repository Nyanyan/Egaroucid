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
#include "function/function_all.hpp"

constexpr int AI_LOSS_GRAPH_SCENE_GRAPH_SX = 50;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH_WIDTH = WINDOW_SIZE_X - AI_LOSS_GRAPH_SCENE_GRAPH_SX * 2;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH_HEIGHT = 130;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH1_SY = 90;
constexpr int AI_LOSS_GRAPH_SCENE_GRAPH2_SY = 270;
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

        update_graph(0, &getData().menu_elements.max_loss_by_move, language.get("ai_settings", "max_loss"), 0, AI_MAX_LOSS_INF, getData().colors.yellow);
        update_graph(1, &getData().menu_elements.loss_percentage_by_move, language.get("ai_settings", "loss_percentage"), 0, AI_LOSS_PERCENTAGE_INF, getData().colors.cyan);
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

    double value_to_y(const RectF& rect, int value, int min_value, int max_value) const {
        if (max_value <= min_value) {
            return rect.y + rect.h;
        }
        const double ratio = std::clamp(static_cast<double>(value - min_value) / static_cast<double>(max_value - min_value), 0.0, 1.0);
        return rect.y + rect.h * (1.0 - ratio);
    }

    int y_to_value(const RectF& rect, double y, int min_value, int max_value) const {
        if (max_value <= min_value) {
            return min_value;
        }
        const double ratio = std::clamp((rect.y + rect.h - y) / rect.h, 0.0, 1.0);
        return static_cast<int>(std::round(min_value + ratio * (max_value - min_value)));
    }

    int find_point_idx(const RectF& rect, const AI_loss_graph_values& values, int min_value, int max_value) const {
        for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
            const double px = idx_to_x(rect, i);
            const double py = value_to_y(rect, values[i], min_value, max_value);
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

    void update_graph(int graph_idx, AI_loss_graph_values* values, const String& label, int min_value, int max_value, const Color& line_color) {
        RectF rect = get_graph_rect(graph_idx);
        rect.rounded(8).draw(ColorF(getData().colors.dark_green, 0.75)).drawFrame(1, getData().colors.white);
        getData().fonts.font(label).draw(16, Arg::topLeft(rect.x + 2, rect.y - 26), getData().colors.white);

        for (int i = 0; i <= 4; ++i) {
            const double y = rect.y + rect.h * i / 4.0;
            const int value = static_cast<int>(std::round(max_value - (max_value - min_value) * (i / 4.0)));
            Line(rect.x, y, rect.x + rect.w, y).draw(1.0, ColorF(getData().colors.white, (i == 4) ? 0.35 : 0.18));
            getData().fonts.font(Format(value)).draw(9, Arg::rightCenter(rect.x - 6, y), getData().colors.white);
        }
        const std::array<int, 7> move_ticks = { 1, 10, 20, 30, 40, 50, 60 };
        for (int move : move_ticks) {
            const int idx = std::clamp(move, 1, AI_LOSS_GRAPH_POINT_COUNT) - 1;
            const double x = idx_to_x(rect, idx);
            Line(x, rect.y, x, rect.y + rect.h).draw(1.0, ColorF(getData().colors.white, 0.14));
            getData().fonts.font(Format(move)).draw(9, Arg::topCenter(x, rect.y + rect.h + 2), getData().colors.white);
        }

        if (dragging_graph_idx == -1 && MouseL.down()) {
            int target_point_idx = find_point_idx(rect, *values, min_value, max_value);
            if (target_point_idx == -1 && rect.mouseOver()) {
                target_point_idx = x_to_idx(rect, Cursor::PosF().x);
            }
            if (target_point_idx != -1) {
                dragging_graph_idx = graph_idx;
                dragging_point_idx = target_point_idx;
            }
        }

        if (dragging_graph_idx == graph_idx && dragging_point_idx != -1 && MouseL.pressed()) {
            int value = y_to_value(rect, Cursor::PosF().y, min_value, max_value);
            (*values)[dragging_point_idx] = std::clamp(value, min_value, max_value);
            Cursor::RequestStyle(CursorStyle::ResizeUpDown);
        }

        LineString curve;
        for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
            curve << Vec2(idx_to_x(rect, i), value_to_y(rect, (*values)[i], min_value, max_value));
        }
        curve.draw(2.0, line_color);

        for (int i = 0; i < AI_LOSS_GRAPH_POINT_COUNT; ++i) {
            const double x = idx_to_x(rect, i);
            const double y = value_to_y(rect, (*values)[i], min_value, max_value);
            const bool is_active = (dragging_graph_idx == graph_idx && dragging_point_idx == i);
            const double radius = is_active ? 5.0 : ((i % 10 == 9 || i == 0 || i + 1 == AI_LOSS_GRAPH_POINT_COUNT) ? 3.5 : AI_LOSS_GRAPH_SCENE_POINT_RADIUS);
            Circle(x, y, radius).draw(line_color);
            if (is_active) {
                Circle(x, y, radius + 2.0).drawFrame(2.0, getData().colors.white);
                getData().fonts.font(Format((*values)[i])).draw(14, Arg::bottomCenter(x, y - 10), getData().colors.white);
            }
        }

        if (rect.mouseOver()) {
            Cursor::RequestStyle(CursorStyle::Hand);
        }
    }
};
