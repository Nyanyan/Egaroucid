/*
    Egaroucid Project

    @file board_image_vector.hpp
        Appearance of board image marks, shared by the editor and its SVG / EPS export
        Standard C++ only; the caller supplies glyph outlines for text.
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>
#include "board_image_edit.hpp"
#include "vector_image.hpp"

// Same values as BLACK / WHITE / VACANT of the engine.
constexpr int BOARD_IMAGE_DISC_BLACK = 0;
constexpr int BOARD_IMAGE_DISC_WHITE = 1;
constexpr int BOARD_IMAGE_DISC_VACANT = 2;

constexpr int BOARD_IMAGE_RECT_SIZE = 20;
constexpr int BOARD_IMAGE_STAR_SIZE = 15;
constexpr int BOARD_IMAGE_FRAME_WIDTH = 2;
constexpr double BOARD_IMAGE_BACKING_WIDTH = 4.0; // black frame behind outline marks on a white board
constexpr double BOARD_IMAGE_VALUE_FONT_SIZE = 22.0;
constexpr double BOARD_IMAGE_FLIP_RADIUS = 11.0;
constexpr double BOARD_IMAGE_FLIP_THICKNESS = 2.0;
constexpr double BOARD_IMAGE_FLIP_BACKING_THICKNESS = 4.0;
constexpr double BOARD_IMAGE_ARROW_WIDTH = 6.0;
constexpr double BOARD_IMAGE_ARROW_HEAD_WIDTH = 18.0;
constexpr double BOARD_IMAGE_ARROW_HEAD_LENGTH = 16.0;
constexpr double BOARD_IMAGE_ARROW_FRAME_WIDTH = 2.0;

constexpr int BOARD_IMAGE_INK_BLACK = 0;
constexpr int BOARD_IMAGE_INK_WHITE = 1;

// Colors of one cell mark. The editor and the vector export both draw from this.
struct Board_image_mark_style {
    int tool = BOARD_IMAGE_TOOL_RECT;
    bool outline = false;              // only the outline of the shape is drawn
    int ink = BOARD_IMAGE_INK_BLACK;   // shape, outline, flip, or value color
    bool halo = false;                 // thin frame in halo_ink around a filled shape
    int halo_ink = BOARD_IMAGE_INK_WHITE;
    bool backing = false;              // wide black frame behind an outline or flip mark
};

inline Board_image_mark_style board_image_mark_style(int mark, int disc, bool monochrome) {
    Board_image_mark_style result;
    // Flip and outline marks contrast with the disc; empty cells use white.
    const int contrast = disc == BOARD_IMAGE_DISC_WHITE ? BOARD_IMAGE_INK_BLACK : BOARD_IMAGE_INK_WHITE;
    // White marks on empty cells of a white (monochrome) board need a black edge.
    const bool white_board_cell = disc == BOARD_IMAGE_DISC_VACANT && monochrome;
    if (mark == BOARD_IMAGE_FLIP) {
        result.tool = BOARD_IMAGE_TOOL_FLIP;
        result.ink = contrast;
        result.backing = white_board_cell;
        return result;
    }
    if (mark == BOARD_IMAGE_VALUE) {
        result.tool = BOARD_IMAGE_TOOL_VALUE;
        result.ink = monochrome ? BOARD_IMAGE_INK_BLACK : BOARD_IMAGE_INK_WHITE;
        return result;
    }
    const auto [tool, style] = board_image_tool_from_mark(mark);
    result.tool = tool;
    result.outline = style == BOARD_IMAGE_STYLE_OUTLINE;
    result.ink = result.outline ? contrast : (style == BOARD_IMAGE_STYLE_BLACK ? BOARD_IMAGE_INK_BLACK : BOARD_IMAGE_INK_WHITE);
    result.halo = !result.outline && ((style == BOARD_IMAGE_STYLE_BLACK && disc == BOARD_IMAGE_DISC_BLACK) ||
        (style == BOARD_IMAGE_STYLE_WHITE && (disc == BOARD_IMAGE_DISC_WHITE || white_board_cell)));
    result.halo_ink = style == BOARD_IMAGE_STYLE_BLACK ? BOARD_IMAGE_INK_WHITE : BOARD_IMAGE_INK_BLACK;
    result.backing = result.outline && white_board_cell;
    return result;
}

// One arrow of the flip mark. Angles use the Siv3D convention (0 is up, clockwise).
struct Board_image_flip_half {
    double start_angle = 0.0;
    double angle = 0.0;
    Vector_point tip;
    Vector_point left;
    Vector_point right;
};

inline Board_image_flip_half board_image_flip_half(const Vector_point& center, int half, double thickness) {
    Board_image_flip_half result;
    result.start_angle = -VECTOR_IMAGE_PI * 0.5 + 0.22 + half * VECTOR_IMAGE_PI;
    result.angle = VECTOR_IMAGE_PI - 0.52;
    const double end = result.start_angle + result.angle;
    const Vector_point radial{std::sin(end), -std::cos(end)};
    const Vector_point tangent{std::cos(end), std::sin(end)};
    result.tip = center + radial * BOARD_IMAGE_FLIP_RADIUS + tangent * 3.0;
    const Vector_point base = result.tip - tangent * (5.0 + thickness);
    result.left = base + radial * (2.0 + thickness);
    result.right = base - radial * (2.0 + thickness);
    return result;
}

// Same vertices as Siv3D's Shape2D::Star(r, center).
inline std::vector<Vector_point> board_image_star_points(const Vector_point& center, double r) {
    constexpr double inner_scale = 0.38196601125010515; // 2 / (3 + sqrt(5))
    std::vector<Vector_point> points;
    for (int i = 0; i < 10; ++i) {
        const double radius = (i % 2) ? r * inner_scale : r;
        const double angle = i * (VECTOR_IMAGE_PI / 5.0);
        points.push_back({center.x + radius * std::sin(angle), center.y - radius * std::cos(angle)});
    }
    return points;
}

// Same vertices as Siv3D's Shape2D::Arrow(from, to, width, Vec2{head_width, head_length}).
inline std::vector<Vector_point> board_image_arrow_points(const Vector_point& from, const Vector_point& to,
    double width = BOARD_IMAGE_ARROW_WIDTH, double head_width = BOARD_IMAGE_ARROW_HEAD_WIDTH,
    double head_length = BOARD_IMAGE_ARROW_HEAD_LENGTH) {
    const double length = vector_length(to - from);
    if (length == 0.0 || width <= 0.0 || head_width <= 0.0 || head_length <= 0.0) {
        return {};
    }
    const double ratio = std::min(head_length, length) / head_length;
    const Vector_point direction = (to - from) * (1.0 / length);
    const Vector_point normal{direction.y, -direction.x};
    const Vector_point left = normal * (width * 0.5);
    const Vector_point gutter = direction * (head_length * ratio);
    const Vector_point edge = normal * (head_width * ratio * 0.5);
    return {from + left, to + left - gutter, to + left - gutter + edge, to,
        to - left - gutter - edge, to - left - gutter, from - left};
}

// Siv3D's th2 in polygon frames: the squared distance of duplicate points and the offset of
// the extra vertex at sharp corners.
constexpr double BOARD_IMAGE_FRAME_TINY = 0.01;

// Outline of Siv3D's Shape2D::drawFrame(thickness): the closed polygon edges are widened
// to both sides with miter joins. Like Siv3D, corners sharper than about 57 degrees get an
// extra vertex, which cuts their tip flat at half the thickness. The result is filled with
// the nonzero rule (outer ring and reversed inner ring).
inline Vector_path board_image_polygon_frame(const std::vector<Vector_point>& points, double thickness) {
    Vector_path path;
    if (points.size() < 3 || thickness <= 0.0) {
        return path;
    }
    std::vector<Vector_point> buffer{points[0]};
    for (size_t i = 1; i < points.size(); ++i) {
        const Vector_point d = points[i] - points[i - 1];
        if (vector_dot(d, d) >= BOARD_IMAGE_FRAME_TINY) {
            buffer.push_back(points[i]);
        }
    }
    if (buffer.size() >= 2) {
        const Vector_point d = buffer.back() - buffer.front();
        if (vector_dot(d, d) <= BOARD_IMAGE_FRAME_TINY) {
            buffer.pop_back();
        }
    }
    const size_t count = buffer.size();
    if (count < 3) {
        return path;
    }
    // Returns the extra point for a sharp corner, placed after (sign > 0) or before the corner.
    const auto sharp_corner = [](const Vector_point& back, const Vector_point& current, const Vector_point& next,
                                 double sign) -> std::optional<Vector_point> {
        const Vector_point v1 = vector_normalized(back - current);
        const Vector_point v2 = vector_normalized(next - current);
        if (vector_dot(v1, v2) <= 0.55) {
            return std::nullopt;
        }
        const Vector_point tangent = vector_normalized(vector_normalized(next - current) + vector_normalized(current - back));
        const double forward = vector_dot(tangent, next - current) >= 0.0 ? 1.0 : -1.0;
        return current + tangent * (forward * sign * BOARD_IMAGE_FRAME_TINY);
    };
    std::vector<Vector_point> ring{buffer[0]};
    for (size_t i = 1; i < count; ++i) {
        ring.push_back(buffer[i]);
        if (const auto extra = sharp_corner(buffer[i - 1], buffer[i], buffer[(i + 1) % count], 1.0)) {
            ring.push_back(*extra);
        }
    }
    if (const auto extra = sharp_corner(buffer[count - 1], buffer[0], buffer[1], -1.0)) {
        ring.push_back(*extra);
    }
    const size_t size = ring.size();
    const double half = thickness * 0.5;
    std::vector<Vector_point> side0(size), side1(size);
    for (size_t k = 0; k < size; ++k) {
        const Vector_point& p0 = ring[(k + size - 1) % size];
        const Vector_point& p1 = ring[k];
        const Vector_point& p2 = ring[(k + 1) % size];
        const Vector_point line = p1 - p0;
        const Vector_point normal = vector_normalized(Vector_point{-line.y, line.x});
        const Vector_point v = vector_normalized(p2 - p1) + vector_normalized(p1 - p0);
        const Vector_point tangent = vector_dot(v, v) > 0.001 ? vector_normalized(v) : vector_normalized(p2 - p0);
        const Vector_point miter{-tangent.y, tangent.x};
        const double length = half / vector_dot(miter, normal);
        side0[k] = p1 + miter * length;
        side1[k] = p1 - miter * length;
    }
    path.add_polygon(side0);
    std::reverse(side1.begin(), side1.end());
    path.add_polygon(side1);
    return path;
}

// Glyph outlines in the units of the font's base size, as given by Siv3D's Font::renderOutlines().
struct Board_image_glyph {
    double x_advance = 0.0;
    Vector_point offset;                             // OutlineGlyph::getOffset()
    std::vector<std::vector<Vector_point>> rings;    // relative to offset, y down
};

struct Board_image_text_outline {
    double base_size = 0.0;    // Font::fontSize()
    double line_height = 0.0;  // Font::height()
    std::vector<Board_image_glyph> glyphs;
};

using Board_image_text_provider = std::function<Board_image_text_outline(const std::string& utf8_text)>;

// Places glyphs like Siv3D's DrawableText::drawAt(size, center): the text box, whose width is
// the sum of the advances and whose height is the line height, is centered on the point.
inline Vector_path board_image_text_path(const Board_image_text_outline& text, double size, const Vector_point& center) {
    Vector_path path;
    if (text.base_size <= 0.0) {
        return path;
    }
    const double scale = size / text.base_size;
    double width = 0.0;
    for (const auto& glyph : text.glyphs) {
        width += glyph.x_advance * scale;
    }
    Vector_point pen{center.x - width * 0.5, center.y - text.line_height * scale * 0.5};
    for (const auto& glyph : text.glyphs) {
        const Vector_point origin = pen + glyph.offset * scale;
        for (const auto& ring : glyph.rings) {
            if (ring.size() < 3) {
                continue;
            }
            std::vector<Vector_point> points;
            points.reserve(ring.size());
            for (const auto& point : ring) {
                points.push_back(origin + point * scale);
            }
            path.add_polygon(points);
        }
        pen.x += glyph.x_advance * scale;
    }
    return path;
}

// Board geometry of the GUI (BOARD_SX, BOARD_SY, BOARD_CELL_SIZE, ...).
struct Board_image_vector_layout {
    double board_x = 40.0;
    double board_y = 60.0;
    double cell_size = 50.0;
    double coordinate_size = 20.0;
    double coordinate_font_size = 15.0;   // draw_empty_board()
    double column_label_lift = 2.0;       // draw_empty_board() draws a-h 2 px higher
    double coordinate_margin = 7.0;       // right/bottom margin of saved images with coordinates
    double frame_width = 10.0;
    double frame_radius = 23.0;           // inner corner radius of the outer frame
    double grid_width = 2.0;
    double dot_radius = 5.0;
    double disc_radius = 20.0;
    double disc_frame_width = 2.0;
};

struct Board_image_vector_palette {
    Vector_color green{36, 153, 114};
    Vector_color black{0, 0, 0};
    Vector_color white{255, 255, 255};
    Vector_color dark_gray{51, 51, 51};
};

struct Board_image_vector_options {
    bool include_coordinate = true;
    bool monochrome = false;
    bool dark_outer_frame = false;  // color mode only
};

struct Board_image_vector_region {
    double x = 0.0;
    double y = 0.0;
    double width = 0.0;
    double height = 0.0;
};

// The same area as the PNG saved by the editor.
inline Board_image_vector_region board_image_vector_region(const Board_image_vector_layout& layout, bool include_coordinate) {
    const double outer_size = layout.cell_size * 8 + layout.frame_width * 2;
    if (include_coordinate) {
        const double size = outer_size + layout.coordinate_size + layout.coordinate_margin;
        return {layout.board_x - layout.frame_width - layout.coordinate_size,
                layout.board_y - layout.frame_width - layout.coordinate_size, size, size};
    }
    return {layout.board_x - layout.frame_width, layout.board_y - layout.frame_width, outer_size, outer_size};
}

inline Vector_point board_image_vector_cell_center(const Board_image_vector_layout& layout, int cell) {
    const double half = std::floor(layout.cell_size / 2);
    return {layout.board_x + (cell % 8) * layout.cell_size + half, layout.board_y + (cell / 8) * layout.cell_size + half};
}

inline void board_image_vector_add_flip(Vector_image& image, const Vector_point& center, const Vector_color& color, double thickness) {
    for (int half = 0; half < 2; ++half) {
        const Board_image_flip_half flip = board_image_flip_half(center, half, thickness);
        Vector_path arc;
        // Circle::drawArc(start, angle, thickness / 2, thickness / 2) has flat ends.
        arc.arc(center, BOARD_IMAGE_FLIP_RADIUS, flip.start_angle - VECTOR_IMAGE_PI * 0.5, flip.angle);
        image.stroke(arc, color, thickness, Vector_line_cap::butt);
        Vector_path head;
        head.add_polygon({flip.tip, flip.left, flip.right});
        image.fill(head, color);
    }
}

// Builds the board image as drawn by the editor (Board_image::update() and draw_board()).
// discs[cell] is BOARD_IMAGE_DISC_*; cell 0 is a1 and cell 63 is h8.
inline Vector_image build_board_image_vector(const std::array<int, 64>& discs, const Board_image_edit& edit,
    const Board_image_vector_options& options, const Board_image_text_provider& text_outline,
    const Board_image_vector_layout& layout = {}, const Board_image_vector_palette& palette = {}) {
    const Board_image_vector_region region = board_image_vector_region(layout, options.include_coordinate);
    Vector_image image(region.width, region.height, region.x, region.y);
    const Vector_color line_color = options.monochrome ? palette.black : palette.dark_gray;
    const Vector_color frame_color = options.monochrome ? palette.black : (options.dark_outer_frame ? palette.dark_gray : palette.white);
    const auto ink = [&palette](int ink_id) { return ink_id == BOARD_IMAGE_INK_BLACK ? palette.black : palette.white; };
    const auto add_text = [&image, &text_outline](const std::string& text, double size, const Vector_point& center, const Vector_color& color) {
        if (!text_outline) {
            return;
        }
        const Vector_path path = board_image_text_path(text_outline(text), size, center);
        image.fill(path, color);
    };
    const double board_size = layout.cell_size * 8;
    const double outer_x = layout.board_x - layout.frame_width;
    const double outer_y = layout.board_y - layout.frame_width;
    const double outer_size = board_size + layout.frame_width * 2;
    const double outer_radius = layout.frame_radius + layout.frame_width;

    image.set_group("background");
    Vector_path background;
    if (options.include_coordinate) {
        background.add_rect(region.x, region.y, region.width, region.height);
    } else {
        // Ends under the middle of the opaque frame; outside the frame stays transparent like the PNG.
        const double inset = layout.frame_width * 0.5;
        background.add_round_rect(outer_x + inset, outer_y + inset, outer_size - inset * 2, outer_size - inset * 2, outer_radius - inset);
    }
    image.fill(background, options.monochrome ? palette.white : palette.green);

    if (options.include_coordinate) {
        image.set_group("coordinates");
        const double half = std::floor(layout.cell_size / 2);
        for (int i = 0; i < 8; ++i) {
            add_text(std::to_string(i + 1), layout.coordinate_font_size,
                {layout.board_x - layout.coordinate_size, layout.board_y + layout.cell_size * i + half}, line_color);
            add_text(std::string(1, static_cast<char>('a' + i)), layout.coordinate_font_size,
                {layout.board_x + layout.cell_size * i + half, layout.board_y - layout.coordinate_size - layout.column_label_lift}, line_color);
        }
    }

    image.set_group("grid");
    Vector_path grid;
    for (int i = 1; i < 8; ++i) {
        const double x = layout.board_x + layout.cell_size * i;
        const double y = layout.board_y + layout.cell_size * i;
        grid.add_line({x, layout.board_y}, {x, layout.board_y + board_size});
        grid.add_line({layout.board_x, y}, {layout.board_x + board_size, y});
    }
    image.stroke(grid, line_color, layout.grid_width, Vector_line_cap::square);
    Vector_path dots;
    for (const int x : {2, 6}) {
        for (const int y : {2, 6}) {
            dots.add_circle({layout.board_x + layout.cell_size * x, layout.board_y + layout.cell_size * y}, layout.dot_radius);
        }
    }
    image.fill(dots, line_color);

    image.set_group("discs");
    for (int cell = 0; cell < 64; ++cell) {
        if (discs[cell] != BOARD_IMAGE_DISC_BLACK && discs[cell] != BOARD_IMAGE_DISC_WHITE) {
            continue;
        }
        Vector_path disc;
        disc.add_circle(board_image_vector_cell_center(layout, cell), layout.disc_radius);
        if (discs[cell] == BOARD_IMAGE_DISC_BLACK) {
            image.fill(disc, palette.black);
        } else {
            image.fill(disc, palette.white);
            if (options.monochrome) {
                // Circle::drawFrame(BOARD_DISC_FRAME_WIDTH, 0): a ring inside the disc edge.
                Vector_path ring;
                ring.add_circle(board_image_vector_cell_center(layout, cell), layout.disc_radius - layout.disc_frame_width * 0.5);
                image.stroke(ring, palette.black, layout.disc_frame_width);
            }
        }
    }

    image.set_group("frame");
    // RoundRect::drawFrame(0, frame_width): between the board edge and the enlarged round rect.
    Vector_path frame;
    frame.add_round_rect(outer_x, outer_y, outer_size, outer_size, outer_radius);
    frame.add_round_rect(layout.board_x, layout.board_y, board_size, board_size, layout.frame_radius);
    image.fill(frame, frame_color, Vector_fill_rule::evenodd);

    image.set_group("marks");
    for (int cell = 0; cell < 64; ++cell) {
        const int mark = edit.marks[cell];
        if (mark == BOARD_IMAGE_NOMARK) {
            continue;
        }
        const Vector_point center = board_image_vector_cell_center(layout, cell);
        const Board_image_mark_style style = board_image_mark_style(mark, discs[cell], options.monochrome);
        const Vector_color color = ink(style.ink);
        const Vector_color halo_color = ink(style.halo_ink);
        if (style.tool == BOARD_IMAGE_TOOL_FLIP) {
            if (style.backing) {
                board_image_vector_add_flip(image, center, palette.black, BOARD_IMAGE_FLIP_BACKING_THICKNESS);
            }
            board_image_vector_add_flip(image, center, color, BOARD_IMAGE_FLIP_THICKNESS);
        } else if (style.tool == BOARD_IMAGE_TOOL_VALUE) {
            add_text(edit.values[cell] ? std::to_string(*edit.values[cell]) : std::string("..."), BOARD_IMAGE_VALUE_FONT_SIZE, center, color);
        } else if (style.tool == BOARD_IMAGE_TOOL_RECT || style.tool == BOARD_IMAGE_TOOL_CIRCLE) {
            // RectF / Circle drawFrame(w / 2, w / 2) is a stroke of width w centered on the edge.
            Vector_path shape;
            if (style.tool == BOARD_IMAGE_TOOL_RECT) {
                const double half = BOARD_IMAGE_RECT_SIZE / 2.0;
                shape.add_rect(center.x - half, center.y - half, BOARD_IMAGE_RECT_SIZE, BOARD_IMAGE_RECT_SIZE);
            } else {
                shape.add_circle(center, BOARD_IMAGE_RECT_SIZE / 2);
            }
            if (style.outline) {
                if (style.backing) {
                    image.stroke(shape, palette.black, BOARD_IMAGE_BACKING_WIDTH);
                }
                image.stroke(shape, color, BOARD_IMAGE_FRAME_WIDTH);
            } else {
                image.fill(shape, color);
                if (style.halo) {
                    image.stroke(shape, halo_color, BOARD_IMAGE_FRAME_WIDTH);
                }
            }
        } else if (style.tool == BOARD_IMAGE_TOOL_STAR) {
            const std::vector<Vector_point> star = board_image_star_points(center, BOARD_IMAGE_STAR_SIZE);
            if (style.outline) {
                if (style.backing) {
                    image.fill(board_image_polygon_frame(star, BOARD_IMAGE_BACKING_WIDTH), palette.black);
                }
                image.fill(board_image_polygon_frame(star, BOARD_IMAGE_FRAME_WIDTH), color);
            } else {
                Vector_path shape;
                shape.add_polygon(star);
                image.fill(shape, color);
                if (style.halo) {
                    image.fill(board_image_polygon_frame(star, BOARD_IMAGE_FRAME_WIDTH), halo_color);
                }
            }
        }
    }

    image.set_group("arrows");
    for (const Board_image_arrow& arrow : edit.arrows) {
        if (arrow.from < 0 || 64 <= arrow.from || arrow.to < 0 || 64 <= arrow.to) {
            continue;
        }
        const Vector_point from = board_image_vector_cell_center(layout, arrow.from);
        const Vector_point to = board_image_vector_cell_center(layout, arrow.to);
        if (vector_length(to - from) < 1.0) {
            continue;
        }
        const std::vector<Vector_point> points = board_image_arrow_points(from, to);
        Vector_path body;
        body.add_polygon(points);
        image.fill(body, palette.black);
        image.fill(board_image_polygon_frame(points, BOARD_IMAGE_ARROW_FRAME_WIDTH), palette.white);
    }
    return image;
}
