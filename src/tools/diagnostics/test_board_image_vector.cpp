// Tests for the SVG / EPS export of the board image editor (no Siv3D needed).
// Usage: test_board_image_vector [OUTPUT_DIRECTORY]
// With an output directory, sample SVG and EPS files are also written for viewers.
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include "../../gui/function/board_image_vector.hpp"

namespace {

bool approx(double a, double b, double tolerance = 1e-6) {
    return std::abs(a - b) <= tolerance;
}

bool approx(const Vector_point& a, const Vector_point& b, double tolerance = 1e-6) {
    return approx(a.x, b.x, tolerance) && approx(a.y, b.y, tolerance);
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

size_t count(const std::string& text, const std::string& pattern) {
    size_t result = 0;
    for (size_t pos = text.find(pattern); pos != std::string::npos; pos = text.find(pattern, pos + pattern.size())) {
        ++result;
    }
    return result;
}

std::vector<Vector_point> path_points(const Vector_path& path) {
    std::vector<Vector_point> points;
    for (const auto& command : path.commands()) {
        if (command.type == Vector_path::Command_type::move || command.type == Vector_path::Command_type::line) {
            points.push_back(command.point);
        }
    }
    return points;
}

// The decision code of Board_image::draw_mark_symbol() before it moved to board_image_mark_style().
Board_image_mark_style original_mark_style(int mark, int disc, bool monochrome) {
    const int black = BOARD_IMAGE_INK_BLACK, white = BOARD_IMAGE_INK_WHITE;
    const int contrast = disc == BOARD_IMAGE_DISC_WHITE ? black : white;
    Board_image_mark_style style;
    if (mark == BOARD_IMAGE_VALUE) {
        style.tool = BOARD_IMAGE_TOOL_VALUE;
        style.ink = monochrome ? black : white;
        return style;
    }
    const auto [tool, mark_style] = board_image_tool_from_mark(mark);
    const bool outline = mark_style == BOARD_IMAGE_STYLE_OUTLINE;
    style.tool = tool;
    style.outline = outline;
    style.ink = outline ? contrast : (mark_style == BOARD_IMAGE_STYLE_BLACK ? black : white);
    style.halo = !outline && ((mark_style == BOARD_IMAGE_STYLE_BLACK && disc == BOARD_IMAGE_DISC_BLACK) ||
        (mark_style == BOARD_IMAGE_STYLE_WHITE && (disc == BOARD_IMAGE_DISC_WHITE || (disc == BOARD_IMAGE_DISC_VACANT && monochrome))));
    style.halo_ink = mark_style == BOARD_IMAGE_STYLE_BLACK ? white : black;
    style.backing = outline && disc == BOARD_IMAGE_DISC_VACANT && monochrome;
    return style;
}

// Rectangular stand-ins for glyphs: advance 30, box 24 x 30 at offset (3, 10), line height 60.
Board_image_text_outline box_font(const std::string& text) {
    Board_image_text_outline outline;
    outline.base_size = 48;
    outline.line_height = 60;
    for (size_t i = 0; i < text.size(); ++i) {
        Board_image_glyph glyph;
        glyph.x_advance = 30;
        glyph.offset = {3, 10};
        glyph.rings.push_back({{0, 0}, {24, 0}, {24, 30}, {0, 30}});
        outline.glyphs.push_back(glyph);
    }
    return outline;
}

std::array<int, 64> initial_board() {
    std::array<int, 64> discs;
    discs.fill(BOARD_IMAGE_DISC_VACANT);
    discs[27] = discs[36] = BOARD_IMAGE_DISC_WHITE; // d4, e5
    discs[28] = discs[35] = BOARD_IMAGE_DISC_BLACK; // e4, d5
    return discs;
}

void write_file(const std::string& path, const std::string& data) {
    std::ofstream stream(path, std::ios::binary);
    stream << data;
    assert(stream.good());
}

void test_numbers() {
    assert(vector_image_number(0) == "0");
    assert(vector_image_number(447) == "447");
    assert(vector_image_number(1.5) == "1.5");
    assert(vector_image_number(0.1 + 0.2) == "0.3");
    assert(vector_image_number(-12.3456) == "-12.346");
    assert(vector_image_number(-0.0004) == "0");
    assert(vector_image_number(-0.0006) == "-0.001");
    assert(vector_image_number(1.0 / 3.0, 4) == "0.3333");
    assert(vector_image_number(NAN) == "0");
}

void test_writers() {
    Vector_image image(20, 10, 5, 7);
    image.set_title("a<b & \"c\"");
    image.set_group("first");
    Vector_path rect;
    rect.add_rect(5, 7, 10, 3);
    image.fill(rect, {36, 153, 114});
    Vector_path circle;
    circle.add_circle({10, 12}, 2);
    image.stroke(circle, {0, 0, 0}, 2);
    image.set_group("second");
    Vector_path lines;
    lines.add_line({5, 7}, {25, 17});
    image.stroke(lines, {51, 51, 51}, 1.5, Vector_line_cap::square, Vector_line_join::round);
    Vector_path ring;
    ring.add_round_rect(5, 7, 20, 10, 3);
    ring.add_rect(8, 9, 2, 2);
    image.fill(ring, {255, 255, 255}, Vector_fill_rule::evenodd);
    Vector_path empty;
    image.fill(empty, {1, 2, 3}); // ignored
    assert(image.shapes().size() == 4);

    const std::string svg = vector_image_to_svg(image);
    assert(svg.rfind("<?xml version=\"1.0\" encoding=\"UTF-8\"", 0) == 0);
    assert(svg.find("width=\"20\" height=\"10\" viewBox=\"0 0 20 10\"") != std::string::npos);
    assert(svg.find("<title>a&lt;b &amp; &quot;c&quot;</title>") != std::string::npos);
    assert(svg.find("<g id=\"first\">") < svg.find("<g id=\"second\">"));
    assert(count(svg, "</g>") == 2);
    assert(svg.find("<path d=\"M0 0L10 0L10 3L0 3Z\" fill=\"#249972\"/>") != std::string::npos);
    assert(svg.find("<circle cx=\"5\" cy=\"5\" r=\"2\" fill=\"none\" stroke=\"#000000\" stroke-width=\"2\"/>") != std::string::npos);
    assert(svg.find("stroke=\"#333333\" stroke-width=\"1.5\" stroke-linecap=\"square\" stroke-linejoin=\"round\"") != std::string::npos);
    // A round rect corner is one 90 degree arc; the sweep flag 1 is clockwise on screen.
    assert(svg.find("M3 0L17 0A3 3 0 0 1 20 3L20 7A3 3 0 0 1 17 10") != std::string::npos);
    assert(svg.find("fill-rule=\"evenodd\"") != std::string::npos);
    assert(svg.find("#010203") == std::string::npos);
    assert(svg.substr(svg.size() - 7) == "</svg>\n");

    const std::string eps = vector_image_to_eps(image);
    assert(eps.rfind("%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 20 10\n", 0) == 0);
    assert(eps.find("%%Title: a<b & \"c\"\n") != std::string::npos);
    assert(eps.find("0 10 translate 1 -1 scale\n") != std::string::npos);
    assert(eps.find("n\n0 0 m\n10 0 l\n10 3 l\n0 3 l\nh\n0.1412 0.6 0.4471 rg f\n") != std::string::npos);
    assert(eps.find("0 0 0 2 0 0 4 st\n") != std::string::npos);
    assert(eps.find("0.2 0.2 0.2 1.5 2 1 4 st\n") != std::string::npos);
    assert(eps.find("1 1 1 rg ef\n") != std::string::npos);
    // The circle is four Bezier quarters; the first one ends at the bottom (clockwise on screen).
    assert(eps.find("7 5 m\n7 6.105 6.105 7 5 7 c\n") != std::string::npos);
    assert(count(eps, " c\n") == 4 + 4);
    assert(ends_with(eps, "grestore\nend\nshowpage\n%%Trailer\n%%EOF\n"));
    for (const char c : eps) {
        assert(c == '\n' || (0x20 <= static_cast<unsigned char>(c) && static_cast<unsigned char>(c) <= 0x7e));
    }

    // Arcs over 90 degrees are split for SVG; negative sweeps are counterclockwise.
    Vector_path arc;
    arc.arc({0, 0}, 1, 0, -VECTOR_IMAGE_PI);
    assert(vector_image_svg_path_data(arc, 0, 0) == "M1 0A1 1 0 0 0 0 -1A1 1 0 0 0 -1 0");
    Vector_point center;
    double radius = 0;
    assert(!arc.is_circle(&center, &radius));
}

void test_mark_style() {
    for (int mark = 0; mark <= BOARD_IMAGE_MARK_MAX; ++mark) {
        if (mark == BOARD_IMAGE_NOMARK || mark == BOARD_IMAGE_ARROW || mark == BOARD_IMAGE_FLIP) {
            continue;
        }
        for (int disc = BOARD_IMAGE_DISC_BLACK; disc <= BOARD_IMAGE_DISC_VACANT; ++disc) {
            for (const bool monochrome : {false, true}) {
                const auto expected = original_mark_style(mark, disc, monochrome);
                const auto actual = board_image_mark_style(mark, disc, monochrome);
                assert(actual.tool == expected.tool && actual.outline == expected.outline && actual.ink == expected.ink);
                assert(actual.halo == expected.halo && actual.backing == expected.backing);
                assert(!actual.halo || actual.halo_ink == expected.halo_ink);
            }
        }
    }
    // Empty cells use white on green, black on monochrome; discs keep contrasting colors (Issue #662).
    constexpr int flip_inks[2][3] = {
        {BOARD_IMAGE_INK_WHITE, BOARD_IMAGE_INK_BLACK, BOARD_IMAGE_INK_WHITE},
        {BOARD_IMAGE_INK_WHITE, BOARD_IMAGE_INK_BLACK, BOARD_IMAGE_INK_BLACK}
    };
    for (int disc = BOARD_IMAGE_DISC_BLACK; disc <= BOARD_IMAGE_DISC_VACANT; ++disc) {
        for (const bool monochrome : {false, true}) {
            const auto style = board_image_mark_style(BOARD_IMAGE_FLIP, disc, monochrome);
            assert(style.tool == BOARD_IMAGE_TOOL_FLIP && style.ink == flip_inks[monochrome][disc]);
            assert(!style.backing && !style.halo && !style.outline);
        }
    }
    // Outline marks contrast with the disc; empty cells use white.
    assert(board_image_mark_style(BOARD_IMAGE_ORECT, BOARD_IMAGE_DISC_WHITE, false).ink == BOARD_IMAGE_INK_BLACK);
    assert(board_image_mark_style(BOARD_IMAGE_OSTAR, BOARD_IMAGE_DISC_BLACK, false).ink == BOARD_IMAGE_INK_WHITE);
    assert(board_image_mark_style(BOARD_IMAGE_WCIRCLE, BOARD_IMAGE_DISC_VACANT, true).halo);
    assert(!board_image_mark_style(BOARD_IMAGE_WCIRCLE, BOARD_IMAGE_DISC_VACANT, false).halo);
    assert(board_image_mark_style(BOARD_IMAGE_OCIRCLE, BOARD_IMAGE_DISC_VACANT, true).backing);
}

void test_geometry() {
    const auto star = board_image_star_points({100, 100}, 15);
    assert(star.size() == 10);
    assert(approx(star[0], {100, 85}));                               // the first tip points up
    assert(approx(star[5], {100, 100 + 15 * 0.38196601125010515}));   // inner vertex below the center
    assert(approx(star[2].x, 100 + 15 * std::sin(2 * VECTOR_IMAGE_PI / 5)));

    const auto arrow = board_image_arrow_points({0, 0}, {100, 0});
    const std::vector<Vector_point> expected_arrow{{0, -3}, {84, -3}, {84, -12}, {100, 0}, {84, 12}, {84, 3}, {0, 3}};
    assert(arrow.size() == expected_arrow.size());
    for (size_t i = 0; i < arrow.size(); ++i) {
        assert(approx(arrow[i], expected_arrow[i]));
    }
    const auto short_arrow = board_image_arrow_points({0, 0}, {8, 0}); // the head shrinks with short arrows
    assert(approx(short_arrow[1], {0, -3}) && approx(short_arrow[2], {0, -7.5}));
    assert(board_image_arrow_points({1, 1}, {1, 1}).empty());

    const auto flip = board_image_flip_half({0, 0}, 0, 2);
    const double end = -VECTOR_IMAGE_PI / 2 + 0.22 + VECTOR_IMAGE_PI - 0.52;
    assert(approx(flip.start_angle, -VECTOR_IMAGE_PI / 2 + 0.22) && approx(flip.angle, VECTOR_IMAGE_PI - 0.52));
    const Vector_point radial{std::sin(end), -std::cos(end)};
    const Vector_point tangent{std::cos(end), std::sin(end)};
    assert(approx(flip.tip, radial * 11 + tangent * 3));
    assert(approx(flip.left, flip.tip - tangent * 7 + radial * 4) && approx(flip.right, flip.tip - tangent * 7 - radial * 4));

    // A square has no sharp corner: plain miters on both sides.
    const Vector_path square = board_image_polygon_frame({{0, 0}, {10, 0}, {10, 10}, {0, 10}}, 2);
    const auto square_points = path_points(square);
    assert(square_points.size() == 8);
    const std::vector<Vector_point> expected_square{{1, 1}, {9, 1}, {9, 9}, {1, 9}, {-1, 11}, {11, 11}, {11, -1}, {-1, -1}};
    for (size_t i = 0; i < square_points.size(); ++i) {
        assert(approx(square_points[i], expected_square[i]));
    }
    // Star tips are sharp: Siv3D cuts them flat at half the thickness instead of a long miter.
    for (const double thickness : {2.0, 4.0}) {
        const Vector_path frame = board_image_polygon_frame(board_image_star_points({100, 100}, 15), thickness);
        double top = 1e9;
        for (const auto& point : path_points(frame)) {
            top = std::min(top, point.y);
        }
        assert(approx(top, 85 - thickness / 2, 0.02));
        assert(path_points(frame).size() == 2 * (10 + 5));
    }
    // Arrow: the tip keeps its miter; the back corners of the head are cut.
    const Vector_path arrow_frame = board_image_polygon_frame(arrow, 2);
    double right = -1e9;
    for (const auto& point : path_points(arrow_frame)) {
        right = std::max(right, point.x);
    }
    assert(approx(right, 100 + 1 / std::sin(std::atan2(12.0, 16.0)), 1e-3));
    assert(path_points(arrow_frame).size() == 2 * (7 + 2));
}

void test_text_layout() {
    const Vector_path path = board_image_text_path(box_font("12"), 24, {100, 100});
    const auto points = path_points(path);
    assert(points.size() == 8);
    // Width 2 * 30 * 0.5 and height 60 * 0.5 are centered; glyph offsets are scaled.
    assert(approx(points[0], {86.5, 90}) && approx(points[2], {98.5, 105}));
    assert(approx(points[4], {101.5, 90}) && approx(points[6], {113.5, 105}));
    Board_image_text_outline invalid;
    assert(board_image_text_path(invalid, 24, {0, 0}).empty());
}

Board_image_edit sample_edit() {
    Board_image_edit edit;
    const int marks[] = {
        BOARD_IMAGE_BRECT, BOARD_IMAGE_WRECT, BOARD_IMAGE_ORECT, BOARD_IMAGE_BSTAR, BOARD_IMAGE_WSTAR,
        BOARD_IMAGE_OSTAR, BOARD_IMAGE_BCIRCLE, BOARD_IMAGE_WCIRCLE, BOARD_IMAGE_OCIRCLE, BOARD_IMAGE_FLIP
    };
    // Row 1 and row 8: each mark on an empty cell; d4/e4/d5/e5: marks on discs.
    for (int i = 0; i < 8; ++i) {
        edit.marks[i] = marks[i];
        edit.marks[56 + i] = marks[(i + 2) % 10];
    }
    edit.marks[27] = BOARD_IMAGE_FLIP;   // white disc
    edit.marks[28] = BOARD_IMAGE_OSTAR;  // black disc
    edit.marks[35] = BOARD_IMAGE_WCIRCLE;
    edit.marks[36] = BOARD_IMAGE_BSTAR;
    edit.marks[19] = BOARD_IMAGE_VALUE;  // d3
    edit.values[19] = -6;
    edit.marks[37] = BOARD_IMAGE_VALUE;  // f5, still calculating
    edit.marks[44] = BOARD_IMAGE_VALUE;  // e6
    edit.values[44] = 12;
    edit.arrows.push_back({16, 23}); // a3 -> h3
    edit.arrows.push_back({50, 41}); // c7 -> b6
    edit.arrows.push_back({20, 20}); // zero length, skipped
    return edit;
}

void test_board(const std::string& output_directory) {
    const std::array<int, 64> discs = initial_board();
    const Board_image_edit edit = sample_edit();
    for (const bool include_coordinate : {true, false}) {
        for (const bool monochrome : {false, true}) {
            for (const bool dark_frame : {false, true}) {
                if (monochrome && dark_frame) {
                    continue;
                }
                Board_image_vector_options options;
                options.include_coordinate = include_coordinate;
                options.monochrome = monochrome;
                options.dark_outer_frame = dark_frame;
                std::vector<std::string> texts;
                const auto provider = [&texts](const std::string& text) {
                    texts.push_back(text);
                    return box_font(text);
                };
                Vector_image image = build_board_image_vector(discs, edit, options, provider);
                image.set_title("Egaroucid board image");
                const Board_image_vector_region region = board_image_vector_region({}, include_coordinate);
                assert(approx(image.width(), include_coordinate ? 447 : 420) && approx(image.width(), image.height()));
                assert(approx(image.origin_x(), region.x) && approx(image.origin_y(), region.y));
                assert(approx(region.x, include_coordinate ? 10 : 30) && approx(region.y, include_coordinate ? 30 : 50));

                // Text: coordinates in GUI order, then values (pending values are "...").
                std::vector<std::string> expected;
                if (include_coordinate) {
                    for (int i = 0; i < 8; ++i) {
                        expected.push_back(std::to_string(i + 1));
                        expected.push_back(std::string(1, static_cast<char>('a' + i)));
                    }
                }
                expected.insert(expected.end(), {"-6", "...", "12"});
                assert(texts == expected);

                const std::string svg = vector_image_to_svg(image);
                const std::string groups = include_coordinate ?
                    "background coordinates grid discs frame marks arrows" : "background grid discs frame marks arrows";
                std::string actual_groups;
                for (size_t pos = svg.find("<g id=\""); pos != std::string::npos; pos = svg.find("<g id=\"", pos + 1)) {
                    const size_t start = pos + 7;
                    actual_groups += (actual_groups.empty() ? "" : " ") + svg.substr(start, svg.find('"', start) - start);
                }
                assert(actual_groups == groups);
                const std::string background = monochrome ? "#ffffff" : "#249972";
                if (include_coordinate) {
                    assert(svg.find("<path d=\"M0 0L447 0L447 447L0 447Z\" fill=\"" + background + "\"/>") != std::string::npos);
                } else {
                    // Transparent outside the frame: the background is a round rect, not the whole canvas.
                    assert(svg.find("<path d=\"M33 5L387 5A28 28 0 0 1 415 33") != std::string::npos);
                }
                const std::string line = monochrome ? "#000000" : "#333333";
                assert(svg.find("fill=\"none\" stroke=\"" + line + "\" stroke-width=\"2\" stroke-linecap=\"square\"") != std::string::npos);
                // Four discs, plus their rings for white discs on monochrome boards and the circle marks.
                size_t expected_circles = 4 + (monochrome ? 2 : 0);
                for (int cell = 0; cell < 64; ++cell) {
                    const int mark = edit.marks[cell];
                    if (mark == BOARD_IMAGE_BCIRCLE || mark == BOARD_IMAGE_WCIRCLE || mark == BOARD_IMAGE_OCIRCLE) {
                        const auto style = board_image_mark_style(mark, discs[cell], monochrome);
                        expected_circles += 1 + (style.halo ? 1 : 0) + (style.backing ? 1 : 0);
                    }
                }
                assert(count(svg, "<circle ") == expected_circles);
                const std::string frame = monochrome ? "#000000" : (dark_frame ? "#333333" : "#ffffff");
                const size_t frame_group = svg.find("<g id=\"frame\">");
                assert(svg.find("fill=\"" + frame + "\" fill-rule=\"evenodd\"", frame_group) < svg.find("</g>", frame_group));
                // Two drawable arrows: a black body and a white Siv3D-style frame for each.
                const size_t arrows = svg.find("<g id=\"arrows\">");
                assert(count(svg.substr(arrows), "fill=\"#000000\"") == 2 && count(svg.substr(arrows), "fill=\"#ffffff\"") == 2);

                const std::string eps = vector_image_to_eps(image);
                const std::string size = include_coordinate ? "447" : "420";
                assert(eps.find("%%BoundingBox: 0 0 " + size + " " + size + "\n") != std::string::npos);
                assert(count(eps, "\nn\n") == image.shapes().size());
                for (const char c : eps) {
                    assert(c == '\n' || (0x20 <= static_cast<unsigned char>(c) && static_cast<unsigned char>(c) <= 0x7e));
                }

                if (!output_directory.empty()) {
                    const std::string name = output_directory + "/board_" + (include_coordinate ? "coord" : "nocoord") + "_" +
                        (monochrome ? "mono" : (dark_frame ? "dark" : "white"));
                    write_file(name + ".svg", svg);
                    write_file(name + ".eps", eps);
                }
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    test_numbers();
    test_writers();
    test_mark_style();
    test_geometry();
    test_text_layout();
    test_board(argc > 1 ? argv[1] : "");
    std::cout << "PASS: numbers, SVG/EPS writers, mark colors, Siv3D shapes and frames, text layout, and board export\n";
}
