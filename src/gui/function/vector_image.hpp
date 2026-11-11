/*
    Egaroucid Project

    @file vector_image.hpp
        Small vector image model with SVG, EPS, and PDF writers
        Standard C++ only (no Siv3D or other libraries), so it can be tested alone.
    @date 2026
    @author Takuto Yamana
    @license GPL-3.0-or-later
*/

#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

constexpr double VECTOR_IMAGE_PI = 3.14159265358979323846;

// Coordinates are pixels with y pointing down, as in the GUI.
struct Vector_point {
    double x = 0.0;
    double y = 0.0;
};

inline Vector_point operator+(const Vector_point& a, const Vector_point& b) { return {a.x + b.x, a.y + b.y}; }
inline Vector_point operator-(const Vector_point& a, const Vector_point& b) { return {a.x - b.x, a.y - b.y}; }
inline Vector_point operator-(const Vector_point& a) { return {-a.x, -a.y}; }
inline Vector_point operator*(const Vector_point& a, double scale) { return {a.x * scale, a.y * scale}; }

inline double vector_dot(const Vector_point& a, const Vector_point& b) {
    return a.x * b.x + a.y * b.y;
}

inline double vector_length(const Vector_point& a) {
    return std::sqrt(vector_dot(a, a));
}

inline Vector_point vector_normalized(const Vector_point& a) {
    const double length = vector_length(a);
    return length > 0.0 ? a * (1.0 / length) : a;
}

// Angles are radians from +x towards +y, so a positive sweep is clockwise on screen.
inline Vector_point vector_arc_point(const Vector_point& center, double radius, double angle) {
    return {center.x + radius * std::cos(angle), center.y + radius * std::sin(angle)};
}

struct Vector_color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

inline bool operator==(const Vector_color& a, const Vector_color& b) {
    return a.r == b.r && a.g == b.g && a.b == b.b;
}

enum class Vector_fill_rule { nonzero, evenodd };
enum class Vector_line_cap { butt, round, square };
enum class Vector_line_join { miter, round, bevel };

class Vector_path {
public:
    enum class Command_type { move, line, arc, close };

    struct Command {
        Command_type type = Command_type::move;
        Vector_point point;   // move / line: destination, arc: center
        double radius = 0.0;  // arc only
        double start = 0.0;   // arc only: start angle
        double sweep = 0.0;   // arc only: signed sweep angle
    };

    void move_to(const Vector_point& point) {
        Command command;
        command.type = Command_type::move;
        command.point = point;
        commands_.push_back(command);
        current_ = point;
        subpath_start_ = point;
        has_current_ = true;
    }

    void line_to(const Vector_point& point) {
        if (!has_current_) {
            move_to(point);
            return;
        }
        Command command;
        command.type = Command_type::line;
        command.point = point;
        commands_.push_back(command);
        current_ = point;
    }

    // Like PostScript's arc, a line joins the current point to the start of the arc.
    // Without a current point, the arc starts a new subpath.
    void arc(const Vector_point& center, double radius, double start, double sweep) {
        const Vector_point from = vector_arc_point(center, radius, start);
        if (!has_current_) {
            move_to(from);
        } else if (vector_length(from - current_) > 1e-9) {
            line_to(from);
        }
        Command command;
        command.type = Command_type::arc;
        command.point = center;
        command.radius = radius;
        command.start = start;
        command.sweep = sweep;
        commands_.push_back(command);
        current_ = vector_arc_point(center, radius, start + sweep);
    }

    void close() {
        if (!has_current_) {
            return;
        }
        Command command;
        command.type = Command_type::close;
        command.point = subpath_start_;
        commands_.push_back(command);
        current_ = subpath_start_;
        has_current_ = false;
    }

    void add_line(const Vector_point& from, const Vector_point& to) {
        move_to(from);
        line_to(to);
    }

    void add_polygon(const std::vector<Vector_point>& points) {
        if (points.size() < 2) {
            return;
        }
        move_to(points[0]);
        for (size_t i = 1; i < points.size(); ++i) {
            line_to(points[i]);
        }
        close();
    }

    void add_rect(double x, double y, double width, double height) {
        add_polygon({{x, y}, {x + width, y}, {x + width, y + height}, {x, y + height}});
    }

    void add_round_rect(double x, double y, double width, double height, double radius) {
        radius = std::clamp(radius, 0.0, std::min(width, height) * 0.5);
        if (radius <= 0.0) {
            add_rect(x, y, width, height);
            return;
        }
        constexpr double quarter = VECTOR_IMAGE_PI * 0.5;
        move_to({x + radius, y});
        arc({x + width - radius, y + radius}, radius, -quarter, quarter);
        arc({x + width - radius, y + height - radius}, radius, 0.0, quarter);
        arc({x + radius, y + height - radius}, radius, quarter, quarter);
        arc({x + radius, y + radius}, radius, VECTOR_IMAGE_PI, quarter);
        close();
    }

    void add_circle(const Vector_point& center, double radius) {
        move_to(vector_arc_point(center, radius, 0.0));
        arc(center, radius, 0.0, 2.0 * VECTOR_IMAGE_PI);
        close();
    }

    bool empty() const {
        return commands_.empty();
    }

    const std::vector<Command>& commands() const {
        return commands_;
    }

    // True when the path is exactly one full circle made by add_circle().
    bool is_circle(Vector_point* center, double* radius) const {
        if (commands_.size() != 3 || commands_[0].type != Command_type::move ||
            commands_[1].type != Command_type::arc || commands_[2].type != Command_type::close ||
            std::abs(commands_[1].sweep) < 2.0 * VECTOR_IMAGE_PI - 1e-9) {
            return false;
        }
        *center = commands_[1].point;
        *radius = commands_[1].radius;
        return true;
    }

private:
    std::vector<Command> commands_;
    Vector_point current_;
    Vector_point subpath_start_;
    bool has_current_ = false;
};

struct Vector_fill {
    Vector_color color;
    Vector_fill_rule rule = Vector_fill_rule::nonzero;
};

struct Vector_stroke {
    Vector_color color;
    double width = 1.0;
    Vector_line_cap cap = Vector_line_cap::butt;
    Vector_line_join join = Vector_line_join::miter;
    double miter_limit = 4.0;
};

struct Vector_shape {
    std::string group;
    Vector_path path;
    std::optional<Vector_fill> fill;
    std::optional<Vector_stroke> stroke;
};

// Shapes are painted in insertion order. The origin is the top-left corner of the
// written image, so the content can stay in the coordinates of the screen.
class Vector_image {
public:
    Vector_image(double width, double height, double origin_x = 0.0, double origin_y = 0.0)
        : width_(width), height_(height), origin_x_(origin_x), origin_y_(origin_y) {}

    double width() const { return width_; }
    double height() const { return height_; }
    double origin_x() const { return origin_x_; }
    double origin_y() const { return origin_y_; }

    const std::string& title() const { return title_; }
    void set_title(const std::string& title) { title_ = title; }

    // Following shapes belong to this group (an SVG <g> element) until changed.
    void set_group(const std::string& group) { group_ = group; }

    void fill(const Vector_path& path, const Vector_color& color, Vector_fill_rule rule = Vector_fill_rule::nonzero) {
        Vector_fill fill;
        fill.color = color;
        fill.rule = rule;
        add(path, fill, std::nullopt);
    }

    void stroke(const Vector_path& path, const Vector_color& color, double width,
                Vector_line_cap cap = Vector_line_cap::butt,
                Vector_line_join join = Vector_line_join::miter, double miter_limit = 4.0) {
        Vector_stroke stroke;
        stroke.color = color;
        stroke.width = width;
        stroke.cap = cap;
        stroke.join = join;
        stroke.miter_limit = miter_limit;
        add(path, std::nullopt, stroke);
    }

    void add(const Vector_path& path, const std::optional<Vector_fill>& fill, const std::optional<Vector_stroke>& stroke) {
        if (path.empty() || (!fill && !stroke)) {
            return;
        }
        Vector_shape shape;
        shape.group = group_;
        shape.path = path;
        shape.fill = fill;
        shape.stroke = stroke;
        shapes_.push_back(std::move(shape));
    }

    const std::vector<Vector_shape>& shapes() const { return shapes_; }

private:
    double width_;
    double height_;
    double origin_x_;
    double origin_y_;
    std::string title_;
    std::string group_;
    std::vector<Vector_shape> shapes_;
};

// Locale-independent fixed-point text without trailing zeros, such as "-12.5".
inline std::string vector_image_number(double value, int decimals = 3) {
    if (!std::isfinite(value)) {
        value = 0.0;
    }
    long long scale = 1;
    for (int i = 0; i < decimals; ++i) {
        scale *= 10;
    }
    const long long scaled = std::llround(value * static_cast<double>(scale));
    const unsigned long long magnitude = scaled < 0 ? 0ULL - static_cast<unsigned long long>(scaled) : static_cast<unsigned long long>(scaled);
    const unsigned long long unit = static_cast<unsigned long long>(scale);
    std::string result = scaled < 0 ? "-" : "";
    result += std::to_string(magnitude / unit);
    const unsigned long long fraction = magnitude % unit;
    if (fraction != 0) {
        std::string digits = std::to_string(fraction);
        digits.insert(0, static_cast<size_t>(decimals) - digits.size(), '0');
        while (digits.back() == '0') {
            digits.pop_back();
        }
        result += "." + digits;
    }
    return result;
}

inline std::string vector_image_xml_escape(const std::string& text) {
    std::string result;
    for (const char c : text) {
        switch (c) {
        case '&': result += "&amp;"; break;
        case '<': result += "&lt;"; break;
        case '>': result += "&gt;"; break;
        case '"': result += "&quot;"; break;
        case '\'': result += "&apos;"; break;
        default: result += c; break;
        }
    }
    return result;
}

inline std::string vector_image_svg_color(const Vector_color& color) {
    constexpr char hex[] = "0123456789abcdef";
    std::string result = "#";
    for (const uint8_t value : {color.r, color.g, color.b}) {
        result += hex[value >> 4];
        result += hex[value & 15];
    }
    return result;
}

// Arcs are split into pieces of at most 90 degrees.
inline int vector_image_arc_pieces(double sweep) {
    return std::max(1, static_cast<int>(std::ceil(std::abs(sweep) / (VECTOR_IMAGE_PI * 0.5) - 1e-9)));
}

inline std::string vector_image_svg_path_data(const Vector_path& path, double origin_x, double origin_y) {
    const auto point = [origin_x, origin_y](const Vector_point& p) {
        return vector_image_number(p.x - origin_x) + " " + vector_image_number(p.y - origin_y);
    };
    std::string data;
    for (const auto& command : path.commands()) {
        switch (command.type) {
        case Vector_path::Command_type::move:
            data += "M" + point(command.point);
            break;
        case Vector_path::Command_type::line:
            data += "L" + point(command.point);
            break;
        case Vector_path::Command_type::arc: {
            const int pieces = vector_image_arc_pieces(command.sweep);
            const std::string radius = vector_image_number(command.radius);
            const char* sweep_flag = command.sweep > 0.0 ? " 0 0 1 " : " 0 0 0 ";
            for (int i = 1; i <= pieces; ++i) {
                const double angle = command.start + command.sweep * i / pieces;
                data += "A" + radius + " " + radius + sweep_flag +
                    point(vector_arc_point(command.point, command.radius, angle));
            }
            break;
        }
        case Vector_path::Command_type::close:
            data += "Z";
            break;
        }
    }
    return data;
}

inline std::string vector_image_svg_paint(const Vector_shape& shape) {
    std::string attributes;
    if (shape.fill) {
        attributes += " fill=\"" + vector_image_svg_color(shape.fill->color) + "\"";
        if (shape.fill->rule == Vector_fill_rule::evenodd) {
            attributes += " fill-rule=\"evenodd\"";
        }
    } else {
        attributes += " fill=\"none\"";
    }
    if (shape.stroke) {
        const Vector_stroke& stroke = *shape.stroke;
        attributes += " stroke=\"" + vector_image_svg_color(stroke.color) + "\" stroke-width=\"" + vector_image_number(stroke.width) + "\"";
        if (stroke.cap == Vector_line_cap::round) {
            attributes += " stroke-linecap=\"round\"";
        } else if (stroke.cap == Vector_line_cap::square) {
            attributes += " stroke-linecap=\"square\"";
        }
        if (stroke.join == Vector_line_join::round) {
            attributes += " stroke-linejoin=\"round\"";
        } else if (stroke.join == Vector_line_join::bevel) {
            attributes += " stroke-linejoin=\"bevel\"";
        } else if (stroke.miter_limit != 4.0) {
            attributes += " stroke-miterlimit=\"" + vector_image_number(stroke.miter_limit) + "\"";
        }
    }
    return attributes;
}

inline std::string vector_image_to_svg(const Vector_image& image) {
    const std::string width = vector_image_number(image.width());
    const std::string height = vector_image_number(image.height());
    std::string svg;
    svg += "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n";
    svg += "<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\" width=\"" + width + "\" height=\"" + height +
        "\" viewBox=\"0 0 " + width + " " + height + "\">\n";
    if (!image.title().empty()) {
        svg += "<title>" + vector_image_xml_escape(image.title()) + "</title>\n";
    }
    std::set<std::string> used_ids;
    std::string open_group;
    bool in_group = false;
    for (const auto& shape : image.shapes()) {
        if (!in_group || shape.group != open_group) {
            if (in_group && !open_group.empty()) {
                svg += "</g>\n";
            }
            if (!shape.group.empty()) {
                std::string id = shape.group;
                for (int i = 2; used_ids.count(id); ++i) {
                    id = shape.group + "-" + std::to_string(i);
                }
                used_ids.insert(id);
                svg += "<g id=\"" + vector_image_xml_escape(id) + "\">\n";
            }
            open_group = shape.group;
            in_group = true;
        }
        const std::string indent = shape.group.empty() ? "" : "  ";
        Vector_point center;
        double radius = 0.0;
        if (shape.path.is_circle(&center, &radius)) {
            svg += indent + "<circle cx=\"" + vector_image_number(center.x - image.origin_x()) + "\" cy=\"" +
                vector_image_number(center.y - image.origin_y()) + "\" r=\"" + vector_image_number(radius) + "\"" +
                vector_image_svg_paint(shape) + "/>\n";
        } else {
            svg += indent + "<path d=\"" + vector_image_svg_path_data(shape.path, image.origin_x(), image.origin_y()) + "\"" +
                vector_image_svg_paint(shape) + "/>\n";
        }
    }
    if (in_group && !open_group.empty()) {
        svg += "</g>\n";
    }
    svg += "</svg>\n";
    return svg;
}

// DSC comments must stay printable ASCII on one line.
inline std::string vector_image_dsc_text(const std::string& text) {
    std::string result;
    for (const char c : text) {
        const unsigned char u = static_cast<unsigned char>(c);
        result += (0x20 <= u && u <= 0x7e) ? c : '?';
    }
    return result;
}

// EPS and PDF share these path operators. Their user space is flipped to y-down, so arcs use
// Bezier curves in the SVG coordinates. Each piece covers at most 90 degrees (radius error < 0.03%).
inline std::string vector_image_eps_path(const Vector_path& path, double origin_x, double origin_y) {
    const auto point = [origin_x, origin_y](const Vector_point& p) {
        return vector_image_number(p.x - origin_x) + " " + vector_image_number(p.y - origin_y);
    };
    std::string data;
    for (const auto& command : path.commands()) {
        switch (command.type) {
        case Vector_path::Command_type::move:
            data += point(command.point) + " m\n";
            break;
        case Vector_path::Command_type::line:
            data += point(command.point) + " l\n";
            break;
        case Vector_path::Command_type::arc: {
            const int pieces = vector_image_arc_pieces(command.sweep);
            const double piece = command.sweep / pieces;
            const double handle = 4.0 / 3.0 * std::tan(piece / 4.0) * command.radius;
            for (int i = 0; i < pieces; ++i) {
                const double a0 = command.start + piece * i;
                const double a1 = a0 + piece;
                const Vector_point p0 = vector_arc_point(command.point, command.radius, a0);
                const Vector_point p3 = vector_arc_point(command.point, command.radius, a1);
                const Vector_point p1 = p0 + Vector_point{-std::sin(a0), std::cos(a0)} * handle;
                const Vector_point p2 = p3 - Vector_point{-std::sin(a1), std::cos(a1)} * handle;
                data += point(p1) + " " + point(p2) + " " + point(p3) + " c\n";
            }
            break;
        }
        case Vector_path::Command_type::close:
            data += "h\n";
            break;
        }
    }
    return data;
}

inline std::string vector_image_eps_color(const Vector_color& color) {
    return vector_image_number(color.r / 255.0, 4) + " " + vector_image_number(color.g / 255.0, 4) + " " +
        vector_image_number(color.b / 255.0, 4);
}

inline std::string vector_image_to_eps(const Vector_image& image) {
    const long long bbox_width = static_cast<long long>(std::ceil(image.width() - 1e-9));
    const long long bbox_height = static_cast<long long>(std::ceil(image.height() - 1e-9));
    std::string eps;
    eps += "%!PS-Adobe-3.0 EPSF-3.0\n";
    eps += "%%BoundingBox: 0 0 " + std::to_string(bbox_width) + " " + std::to_string(bbox_height) + "\n";
    eps += "%%HiResBoundingBox: 0 0 " + vector_image_number(image.width()) + " " + vector_image_number(image.height()) + "\n";
    if (!image.title().empty()) {
        eps += "%%Title: " + vector_image_dsc_text(image.title()) + "\n";
    }
    eps += "%%Creator: Egaroucid\n";
    eps += "%%LanguageLevel: 2\n";
    eps += "%%DocumentData: Clean7Bit\n";
    eps += "%%Pages: 1\n";
    eps += "%%EndComments\n";
    eps += "%%BeginProlog\n";
    // A private dictionary keeps the short names out of the importing document.
    eps += "/EgaroucidVectorDict 16 dict def\n";
    eps += "EgaroucidVectorDict begin\n";
    eps += "/m {moveto} bind def\n";
    eps += "/l {lineto} bind def\n";
    eps += "/c {curveto} bind def\n";
    eps += "/h {closepath} bind def\n";
    eps += "/n {newpath} bind def\n";
    eps += "/rg {setrgbcolor} bind def\n";
    eps += "/f {fill} bind def\n";
    eps += "/ef {eofill} bind def\n";
    eps += "% r g b width cap join miterlimit st\n";
    eps += "/st {setmiterlimit setlinejoin setlinecap setlinewidth setrgbcolor stroke} bind def\n";
    eps += "end\n";
    eps += "%%EndProlog\n";
    eps += "%%Page: 1 1\n";
    eps += "EgaroucidVectorDict begin\n";
    eps += "gsave\n";
    eps += "0 " + vector_image_number(image.height()) + " translate 1 -1 scale\n";
    std::string current_group;
    bool first = true;
    for (const auto& shape : image.shapes()) {
        if (first || shape.group != current_group) {
            if (!shape.group.empty()) {
                eps += "% " + vector_image_dsc_text(shape.group) + "\n";
            }
            current_group = shape.group;
            first = false;
        }
        eps += "n\n" + vector_image_eps_path(shape.path, image.origin_x(), image.origin_y());
        std::string fill;
        if (shape.fill) {
            fill = vector_image_eps_color(shape.fill->color) + " rg " +
                (shape.fill->rule == Vector_fill_rule::evenodd ? "ef" : "f");
        }
        std::string stroke;
        if (shape.stroke) {
            const Vector_stroke& s = *shape.stroke;
            const int cap = s.cap == Vector_line_cap::butt ? 0 : s.cap == Vector_line_cap::round ? 1 : 2;
            const int join = s.join == Vector_line_join::miter ? 0 : s.join == Vector_line_join::round ? 1 : 2;
            stroke = vector_image_eps_color(s.color) + " " + vector_image_number(s.width) + " " + std::to_string(cap) + " " +
                std::to_string(join) + " " + vector_image_number(s.miter_limit) + " st";
        }
        if (shape.fill && shape.stroke) {
            eps += "gsave " + fill + " grestore\n" + stroke + "\n";
        } else {
            eps += (shape.fill ? fill : stroke) + "\n";
        }
    }
    eps += "grestore\n";
    eps += "end\n";
    eps += "showpage\n";
    eps += "%%Trailer\n";
    eps += "%%EOF\n";
    return eps;
}

// One page, sized to the board, with paths only. Text has already been converted to outlines.
inline std::string vector_image_to_pdf(const Vector_image& image) {
    const std::string width = vector_image_number(image.width());
    const std::string height = vector_image_number(image.height());
    std::string content = "q\n1 0 0 -1 0 " + height + " cm\n";
    for (const auto& shape : image.shapes()) {
        if (shape.fill) {
            content += vector_image_eps_color(shape.fill->color) + " rg\n";
        }
        if (shape.stroke) {
            const Vector_stroke& s = *shape.stroke;
            const int cap = s.cap == Vector_line_cap::butt ? 0 : s.cap == Vector_line_cap::round ? 1 : 2;
            const int join = s.join == Vector_line_join::miter ? 0 : s.join == Vector_line_join::round ? 1 : 2;
            content += vector_image_eps_color(s.color) + " RG\n" + vector_image_number(s.width) + " w\n" +
                std::to_string(cap) + " J\n" + std::to_string(join) + " j\n" + vector_image_number(s.miter_limit) + " M\n";
        }
        content += vector_image_eps_path(shape.path, image.origin_x(), image.origin_y());
        if (shape.fill) {
            content += shape.stroke ? "B" : "f";
            content += shape.fill->rule == Vector_fill_rule::evenodd ? "*\n" : "\n";
        } else {
            content += "S\n";
        }
    }
    content += "Q\n";

    const std::vector<std::string> objects = {
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + width + " " + height +
            "] /Resources << >> /Contents 4 0 R >>",
        "<< /Length " + std::to_string(content.size()) + " >>\nstream\n" + content + "endstream"
    };
    std::string pdf = "%PDF-1.4\n%\xe2\xe3\xcf\xd3\n";
    std::vector<size_t> offsets;
    for (size_t i = 0; i < objects.size(); ++i) {
        offsets.push_back(pdf.size());
        pdf += std::to_string(i + 1) + " 0 obj\n" + objects[i] + "\nendobj\n";
    }
    const size_t xref = pdf.size();
    pdf += "xref\n0 " + std::to_string(objects.size() + 1) + "\n0000000000 65535 f \n";
    for (const size_t offset : offsets) {
        const std::string number = std::to_string(offset);
        pdf += std::string(10 - number.size(), '0') + number + " 00000 n \n";
    }
    pdf += "trailer\n<< /Size " + std::to_string(objects.size() + 1) + " /Root 1 0 R >>\nstartxref\n" +
        std::to_string(xref) + "\n%%EOF\n";
    return pdf;
}
