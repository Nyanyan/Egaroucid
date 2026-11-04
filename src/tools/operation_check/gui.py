"""Run isolated native GUI checks. Requires a GUI Portable build with #652 support."""

import json
import os
from pathlib import Path
import re
import shutil
import uuid
import xml.etree.ElementTree as ET

from common import Case, parser_for, require, run_suite
from reference_board import Board


LANGUAGES = ("english", "japanese", "simplified_chinese", "traditional_chinese_taiwan")


def require_probe_support(executable):
    # Old installed binaries ignore the flag and would load personal settings.
    # Refuse them before starting a process. Siv3D stores U"" literals as UTF-32.
    marker = "egaroucid-operation-check-v1"
    binary = executable.read_bytes()
    require(any(marker.encode(encoding) in binary for encoding in ("ascii", "utf-16-le", "utf-32-le")),
            "GUI binary lacks isolated operation-check support; rebuild a GUI Portable configuration")


def fixture(ctx, *, language="japanese", settings=None):
    application = ctx.application().parent
    # Windows GUI resources are relative to cwd. Link the disposable snapshot
    # (never the user's files), falling back to copies across filesystems.
    def link_or_copy(source, target):
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    shutil.copytree(application / "resources", ctx.directory / "resources", copy_function=link_or_copy)
    if (application / "LICENSE").is_file():
        shutil.copy2(application / "LICENSE", ctx.directory / "LICENSE")
    for folder in ("appdata", "document/games", "document/screenshots", "document/forced_openings"):
        (ctx.directory / folder).mkdir(parents=True, exist_ok=True)
    shutil.copy2(ctx.args.resources / "empty_book.egbk3", ctx.directory / "document/book.egbk3")
    values = {
        "lang_name": language, "book_file": "./document/book.egbk3", "n_threads": 1,
        "hash_level": 18, "level": 1, "use_book": False,
        "ai_put_black": False, "ai_put_white": False, "use_disc_hint": False,
        "use_umigame_value": False, "show_principal_variation": False,
        "show_book_accuracy": False, "show_ai_focus": False,
        "window_scale": 1.0, "window_pos_x": 100, "window_pos_y": 100,
        "screenshot_saving_dir": "./document/screenshots/",
    }
    values.update(settings or {})
    (ctx.directory / "appdata/setting.json").write_text(json.dumps(values), encoding="utf-8")


def native(ctx, name):
    executable = ctx.application()
    require_probe_support(executable)
    token = str(uuid.uuid4())
    request = {"marker": "egaroucid-operation-check-v1", "case": name, "token": token}
    (ctx.directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
    result = ctx.run([executable, "--operation-check", ctx.directory], cwd=ctx.directory)
    require(result.returncode == 0, f"GUI exited with code {result.returncode}")
    path = ctx.directory / "native-result.json"
    require(path.is_file(), "GUI did not write a result; use a Portable build with #652 support (see GUI log)")
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    require(report.get("protocol") == 1 and report.get("token") == token and report.get("case") == name,
            "Stale or incompatible native result")
    require(report.get("passed") is True, report.get("error", "Native check failed"))
    require(report.get("portable") is True, "GUI did not run in portable mode")
    require((ctx.directory / "appdata/shortcut_key.json").is_file(), "Normal shutdown did not save shortcuts")
    # Keep each invocation's report as well as the latest one for restart checks.
    shutil.copy2(path, ctx.directory / f"native-{ctx.process_index:02d}.json")
    return report


def simple(ctx, name):
    fixture(ctx)
    return native(ctx, name)


def settings(ctx):
    fixture(ctx)
    native(ctx, "settings")
    path = ctx.directory / "appdata/setting.json"
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    keys = ("board_image_mark", "board_image_coordinate", "board_image_color", "board_image_outer_frame_color",
            "board_image_format")
    require([saved.get(key) for key in keys] == [11, 1, 1, 1, 1], "Shutdown did not save editor settings")
    require(saved.get("timer_time_limit_min") == 7 and saved.get("show_legal") is False, "Shutdown display settings differ")
    # Restart from the actual saved files, without rebuilding the fixture.
    native(ctx, "settings-restored")
    require(json.loads(path.read_text(encoding="utf-8-sig"))["board_image_mark"] == 11, "Restart changed selected mark")


def invalid_settings(ctx):
    fixture(ctx, settings={"board_image_mark": 999, "board_image_coordinate": -9,
                           "board_image_color": "invalid", "board_image_outer_frame_color": 7,
                           "board_image_format": 9})
    native(ctx, "settings-invalid")


def transcript(ctx, name):
    report = simple(ctx, name)
    expected = Board().replay("f5d6c3d3")
    actual = report.get("board", "").replace("X", "B").replace("O", "W")
    require(actual == "".join(expected.cells) + " " + expected.turn, "Imported GUI board differs from independent rules")


def scenes(ctx, language):
    fixture(ctx, language=language)
    report = native(ctx, "scenes")
    require(report.get("language") == language, "GUI fell back to a different language")
    frames = list(ctx.directory.glob("frame-*.png"))
    require(len(frames) == 9, "Not all nine scenes produced screenshots")
    for frame in frames:
        require(frame.stat().st_size > 1000 and frame.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n",
                "Missing or invalid scene PNG: " + frame.name)


def board_image_vector(ctx):
    fixture(ctx)
    report = native(ctx, "board-image-vector")
    require(report.get("vector_files") == 8, "GUI did not report eight saved vector images")
    folder = ctx.directory / "document/screenshots"
    svgs = sorted(folder.glob("*_vector_*.svg"))
    epss = sorted(folder.glob("*_vector_*.eps"))
    require(len(svgs) == 4 and len(epss) == 4, "Missing SVG/EPS files")
    ns = "{http://www.w3.org/2000/svg}"
    for path in svgs:
        root = ET.parse(path).getroot()
        size = "447" if "_coord_" in path.name else "420"
        require(root.tag == ns + "svg" and root.get("width") == size and root.get("height") == size,
                "SVG size differs: " + path.name)
        groups = [group.get("id") for group in root.findall(ns + "g")]
        expected = ["background", "coordinates", "grid", "discs", "frame", "marks", "arrows"]
        if size == "420":
            expected.remove("coordinates")
        require(groups == expected, "SVG layers differ: " + path.name)
        # Vector shapes only: an embedded bitmap would defeat the purpose of SVG.
        require(all(element.tag in (ns + "svg", ns + "title", ns + "g", ns + "path", ns + "circle")
                    for element in root.iter()), "Unexpected SVG element: " + path.name)
        if size == "447":
            labels = root.find(ns + "g[@id='coordinates']")
            # Each of 1-8 and a-h is a filled glyph outline; together they have hundreds of vertices.
            vertices = [label.get("d", "").count("L") for label in labels]
            require(len(labels) == 16 and min(vertices) >= 3 and sum(vertices) >= 300,
                    "Coordinate glyph outlines are missing: " + path.name)
    for path in epss:
        text = path.read_bytes().decode("ascii")
        size = "447" if "_coord_" in path.name else "420"
        require(text.startswith("%!PS-Adobe-3.0 EPSF-3.0\n") and f"%%BoundingBox: 0 0 {size} {size}\n" in text,
                "EPS header differs: " + path.name)
        require(text.endswith("showpage\n%%Trailer\n%%EOF\n"), "EPS trailer differs: " + path.name)
        body = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
        require(re.search(r"\b(?:color)?image(?:mask)?\b", body) is None, "EPS must not contain a bitmap: " + path.name)
        require(max(len(line) for line in text.splitlines()) <= 255, "EPS line too long: " + path.name)


def resources(ctx):
    for name in ("eval.egev2", "eval_move_ordering_end.egev", "empty_book.egbk3",
                 "img/icon.png", "img/logo.png", "languages/languages.json",
                 "openings/japanese/openings.txt", "openings/english/openings.txt", "xot/openingslarge.txt"):
        path = ctx.args.resources / name
        require(path.is_file() and path.stat().st_size > 0, "Missing/empty resource: " + name)
    for language in LANGUAGES:
        path = ctx.args.resources / "languages" / (language + ".json")
        content = json.loads(path.read_text(encoding="utf-8-sig"))
        for key in ("mark", "value", "save_image", "clear", "arrow", "flip", "coordinate",
                    "file_format", "png", "svg", "eps"):
            require(isinstance(content.get("board_image", {}).get(key), str) and content["board_image"][key],
                    f"Missing {language} board_image.{key}")
        if ctx.args.templates:
            template = json.loads((ctx.args.templates / (language + ".json")).read_text(encoding="utf-8-sig"))
            require(content == template, f"Release template differs: {language}")


CASES = [
    Case("resources", "Required resources and four language packs", resources),
    Case("startup", "Normal loading, initial board, rendering, shutdown", lambda ctx: simple(ctx, "startup")),
    Case("settings-defaults", "Old settings missing editor keys retain defaults", lambda ctx: simple(ctx, "settings-defaults")),
    Case("settings-invalid", "Invalid editor settings fall back/clamp", invalid_settings),
    Case("settings", "Save/reload settings and restart the GUI", settings),
    Case("transcript", "Transcript detection and board text roundtrip", lambda ctx: transcript(ctx, "transcript")),
    Case("profiles", "AI/display profiles and multilingual names", lambda ctx: simple(ctx, "profiles")),
    Case("game-files", "Save game JSON/CSV and reload history/metadata", lambda ctx: transcript(ctx, "game-files")),
    Case("annotations", "Annotation lifetime, arrows, clearing, grid snapping", lambda ctx: simple(ctx, "annotations")),
    Case("board-image-vector", "Save edited board images as SVG/EPS with the GUI fonts", board_image_vector),
    Case("forced-openings", "Forced opening returns legal continuations", lambda ctx: simple(ctx, "forced-openings")),
    Case("search", "AI search through GUI-initialized engine", lambda ctx: simple(ctx, "search")),
] + [Case("scenes-" + language.replace("_", "-"), "Render nine scenes in " + language,
          lambda ctx, language=language: scenes(ctx, language)) for language in LANGUAGES]


if __name__ == "__main__":
    parser = parser_for("gui", "Egaroucid.exe")
    parser.add_argument("--templates", type=Path, help="Also compare language JSONs against this release-template directory")
    raise SystemExit(run_suite("gui", parser, CASES))
