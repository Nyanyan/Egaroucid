"""Draggable point labels with positions stored per figure and dataset/model ID."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from matplotlib.text import Annotation, Text
from matplotlib.ticker import Formatter
from matplotlib.transforms import Bbox


LABEL_PREFIX = "model-label:"
LEADER_LINE_THRESHOLD_POINTS = 18.0


@dataclass(frozen=True)
class LabelPosition:
    dx: float
    dy: float
    horizontal_alignment: str = "left"
    vertical_alignment: str = "baseline"
    text: str | None = None


def annotation_position(annotation):
    dx, dy = annotation.xyann
    return LabelPosition(float(dx), float(dy), annotation.get_ha(), annotation.get_va(), annotation.get_text())


def update_leader_line(annotation):
    if annotation.arrow_patch is not None:
        annotation.arrow_patch.set_visible(math.hypot(*annotation.xyann) > LEADER_LINE_THRESHOLD_POINTS)


def apply_position(annotation, position):
    annotation.xyann = (position.dx, position.dy)
    annotation.set_ha(position.horizontal_alignment)
    annotation.set_va(position.vertical_alignment)
    if position.text is not None:
        annotation.set_text(position.text)
    update_leader_line(annotation)


def figure_annotations(figure):
    return {label.get_gid()[len(LABEL_PREFIX):]: label
            for axis in figure.axes for label in axis.texts
            if isinstance(label, Annotation) and (label.get_gid() or "").startswith(LABEL_PREFIX)}


def load_label_positions(path):
    path = Path(path)
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("version") != 2 or not isinstance(document.get("figures"), dict):
        raise ValueError(f"Expected version 2 label positions for the new figures: {path}")
    figures = {}
    for name, positions in document["figures"].items():
        figures[name] = {}
        for key, raw in positions.items():
            position = LabelPosition(**raw)
            if not math.isfinite(position.dx) or not math.isfinite(position.dy):
                raise ValueError(f"Non-finite label position: {name}/{key}")
            if position.horizontal_alignment not in {"left", "center", "right"}:
                raise ValueError(f"Invalid label alignment: {name}/{key}")
            if position.vertical_alignment not in {"top", "center", "bottom", "baseline", "center_baseline"}:
                raise ValueError(f"Invalid label alignment: {name}/{key}")
            if position.text is not None and not isinstance(position.text, str):
                raise ValueError(f"Invalid label text: {name}/{key}")
            figures[name][key] = position
    return figures


def load_text_overrides(path):
    path = Path(path)
    if not path.exists():
        return {}
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("version") != 2:
        raise ValueError(f"Expected version 2 label settings: {path}")
    figures = document.get("texts", {})
    if not isinstance(figures, dict) or any(
        not isinstance(labels, dict) or any(not isinstance(text, str) for text in labels.values())
        for labels in figures.values()
    ):
        raise ValueError(f"Invalid text settings: {path}")
    return figures


def save_label_positions(path, figure_name, positions, texts=None):
    path = Path(path)
    # Re-read on each save to retain edits to other figures/windows.
    figures = load_label_positions(path)
    figures.setdefault(figure_name, {}).update(positions)
    all_texts = load_text_overrides(path)
    if texts is not None:
        all_texts.setdefault(figure_name, {}).update(texts)
    document = {"version": 2, "figures": {
        name: {key: asdict(position) for key, position in labels.items()}
        for name, labels in figures.items()
    }, "texts": all_texts}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def apply_label_positions(figure, positions):
    annotations = figure_annotations(figure)
    for key, annotation in annotations.items():
        if key in positions:
            apply_position(annotation, positions[key])
        else:
            update_leader_line(annotation)
    arrange_unplaced_labels(figure, annotations, set(positions))
    return annotations


@dataclass
class TextTarget:
    artist: Text
    setter: object = None

    def set_text(self, text):
        if self.setter is not None:
            self.setter(text)
        self.artist.set_text(text)


class EditedTickFormatter(Formatter):
    """Retain tick names across redraws without changing the axis scale."""
    def __init__(self, original):
        self.original = original
        self.names = {}
        self.offset_text = None

    def __call__(self, value, position=None):
        key = format(float(value), ".12g")
        return self.names[key] if key in self.names else self.original(value, position)

    def set_locs(self, locations):
        self.locs = locations
        self.original.set_locs(locations)

    def get_offset(self):
        return self.original.get_offset() if self.offset_text is None else self.offset_text


def figure_text_targets(figure):
    """Register all visible text, with stable IDs independent of edited names."""
    figure.canvas.draw()
    targets = {}

    def add(key, artist, setter=None):
        if artist is not None and artist.get_visible() and artist.get_text():
            targets[key] = TextTarget(artist, setter)

    for i, artist in enumerate(figure.texts):
        key = "figure/title" if artist is getattr(figure, "_suptitle", None) else f"figure/text/{artist.get_gid() or i}"
        add(key, artist)
    for i, legend in enumerate(figure.legends):
        add(f"figure/legend/{i}/title", legend.get_title())
        for j, artist in enumerate(legend.get_texts()):
            add(f"figure/legend/{i}/label/{j}", artist)
    for i, axis in enumerate(figure.axes):
        if not axis.get_visible():
            continue
        prefix = f"axes/{axis.get_gid() or i}"
        add(f"{prefix}/title", axis.title)
        add(f"{prefix}/title-left", axis._left_title)
        add(f"{prefix}/title-right", axis._right_title)
        for direction, coordinate in (("x", axis.xaxis), ("y", axis.yaxis)):
            add(f"{prefix}/{direction}/label", coordinate.label)
            for kind in ("major", "minor"):
                original = getattr(coordinate, f"get_{kind}_formatter")()
                formatter = original if isinstance(original, EditedTickFormatter) else EditedTickFormatter(original)
                install = getattr(coordinate, f"set_{kind}_formatter")
                for tick in getattr(coordinate, f"get_{kind}_ticks")():
                    lower, upper = sorted(coordinate.get_view_interval())
                    if not lower <= tick.get_loc() <= upper:
                        continue
                    value_key = format(float(tick.get_loc()), ".12g")
                    def rename_tick(text, key=value_key, wrapper=formatter, set_formatter=install):
                        wrapper.names[key] = text
                        set_formatter(wrapper)
                    for side, artist in (("first", tick.label1), ("second", tick.label2)):
                        add(f"{prefix}/{direction}/{kind}/{value_key}/{side}", artist, rename_tick)
                if kind == "major":
                    def rename_offset(text, wrapper=formatter, set_formatter=install):
                        wrapper.offset_text = text
                        set_formatter(wrapper)
                    add(f"{prefix}/{direction}/offset", coordinate.get_offset_text(), rename_offset)
        for j, artist in enumerate(axis.texts):
            if not (artist.get_gid() or "").startswith(LABEL_PREFIX):
                add(f"{prefix}/text/{j}", artist)
        legend = axis.get_legend()
        if legend is not None:
            add(f"{prefix}/legend/title", legend.get_title())
            for j, artist in enumerate(legend.get_texts()):
                add(f"{prefix}/legend/label/{j}", artist)
    return targets


def apply_text_overrides(figure, texts):
    targets = figure_text_targets(figure)
    for key, target in targets.items():
        if key in texts:
            target.set_text(texts[key])
    return targets


def arrange_unplaced_labels(figure, annotations, fixed_keys):
    """Place every unsaved label without hiding points or saved label positions."""
    if not annotations:
        return
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    pixels_per_point = figure.dpi / 72.0
    for axis in figure.axes:
        labels = {key: label for key, label in annotations.items() if label.axes is axis}
        if not labels:
            continue
        bounds = axis.get_window_extent(renderer).padded(-3 * pixels_per_point)
        points = {key: axis.transData.transform(label.xy) for key, label in labels.items()}
        radius = 5 * pixels_per_point
        blocked = [Bbox.from_bounds(x - radius, y - radius, 2 * radius, 2 * radius)
                   for x, y in points.values()]
        blocked.extend(Text.get_window_extent(label, renderer).padded(2 * pixels_per_point)
                       for key, label in labels.items() if key in fixed_keys)

        def crowding(key):
            x, y = points[key]
            return sum(math.hypot(x - px, y - py) < 70 * pixels_per_point
                       for px, py in points.values())

        for key in sorted((key for key in labels if key not in fixed_keys), key=crowding, reverse=True):
            label = labels[key]
            candidates = [annotation_position(label)]
            for distance in (12, 18, 26, 36, 48, 64, 84, 108, 140):
                for dx, dy in ((distance, 0), (-distance, 0), (0, distance), (0, -distance),
                               (distance, distance), (-distance, distance),
                               (distance, -distance), (-distance, -distance)):
                    candidates.append(LabelPosition(dx, dy,
                        "left" if dx > 0 else "right" if dx < 0 else "center",
                        "bottom" if dy > 0 else "top" if dy < 0 else "center"))
            best = candidates[0]
            best_penalty = math.inf
            for candidate in candidates:
                apply_position(label, candidate)
                label.update_positions(renderer)
                box = Text.get_window_extent(label, renderer).padded(2 * pixels_per_point)
                outside = not (bounds.x0 <= box.x0 and box.x1 <= bounds.x1
                               and bounds.y0 <= box.y0 and box.y1 <= bounds.y1)
                penalty = 1000000 if outside else 0
                for obstacle in blocked:
                    overlap = Bbox.intersection(box, obstacle)
                    if overlap is not None:
                        penalty += overlap.width * overlap.height
                if penalty < best_penalty:
                    best, best_penalty = candidate, penalty
                if penalty == 0:
                    break
            # Even in an exceptionally crowded figure, retain every label.
            apply_position(label, best)
            label.update_positions(renderer)
            blocked.append(Text.get_window_extent(label, renderer).padded(2 * pixels_per_point))


def request_label_text(figure, current_text):
    window = figure.canvas.manager.window
    if hasattr(window, "tk"):
        from tkinter import simpledialog
        return simpledialog.askstring("ラベル名の変更", "表示名", initialvalue=current_text, parent=window)
    from matplotlib.backends.qt_compat import QtWidgets
    text, accepted = QtWidgets.QInputDialog.getMultiLineText(window, "文字の変更", "表示する文字", text=current_text)
    return text if accepted else None


class LabelEditor:
    def __init__(self, figure, annotations, figure_name, positions_path, on_save, text_targets=None):
        self.figure = figure
        self.annotations = annotations
        self.figure_name = figure_name
        self.positions_path = Path(positions_path)
        self.on_save = on_save
        self.initial_positions = self.current_positions()
        self.text_targets = figure_text_targets(figure) if text_targets is None else text_targets
        self.initial_texts = self.current_texts()
        self.drag_state = None
        self.selected_annotation = None
        self.selected_text_target = None
        self.connections = [
            figure.canvas.mpl_connect("button_press_event", self.on_press),
            figure.canvas.mpl_connect("motion_notify_event", self.on_motion),
            figure.canvas.mpl_connect("button_release_event", self.on_release),
            figure.canvas.mpl_connect("key_press_event", self.on_key_press),
        ]

    def on_press(self, event):
        if getattr(event, "button", None) != 1 or event.x is None or event.y is None:
            return
        toolbar = getattr(self.figure.canvas.manager, "toolbar", None)
        if toolbar is not None and toolbar.mode:
            return  # Leave pan/zoom gestures to the toolbar.
        renderer = self.figure.canvas.get_renderer()
        for annotation in reversed(list(self.annotations.values())):
            # Hit-test the text, not the potentially long leader line.
            if annotation.get_visible() and Text.get_window_extent(annotation, renderer).contains(event.x, event.y):
                self.selected_annotation = annotation
                self.selected_text_target = None
                if getattr(event, "dblclick", False):
                    self.drag_state = None
                    self.rename_label(annotation)
                    return
                dx, dy = annotation.xyann
                self.drag_state = annotation, event.x, event.y, dx, dy
                return
        for target in reversed(list(self.text_targets.values())):
            artist = target.artist
            if artist.get_visible() and Text.get_window_extent(artist, renderer).contains(event.x, event.y):
                self.selected_annotation = None
                self.selected_text_target = target
                if getattr(event, "dblclick", False):
                    self.rename_text(target)
                return

    def on_motion(self, event):
        if self.drag_state is None or event.x is None or event.y is None:
            return
        annotation, x, y, dx, dy = self.drag_state
        scale = 72.0 / self.figure.dpi
        annotation.xyann = (dx + (event.x - x) * scale, dy + (event.y - y) * scale)
        update_leader_line(annotation)
        self.figure.canvas.draw_idle()

    def on_release(self, _event):
        self.drag_state = None

    def current_positions(self):
        return {key: annotation_position(annotation) for key, annotation in self.annotations.items()}

    def current_texts(self):
        return {key: target.artist.get_text() for key, target in self.text_targets.items()}

    def rename_text(self, target):
        text = request_label_text(self.figure, target.artist.get_text())
        if text is not None:
            target.set_text(text)
            self.figure.canvas.draw_idle()

    def rename_label(self, annotation):
        text = request_label_text(self.figure, annotation.get_text())
        if text is not None:
            annotation.set_text(text)
            self.figure.canvas.draw_idle()

    def save(self):
        save_label_positions(self.positions_path, self.figure_name, self.current_positions(), self.current_texts())
        self.on_save()
        print(f"Saved label positions and figures: {self.figure_name}\n{self.positions_path}", flush=True)

    def reset(self):
        self.drag_state = None
        for key, position in self.initial_positions.items():
            apply_position(self.annotations[key], position)
        for key, text in self.initial_texts.items():
            self.text_targets[key].set_text(text)
        self.figure.canvas.draw_idle()

    def on_key_press(self, event):
        key = str(getattr(event, "key", "")).lower()
        if key in {"s", "ctrl+s"}:
            self.save()
        elif key == "r":
            self.reset()
        elif key == "e" and self.selected_annotation is not None:
            self.rename_label(self.selected_annotation)
        elif key == "e" and self.selected_text_target is not None:
            self.rename_text(self.selected_text_target)
