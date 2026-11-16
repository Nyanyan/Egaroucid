"""Place chart labels clear of text, plotted lines, markers and legends."""
from __future__ import annotations

import math
import numpy as np
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from label_editor import update_leader_line


def obstacles(axis, renderer, padding, exclude_gids=()):
    paths, points = [], []
    for line in axis.lines:
        if not line.get_visible() or line.get_gid() in exclude_gids:
            continue
        path = line.get_transform().transform_path(line.get_path())
        if line.get_linestyle() not in ("None", "none", "", " ") and len(path.vertices) > 1:
            paths.append(path)
        if line.get_marker() not in ("None", "none", "", " "):
            radius = line.get_markersize() * axis.figure.dpi / 144 + padding
            for x, y in path.vertices:
                if axis.bbox.contains(x, y):
                    points.append(Bbox.from_bounds(x-radius, y-radius, 2*radius, 2*radius))
    legend = axis.get_legend()
    legends = [legend.get_window_extent(renderer).padded(padding)] if legend else []
    return paths, points, legends


def leader_path(annotation, renderer):
    annotation.update_positions(renderer)
    patch = annotation.arrow_patch
    if patch is not None and patch.get_visible():
        return patch.get_transform().transform_path(patch.get_path())
    return None


def inspect_layout(figure, annotations):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    padding = figure.dpi / 72
    report = {key: [] for key in ("text", "line", "point", "legend", "leader", "outside", "leader_outside")}
    for axis in figure.axes:
        labels = {uid: a for uid, a in annotations.items() if a.axes is axis}
        paths, points, legends = obstacles(axis, renderer, padding)
        reference_paths = obstacles(axis, renderer, padding, ("human-reference-line",))[0]
        boxes = {uid: Text.get_window_extent(a, renderer).padded(padding) for uid, a in labels.items()}
        leaders = {uid: leader_path(a, renderer) for uid, a in labels.items()}
        for i, (uid, box) in enumerate(boxes.items()):
            for other in list(boxes)[i+1:]:
                if box.overlaps(boxes[other]):
                    report["text"].append([uid, other])
            # Opaque reference captions mask reference lines behind them.
            caption_paths = reference_paths if uid.startswith("reference/") else paths
            if any(path.intersects_bbox(box, filled=False) for path in caption_paths):
                report["line"].append(uid)
            if any(box.overlaps(point) for point in points):
                report["point"].append(uid)
            if any(box.overlaps(legend) for legend in legends):
                report["legend"].append(uid)
            if not (axis.bbox.x0 <= box.x0 and box.x1 <= axis.bbox.x1
                    and axis.bbox.y0 <= box.y0 and box.y1 <= axis.bbox.y1):
                report["outside"].append(uid)
            for other, path in leaders.items():
                if uid != other and path is not None and path.intersects_bbox(box, filled=False):
                    report["leader"].append([uid, other])
            path = leaders[uid]
            if path is not None:
                extent = path.get_extents()
                bounds = axis.bbox.padded(padding)
                if not (bounds.x0 <= extent.x0 and extent.x1 <= bounds.x1
                        and bounds.y0 <= extent.y0 and extent.y1 <= bounds.y1):
                    report["leader_outside"].append(uid)
    return report


def route_leaders(figure, annotations):
    """Bend a leader only when its straight path crosses another label."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    padding = figure.dpi / 72
    styles = ["arc3,rad=0"] + [f"arc3,rad={sign*radius}" for radius in (.12, .25, .4, .65, 1., 1.5)
                              for sign in (1, -1)]
    for axis in figure.axes:
        labels = {uid: a for uid, a in annotations.items() if a.axes is axis}
        boxes = {uid: Text.get_window_extent(a, renderer).padded(2*padding) for uid, a in labels.items()}
        legend = axis.get_legend()
        legend_boxes = [legend.get_window_extent(renderer).padded(padding)] if legend else []
        for uid, a in labels.items():
            if a.arrow_patch is None or not a.arrow_patch.get_visible():
                continue
            blocked = [box for other, box in boxes.items() if other != uid] + legend_boxes
            best, score = styles[0], float("inf")
            for style in styles:
                a.arrow_patch.set_connectionstyle(style)
                path = leader_path(a, renderer)
                conflicts = sum(path.intersects_bbox(box, filled=False) for box in blocked)
                extent = path.get_extents()
                if not (axis.bbox.x0 <= extent.x0 and extent.x1 <= axis.bbox.x1
                        and axis.bbox.y0 <= extent.y0 and extent.y1 <= axis.bbox.y1):
                    conflicts += 1
                if conflicts < score:
                    best, score = style, conflicts
                if conflicts == 0:
                    break
            a.arrow_patch.set_connectionstyle(best)
            a.update_positions(renderer)


def arrange_labels(figure, annotations, fixed_keys=()):
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    unit = figure.dpi / 72
    padding = 2*unit
    fixed_keys = set(fixed_keys)
    for axis in figure.axes:
        labels = {uid: a for uid, a in annotations.items() if a.axes is axis}
        if not labels:
            continue
        paths, points, legends = obstacles(axis, renderer, padding)
        bounds = axis.bbox.padded(-3*unit)
        anchors = {uid: axis.transData.transform(a.xy) for uid, a in labels.items()}
        placed = [Text.get_window_extent(a, renderer).padded(padding)
                  for uid, a in labels.items() if uid in fixed_keys]
        fixed_leaders = [leader_path(a, renderer) for uid, a in labels.items() if uid in fixed_keys]
        fixed_leaders = [path for path in fixed_leaders if path is not None]

        def priority(uid):
            a = labels[uid]
            crowding = sum(np.linalg.norm(anchors[uid]-p) < 70*unit for p in anchors.values())
            box = Text.get_window_extent(a, renderer)
            return (box.width*box.height, crowding)

        for uid in sorted(set(labels)-fixed_keys, key=priority, reverse=True):
            a = labels[uid]
            box = Text.get_window_extent(a, renderer)
            width, height = box.width + 2*padding, box.height + 2*padding
            anchor = anchors[uid]
            centers = [np.array([(box.x0+box.x1)/2, (box.y0+box.y1)/2])]
            for radius in (12, 20, 30, 44, 60, 80, 105, 135, 170, 210, 260, 320):
                for angle in range(0, 360, 15):
                    theta = math.radians(angle)
                    centers.append(anchor + radius*unit*np.array([math.cos(theta), math.sin(theta)]))
            centers.sort(key=lambda center: np.linalg.norm(center-anchor))
            best, best_score = centers[0], float("inf")
            for center in centers:
                candidate = Bbox.from_bounds(center[0]-width/2, center[1]-height/2, width, height)
                if not (bounds.x0 <= candidate.x0 and candidate.x1 <= bounds.x1
                        and bounds.y0 <= candidate.y0 and candidate.y1 <= bounds.y1):
                    continue
                conflicts = (sum(candidate.overlaps(v) for v in placed)
                             + sum(candidate.overlaps(v) for v in points)
                             + sum(candidate.overlaps(v) for v in legends)
                             + sum(v.intersects_bbox(candidate, filled=False) for v in paths)
                             + sum(v.intersects_bbox(candidate, filled=False) for v in fixed_leaders))
                score = conflicts*1e8 + np.linalg.norm(center-anchor)
                if score < best_score:
                    best, best_score = center, score
                if conflicts == 0:
                    break
            a.set_ha("center")
            a.set_va("center")
            a.xyann = tuple((best-anchor)/unit)
            update_leader_line(a)
            a.update_positions(renderer)
            placed.append(Text.get_window_extent(a, renderer).padded(padding))
    route_leaders(figure, annotations)
    return inspect_layout(figure, annotations)
