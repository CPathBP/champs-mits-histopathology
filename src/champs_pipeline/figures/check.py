"""Verification of a figure against the journal requirements before it is saved.

The checks cover the properties that can be measured on the drawn figure:
the figure width, a white background, the typeface and size of every text
element, and the width of every visible line, marker edge and outline.
``layout_problems`` checks that text stays on the canvas, clear of other text
and clear of the data drawn in the axes a figure names. Colour choice and
wording are not checked.
"""

import numpy as np
from matplotlib.collections import Collection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from champs_pipeline.figures import style

TOLERANCE = 1e-3
NO_LINE_STYLES = ("None", "none", "", " ")
NO_MARKERS = (None, "None", "none", "", " ")


def problems(fig):
    """Return a description of each requirement the figure violates; empty if it complies."""
    fig.draw_without_rendering()
    found = []
    width_mm = fig.get_figwidth() * style.MM_PER_INCH
    if width_mm > style.FULL_WIDTH_MM + TOLERANCE:
        found.append(f"the figure is {width_mm:.0f} mm wide; "
                     f"the page width is {style.FULL_WIDTH_MM} mm")
    if not is_white(fig.get_facecolor()):
        found.append("the figure background is not white")
    for ax in fig.axes:
        if ax.get_visible() and ax.patch.get_visible() and not is_white(ax.get_facecolor()):
            found.append("an axes background is not white")
    for artist in fig.findobj():
        if not artist.get_visible():
            continue
        if isinstance(artist, Text):
            found.extend(text_problems(artist))
        elif isinstance(artist, Line2D):
            found.extend(line_problems(artist))
        elif isinstance(artist, Patch):
            found.extend(patch_problems(artist))
        elif isinstance(artist, Collection):
            found.extend(collection_problems(artist))
    return found


def is_white(color):
    return to_rgba(color)[:3] == (1.0, 1.0, 1.0)


def is_thin(width):
    return 0 < width < style.LINE_WIDTH - TOLERANCE


def text_problems(text):
    if not text.get_text().strip():
        return []
    found = []
    if abs(text.get_fontsize() - style.FONT_SIZE) > TOLERANCE:
        found.append(f"text {text.get_text()!r} is {text.get_fontsize():g} pt, "
                     f"not {style.FONT_SIZE} pt")
    if text.get_fontname() != style.FONT_FAMILY:
        found.append(f"text {text.get_text()!r} is set in {text.get_fontname()}, "
                     f"not {style.FONT_FAMILY}")
    return found


def line_problems(line):
    found = []
    if line.get_linestyle() not in NO_LINE_STYLES and is_thin(line.get_linewidth()):
        found.append(f"a line ({describe(line)}) is {line.get_linewidth():g} pt wide")
    if line.get_marker() not in NO_MARKERS and is_thin(line.get_markeredgewidth()):
        found.append(f"a marker edge ({describe(line)}) is "
                     f"{line.get_markeredgewidth():g} pt wide")
    return found


def patch_problems(patch):
    has_edge = to_rgba(patch.get_edgecolor())[3] > 0
    if has_edge and is_thin(patch.get_linewidth()):
        return [f"an outline ({describe(patch)}) is {patch.get_linewidth():g} pt wide"]
    return []


def collection_problems(collection):
    edges = collection.get_edgecolor()
    if len(edges) == 0 or max(edge[3] for edge in edges) == 0:
        return []
    thin = [width for width in collection.get_linewidths() if is_thin(width)]
    if thin:
        return [f"a line or edge ({describe(collection)}) is {min(thin):g} pt wide"]
    return []


def layout_problems(fig, data_axes=(), clear_axes=False):
    """Describe text that leaves the canvas, overlaps other text or covers data.

    The data are the visible lines and collections of ``data_axes``. A line is
    tested along its drawn path, a collection by its extent. With
    ``clear_axes``, no text may enter the ``data_axes`` at all. Overlap between
    texts is tested for upright text only: the extent of tilted text is its
    upright bounding box, which meets a tilted neighbour even when the letters
    do not.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    undrawn = undrawn_tick_labels(fig)
    boxes = [(text, text.get_window_extent(renderer)) for text in fig.findobj(Text)
             if text.get_visible() and text.get_text().strip() and text not in undrawn]
    found = []
    for text, box in boxes:
        if not fig.bbox.contains(box.x0, box.y0) or not fig.bbox.contains(box.x1, box.y1):
            found.append(f"text {text.get_text()!r} leaves the figure")
    upright = [(text, box) for text, box in boxes if text.get_rotation() % 90 == 0]
    for index, (first, first_box) in enumerate(upright):
        for second, second_box in upright[index + 1:]:
            if first_box.overlaps(second_box):
                found.append(f"text {first.get_text()!r} overlaps {second.get_text()!r}")
    for ax in data_axes:
        for text, box in boxes:
            if clear_axes and box.overlaps(ax.bbox):
                found.append(f"text {text.get_text()!r} enters the data area of {ax.get_title()!r}")
            elif text.axes is ax and covers_data(ax, box, renderer):
                found.append(f"text {text.get_text()!r} covers data in {ax.get_title()!r}")
    return found


def undrawn_tick_labels(fig):
    """Tick labels outside the view limits of their axis, which matplotlib does not draw."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            tolerance = (high - low) * 1e-9
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low - tolerance <= tick.get_loc() <= high + tolerance:
                    hidden.update({tick.label1, tick.label2})
    return hidden


def covers_data(ax, box, renderer):
    """Whether a text box meets a drawn line or collection of ``ax``."""
    for line in ax.lines:
        if line.get_visible() and line_meets(line, box, ax.figure.dpi):
            return True
    for collection in ax.collections:
        extent = collection_extent(collection, ax, renderer)
        if collection.get_visible() and extent is not None and box.overlaps(extent):
            return True
    return False


def line_meets(line, box, dpi):
    """Whether the drawn path of a line, widened by half its stroke, meets a box."""
    points = np.column_stack((line.get_xdata(orig=False), line.get_ydata(orig=False)))
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        return False
    if len(points) == 1 or line.get_linestyle() in NO_LINE_STYLES:
        corner = line.get_transform().transform(points)
        return any(box.padded(line.get_markersize()).contains(x, y) for x, y in corner)
    path = line.get_transform().transform_path(Path(points))
    return path.intersects_bbox(box.padded(line.get_linewidth() * dpi / 144), filled=False)


def collection_extent(collection, ax, renderer):
    """The display extent of a collection, widened by 2 pixels; None when it has none."""
    extent = collection.get_window_extent(renderer)
    if not np.isfinite(extent.get_points()).all() or not extent.width or not extent.height:
        extent = collection.get_datalim(ax.transData).transformed(ax.transData)
    if not np.isfinite(extent.get_points()).all():
        return None
    return Bbox.from_extents(extent.x0 - 2, extent.y0 - 2, extent.x1 + 2, extent.y1 + 2)


def describe(artist):
    label = artist.get_label()
    if label and not label.startswith("_"):
        return f"{type(artist).__name__} {label!r}"
    return type(artist).__name__
