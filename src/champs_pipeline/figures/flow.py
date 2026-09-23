"""Flow diagrams and labelled boxes, drawn at print size.

A diagram is drawn on one axes that spans the figure, with coordinates in
millimetres from the lower-left corner. The text of a box is wrapped to the
box width by measuring the rendered text, and the height of the box follows
from the wrapped text, so that every box fits its content. A box can carry a
bold title above its text. Call ``style.apply()`` before drawing.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from champs_pipeline.figures import style

PADDING_MM = 2.0
LINE_SPACING = 1.25
TITLE_GAP_MM = 0.6
ROUNDING_MM = 1.2
ARROW_HEAD_SIZE = 6
# A line of box text that starts with this marker is a list item; its continuation
# lines are indented to the text after the marker.
LIST_MARKER = "• "


@dataclass
class Box:
    """The position and size of a drawn box, in millimetres."""

    left: float
    top: float
    width: float
    height: float

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top - self.height

    @property
    def center_x(self):
        return self.left + self.width / 2

    @property
    def middle_y(self):
        return self.top - self.height / 2


def canvas(width_mm, height_mm):
    """Create a figure of the given print size with one frameless axes in millimetres."""
    fig = plt.figure(figsize=style.figure_size(width_mm, height_mm), layout="none")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width_mm)
    ax.set_ylim(0, height_mm)
    ax.set_axis_off()
    return fig, ax


def pixels_to_mm(fig, pixels):
    return pixels / fig.dpi * style.MM_PER_INCH


def text_width_mm(ax, text):
    """The rendered width of a single line of text, in millimetres."""
    probe = ax.text(0, 0, text)
    width = probe.get_window_extent(ax.figure.canvas.get_renderer()).width
    probe.remove()
    return pixels_to_mm(ax.figure, width)


def text_height_mm(ax, label):
    """The rendered height of a drawn text object, in millimetres."""
    return pixels_to_mm(ax.figure, label.get_window_extent(ax.figure.canvas.get_renderer()).height)


def list_indent(ax):
    """Spaces as wide as the list marker, to indent the continuation lines of a list item."""
    space_width = text_width_mm(ax, "x x") - text_width_mm(ax, "xx")
    marker_width = text_width_mm(ax, f"{LIST_MARKER}x") - text_width_mm(ax, "x")
    return " " * max(1, round(marker_width / space_width))


def wrap(ax, text, width_mm):
    """Break each line of the text at spaces so that no line is wider than ``width_mm``."""
    indent = list_indent(ax)
    wrapped = []
    for paragraph in text.split("\n"):
        continuation = ""
        if paragraph.startswith(LIST_MARKER):
            continuation = indent
        line = ""
        for word in paragraph.split(" "):
            candidate = f"{line} {word}" if line else word
            if line.strip() and text_width_mm(ax, candidate) > width_mm:
                wrapped.append(line)
                line = f"{continuation}{word}"
            else:
                line = candidate
        wrapped.append(line)
    return "\n".join(wrapped)


def stacked_text(ax, left, top, width, title=None, text=None, centered=False):
    """Draw an optional bold title with an optional text below, both wrapped to ``width``.

    Returns the drawn text objects and the bottom of the text.
    """
    x = left
    alignment = "left"
    if centered:
        x = left + width / 2
        alignment = "center"
    labels = []
    y = top
    for content, weight in [(title, "bold"), (text, "normal")]:
        if not content:
            continue
        if labels:
            y -= TITLE_GAP_MM
        label = ax.text(x, y, wrap(ax, content, width), ha=alignment, va="top",
                        multialignment=alignment, linespacing=LINE_SPACING,
                        fontweight=weight, zorder=3)
        labels.append(label)
        y -= text_height_mm(ax, label)
    return labels, y


def text_block(ax, left, top, width, text=None, centered=False, title=None):
    """Draw unboxed text, with an optional bold title, from (``left``, ``top``); return its bottom.

    The text is aligned left, or centred on the width if ``centered`` is true.
    """
    _, bottom = stacked_text(ax, left, top, width, title, text, centered)
    return bottom


def outline(ax, box, edgecolor=style.OUTLINE, facecolor="none", rounded=False, zorder=0.5):
    """Draw the outline of a ``Box``, optionally filled and with rounded corners."""
    if rounded:
        patch = FancyBboxPatch((box.left, box.bottom), box.width, box.height,
                               boxstyle=f"round,pad=0,rounding_size={ROUNDING_MM}")
    else:
        patch = Rectangle((box.left, box.bottom), box.width, box.height)
    patch.set(facecolor=facecolor, edgecolor=edgecolor, linewidth=style.LINE_WIDTH, zorder=zorder)
    ax.add_patch(patch)


def box(ax, left, top, width, text=None, centered=True, edgecolor=style.OUTLINE, facecolor="none",
        rounded=False, title=None, middle_y=None, padding=PADDING_MM):
    """Draw a box around an optional bold title and a text, and return its geometry.

    The top-left corner of the box is at (``left``, ``top``); if ``middle_y`` is
    given, the box is instead centred vertically on it. The text is centred, or
    aligned left if ``centered`` is false. The outline is dark grey unless
    ``edgecolor`` gives the colour of a meaning; ``facecolor`` fills the box and
    ``rounded`` rounds its corners.
    """
    labels, text_bottom = stacked_text(ax, left + padding, top - padding, width - 2 * padding,
                                       title, text, centered)
    height = top - text_bottom + padding
    if middle_y is not None:
        shift = middle_y + height / 2 - top
        for label in labels:
            x, y = label.get_position()
            label.set_position((x, y + shift))
        top += shift
    drawn = Box(left, top, width, height)
    outline(ax, drawn, edgecolor=edgecolor, facecolor=facecolor, rounded=rounded)
    return drawn


def arrow(ax, start, end):
    """Draw an arrow from ``start`` to ``end``, each an (x, y) point in millimetres."""
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=ARROW_HEAD_SIZE,
                                 linewidth=style.LINE_WIDTH, color=style.OUTLINE,
                                 shrinkA=0, shrinkB=0, zorder=2))


def line(ax, points):
    """Draw a line through a sequence of (x, y) points in millimetres."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.add_line(Line2D(xs, ys, linewidth=style.LINE_WIDTH, color=style.OUTLINE,
                       solid_capstyle="projecting", solid_joinstyle="miter", zorder=2))


def crop(fig, ax, bottom_mm, margin_mm=1.0):
    """Shorten the figure from below so that it ends ``margin_mm`` under ``bottom_mm``."""
    width = ax.get_xlim()[1]
    top = ax.get_ylim()[1]
    height = top - bottom_mm + margin_mm
    fig.set_size_inches(*style.figure_size(width, height))
    ax.set_ylim(top - height, top)
