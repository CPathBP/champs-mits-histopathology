"""Schematic elements of study-overview figures, drawn in millimetre coordinates.

The elements follow the conventions of overview figures in computational
pathology: a slide thumbnail with a scale bar, a zoom from the slide to a stack
of tiles, the encoder as a tapering block, feature vectors as columns of cells,
and plain glyphs for documents, glass slides, tables and readers. Outlines are
1 pt wide and text uses the figure style, so a schematic passes the figure
checks. Draw on an axes from ``flow.canvas`` after ``style.apply()``.
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

from champs_pipeline.figures import style
from champs_pipeline.figures.flow import ROUNDING_MM, Box

SCALE_BAR_WIDTH = 2.0
ICON_DOT_MM = 0.65
ZOOM_SHADE = style.tint(style.GREY, 0.35)
PLACEHOLDER = style.tint(style.GREY, 0.6)


def image(ax, picture, left, top, width, height):
    """Place an image with its top-left corner at (``left``, ``top``)."""
    ax.imshow(np.asarray(picture), extent=(left, left + width, top - height, top),
              aspect="auto", interpolation="antialiased", zorder=1)
    return Box(left, top, width, height)


def rectangle(ax, box, edgecolor=style.OUTLINE, facecolor="none", rounded=False, hatch=None,
              zorder=2.0):
    """Draw the outline of a ``Box``, optionally filled, rounded or hatched."""
    if rounded:
        patch = FancyBboxPatch((box.left, box.bottom), box.width, box.height,
                               boxstyle=f"round,pad=0,rounding_size={ROUNDING_MM}")
    else:
        patch = Rectangle((box.left, box.bottom), box.width, box.height)
    patch.set(facecolor=facecolor, edgecolor=edgecolor, linewidth=style.LINE_WIDTH,
              hatch=hatch, zorder=zorder)
    ax.add_patch(patch)


def line(ax, points, color=style.OUTLINE, width=style.LINE_WIDTH, zorder=2.0):
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    ax.add_line(Line2D(xs, ys, color=color, linewidth=width, zorder=zorder,
                       solid_capstyle="butt"))


def scale_bar(ax, left, y, length_mm, label):
    """A horizontal scale bar of ``length_mm`` on the page, with its label below."""
    line(ax, [(left, y), (left + length_mm, y)], width=SCALE_BAR_WIDTH)
    ax.text(left + length_mm / 2, y - 0.8, label, ha="center", va="top")


def zoom(ax, source, target):
    """A light wedge from the right edge of a source ``Box`` to the left edge of a target."""
    points = [(source.right, source.top), (target.left, target.top),
              (target.left, target.bottom), (source.right, source.bottom)]
    ax.add_patch(Polygon(points, closed=True, facecolor=ZOOM_SHADE, edgecolor="none",
                         zorder=0.5))


def tile_stack(ax, tiles, left, top, size, offset):
    """Tiles as an overlapping stack from the back (top left) to the front (bottom right)."""
    for index, tile in enumerate(tiles):
        tile_box = image(ax, tile, left + index * offset, top - index * offset, size, size)
        rectangle(ax, tile_box, edgecolor=style.WHITE, zorder=1.5)
    depth = (len(tiles) - 1) * offset
    return Box(left, top, size + depth, size + depth)


def tapering_block(ax, left, top, width, height, taper, colour, label):
    """A block that narrows from left to right, the usual glyph of an encoder."""
    points = [(left, top), (left + width, top - taper), (left + width, top - height + taper),
              (left, top - height)]
    ax.add_patch(Polygon(points, closed=True, facecolor=style.tint(colour),
                         edgecolor=colour, linewidth=style.LINE_WIDTH, zorder=2))
    ax.text(left + width / 2, top - height / 2, label, ha="center", va="center", zorder=3)
    return Box(left, top, width, height)


def cells(ax, left, top, cell, rows, columns, colour):
    """A vector or a matrix drawn as cells in shades of one colour, with an outer outline."""
    shades = [style.tint(colour, strength) for strength in (0.25, 0.5, 0.75)]
    for row in range(rows):
        for column in range(columns):
            shade = shades[(row + 2 * column) % len(shades)]
            ax.add_patch(Rectangle((left + column * cell, top - (row + 1) * cell), cell, cell,
                                   facecolor=shade, edgecolor=style.WHITE,
                                   linewidth=style.LINE_WIDTH, zorder=2))
    outline = Box(left, top, columns * cell, rows * cell)
    rectangle(ax, outline, edgecolor=colour, zorder=2.5)
    return outline


def document(ax, left, top, width, height, fold=2.5):
    """A page with a folded top-right corner."""
    points = [(left, top), (left + width - fold, top), (left + width, top - fold),
              (left + width, top - height), (left, top - height)]
    ax.add_patch(Polygon(points, closed=True, facecolor=style.WHITE, edgecolor=style.OUTLINE,
                         linewidth=style.LINE_WIDTH, zorder=1))
    line(ax, [(left + width - fold, top), (left + width - fold, top - fold),
              (left + width, top - fold)])
    return Box(left, top, width, height)


def text_lines(ax, box, fractions, first_offset=3.0, spacing=2.0):
    """Grey placeholder lines of text inside a ``Box``, one per fraction of its width."""
    for index, fraction in enumerate(fractions):
        y = box.top - first_offset - index * spacing
        line(ax, [(box.left + 1.5, y), (box.left + 1.5 + fraction * (box.width - 3), y)],
             color=PLACEHOLDER)


def glass_slide(ax, left, top, width, height, picture=None, tissue=None, zorder=1.0):
    """A glass slide with a label area on the left and tissue, as an image or a shape.

    Give each slide of a stack a higher ``zorder`` than the slide behind it.
    """
    slide = Box(left, top, width, height)
    rectangle(ax, slide, facecolor=style.WHITE, zorder=zorder)
    label_width = width * 0.28
    rectangle(ax, Box(left, top, label_width, height), facecolor=style.tint(style.GREY, 0.35),
              zorder=zorder + 0.1)
    tissue_box = Box(left + label_width + 0.8, top - 0.6, width - label_width - 1.6, height - 1.2)
    if picture is not None:
        scale = min(tissue_box.width / picture.width, tissue_box.height / picture.height)
        fit_width = picture.width * scale
        fit_height = picture.height * scale
        image(ax, picture, tissue_box.left + (tissue_box.width - fit_width) / 2,
              tissue_box.top - (tissue_box.height - fit_height) / 2, fit_width, fit_height)
    elif tissue is not None:
        ax.add_patch(Polygon(_blob(tissue_box), closed=True, facecolor=tissue,
                             edgecolor="none", zorder=zorder + 0.2))
    return slide


def _blob(box):
    angles = np.linspace(0, 2 * np.pi, 13)[:-1]
    radii = 0.8 + 0.2 * np.cos(3 * angles)
    center_x = box.left + box.width / 2
    center_y = box.top - box.height / 2
    return [(center_x + r * np.cos(a) * box.width / 2, center_y + r * np.sin(a) * box.height / 2)
            for r, a in zip(radii, angles)]


def table_glyph(ax, left, top, width, height, rows=4, columns=3):
    """A small table with a shaded header row."""
    row_height = height / rows
    column_width = width / columns
    for row in range(rows):
        fill = style.WHITE
        if row == 0:
            fill = style.tint(style.GREY, 0.35)
        for column in range(columns):
            rectangle(ax, Box(left + column * column_width, top - row * row_height,
                              column_width, row_height), facecolor=fill, zorder=1)
    return Box(left, top, width, height)


def person(ax, center_x, bottom, height, colour):
    """A person glyph: a head above rounded shoulders."""
    radius = height * 0.2
    body_width = height * 0.75
    body_height = height * 0.5
    ax.add_patch(FancyBboxPatch((center_x - body_width / 2, bottom), body_width, body_height,
                                boxstyle=f"round,pad=0,rounding_size={body_width / 2.5}",
                                facecolor=style.tint(colour, 0.35), edgecolor=colour,
                                linewidth=style.LINE_WIDTH, zorder=2))
    ax.add_patch(Circle((center_x, bottom + height - radius), radius,
                        facecolor=style.tint(colour, 0.35), edgecolor=colour,
                        linewidth=style.LINE_WIDTH, zorder=2))


def group_of_people(ax, left, bottom, height, colour):
    """Three overlapping person glyphs, the usual symbol for a group of readers."""
    spacing = height * 0.55
    # The middle glyph is drawn last, so that it stands in front of the other two.
    for index in (0, 2, 1):
        person(ax, left + height * 0.4 + index * spacing, bottom, height, colour)
    return Box(left, bottom + height, height * 0.8 + 2 * spacing, height)


def numbered_squares(ax, left, top, size, gap, labels, highlight, shades):
    """A row of squares with a label each; the square at index ``highlight`` is dark.

    ``shades`` gives the light fill and the dark fill. Returns the drawn row as a ``Box``.
    """
    light, dark = shades
    for index, label in enumerate(labels):
        fill = light
        text_colour = style.BLACK
        if index == highlight:
            fill = dark
            text_colour = style.WHITE
        square = Box(left + index * (size + gap), top, size, size)
        rectangle(ax, square, edgecolor="none", facecolor=fill)
        ax.text(square.center_x, square.middle_y, label, ha="center", va="center",
                color=text_colour, zorder=3)
    return Box(left, top, len(labels) * size + (len(labels) - 1) * gap, size)


def labelled_circles(ax, left, top, diameter, gap, labels, highlight, shades):
    """A row of circles with a label each, such as site markers; one circle is dark."""
    light, dark = shades
    radius = diameter / 2
    for index, label in enumerate(labels):
        fill = light
        text_colour = style.BLACK
        if index == highlight:
            fill = dark
            text_colour = style.WHITE
        center_x = left + radius + index * (diameter + gap)
        ax.add_patch(Circle((center_x, top - radius), radius, facecolor=fill, edgecolor="none",
                            zorder=2))
        ax.text(center_x, top - radius, label, ha="center", va="center", color=text_colour,
                zorder=3)
    return Box(left, top, len(labels) * diameter + (len(labels) - 1) * gap, diameter)


def _plot_frame(ax, left, top, width, height):
    line(ax, [(left, top), (left, top - height), (left + width, top - height)])
    return Box(left, top, width, height)


def curve_icon(ax, left, top, width, height, colour):
    """A schematic ROC curve above the chance diagonal; an icon, not data."""
    frame = _plot_frame(ax, left, top, width, height)
    line(ax, [(left, frame.bottom), (left + width, top)], color=style.tint(style.GREY, 0.7))
    xs = np.linspace(0, 1, 40)
    ys = 1 - (1 - xs) ** 5
    line(ax, [(left + x * width, frame.bottom + y * height) for x, y in zip(xs, ys)],
         color=colour, width=1.5)
    return frame


def forest_icon(ax, left, top, width, height, colour):
    """A schematic forest plot: estimates with intervals around a reference line."""
    line(ax, [(left, top - height), (left + width, top - height)])
    center = left + width * 0.55
    line(ax, [(center, top), (center, top - height)], color=style.tint(style.GREY, 0.7))
    offsets = [0.1, -0.08, 0.16, 0.03]
    half_widths = [0.2, 0.14, 0.26, 0.16]
    step = (height - 1) / len(offsets)
    for index, (offset, half_width) in enumerate(zip(offsets, half_widths)):
        y = top - step * (index + 0.5)
        x = center + offset * width
        line(ax, [(x - half_width * width, y), (x + half_width * width, y)], color=colour)
        ax.add_patch(Circle((x, y), ICON_DOT_MM, facecolor=colour, edgecolor="none", zorder=3))
    return Box(left, top, width, height)


def bar_icon(ax, left, top, width, height, colour):
    """Schematic bars rising across ordered categories; an icon, not data."""
    frame = _plot_frame(ax, left, top, width, height)
    heights = [0.3, 0.5, 0.72, 0.92]
    spacing = width / len(heights)
    for index, fraction in enumerate(heights):
        bar = Box(left + (index + 0.25) * spacing, frame.bottom + fraction * height,
                  0.6 * spacing, fraction * height)
        rectangle(ax, bar, edgecolor="none", facecolor=style.tint(colour, 0.4 + 0.2 * index))
    return frame


def swatch(ax, left, top, size, text, facecolor, edgecolor=style.OUTLINE, hatch=None):
    """A legend square with its text on the right; returns the right end of the text."""
    rectangle(ax, Box(left, top, size, size), edgecolor=edgecolor, facecolor=facecolor,
              hatch=hatch)
    label = ax.text(left + size + 1, top - size / 2, text, va="center")
    width = label.get_window_extent(ax.figure.canvas.get_renderer()).width
    return left + size + 1 + width / ax.figure.dpi * style.MM_PER_INCH
