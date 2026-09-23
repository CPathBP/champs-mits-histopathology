"""Figure style following the npj Digital Medicine guidelines for display items.

The guidelines require a single sans-serif typeface at a single size across
all figures (8 pt at print size), lines of at least 1 pt, a white
background without decorative elements, colours that remain distinguishable
under common colour-vision deficiencies, and figure parts labelled with a
bold lower-case letter at the size of the remaining text. Figures are drawn
at print size, either one column (88 mm) or the full page width (180 mm).

The typeface is Arimo, which is metrically compatible with Arial. The font
files and their licence are distributed in ``fonts/`` so that figures render
identically on every system.
"""

from pathlib import Path

import matplotlib
from cycler import cycler
from matplotlib import font_manager
from matplotlib.colors import to_hex, to_rgb
from matplotlib.transforms import ScaledTranslation

FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONT_FAMILY = "Arimo"
FONT_SIZE = 8
LINE_WIDTH = 1.0
TICK_LENGTH = 3.0

MM_PER_INCH = 25.4
POINTS_PER_INCH = 72
ONE_COLUMN_MM = 88
FULL_WIDTH_MM = 180
PNG_DPI = 300

WHITE = "#FFFFFF"
BLACK = "#000000"
GREY = "#999999"
# Outlines and arrows of schematic figures: dark grey is softer than black and keeps contrast.
OUTLINE = "#404040"

# Okabe-Ito palette, distinguishable under the common colour-vision deficiencies.
BLUE = "#0072B2"
ORANGE = "#E69F00"
BLUISH_GREEN = "#009E73"
REDDISH_PURPLE = "#CC79A7"
SKY_BLUE = "#56B4E9"
VERMILLION = "#D55E00"
# One colour per meaning, used in every figure of the manuscript.
REFERENCE_A = ORANGE
REFERENCE_B = BLUE
MODEL = BLUISH_GREEN

# Okabe-Ito yellow has too little contrast on white for small marks and is left out.
CATEGORICAL = [BLUE, ORANGE, BLUISH_GREEN, REDDISH_PURPLE, SKY_BLUE, VERMILLION]

# Ordered categories such as grades use shades of one hue, so that the order is
# preserved in greyscale.
ORDERED_COLORMAP = "Blues"
ORDERED_LIGHTEST = 0.35
ORDERED_DARKEST = 0.95


def apply():
    """Register the bundled typeface and set the matplotlib defaults to the journal style."""
    for path in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(path))
    text_sizes = ["font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize",
                  "ytick.labelsize", "legend.fontsize", "legend.title_fontsize",
                  "figure.titlesize", "figure.labelsize"]
    line_widths = ["axes.linewidth", "lines.linewidth", "lines.markeredgewidth",
                   "patch.linewidth", "hatch.linewidth", "grid.linewidth",
                   "xtick.major.width", "xtick.minor.width",
                   "ytick.major.width", "ytick.minor.width"]
    matplotlib.rcParams.update({name: FONT_SIZE for name in text_sizes})
    matplotlib.rcParams.update({name: LINE_WIDTH for name in line_widths})
    matplotlib.rcParams.update({
        "font.family": FONT_FAMILY,
        "mathtext.default": "regular",
        "text.color": BLACK,
        "axes.labelcolor": BLACK,
        "axes.edgecolor": BLACK,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": cycler(color=CATEGORICAL),
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "legend.frameon": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.dpi": PNG_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def figure_size(width_mm, height_mm):
    """Return the ``figsize`` in inches for a print size given in millimetres."""
    return width_mm / MM_PER_INCH, height_mm / MM_PER_INCH


def ordered_colors(n, colormap=ORDERED_COLORMAP):
    """Return ``n`` shades of a colormap (blue by default) from light to dark."""
    colormap = matplotlib.colormaps[colormap]
    if n == 1:
        return [to_hex(colormap(ORDERED_DARKEST))]
    step = (ORDERED_DARKEST - ORDERED_LIGHTEST) / (n - 1)
    return [to_hex(colormap(ORDERED_LIGHTEST + i * step)) for i in range(n)]


def tint(color, strength=0.18):
    """A light version of a colour, mixed with white; ``strength`` 1 returns the colour itself."""
    red, green, blue = to_rgb(color)
    return to_hex((1 - strength + strength * red, 1 - strength + strength * green,
                   1 - strength + strength * blue))


def panel_label(ax, letter, dx_pt=-14, dy_pt=4):
    """Place a bold lower-case part label above the top-left corner of the axes.

    ``dx_pt`` and ``dy_pt`` offset the label from the corner, in points. A larger
    negative ``dx_pt`` keeps the label clear of wide tick labels.
    """
    offset = ScaledTranslation(dx_pt / POINTS_PER_INCH, dy_pt / POINTS_PER_INCH,
                               ax.figure.dpi_scale_trans)
    ax.text(0, 1, letter.lower(), transform=ax.transAxes + offset, fontweight="bold",
            ha="left", va="bottom")


def thousands(value):
    """Format a number as an integer with comma thousands separators (1,000)."""
    return f"{value:,.0f}"

