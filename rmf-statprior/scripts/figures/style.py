"""
Shared style for all figures of the paper. Import it before plotting:

    import style; style.apply()

Colours have a fixed meaning across figures: uniform prior (grey), statistical prior (blue), statistical prior
alone / independent coupling (light blue), kernel density estimate (orange), real data (dark grey). Density maps use one
sequential colour map in every panel, so that methods are distinguished by panel titles rather than by colour.
"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

UNIFORM = "#8c8c8c"
STAT = "#1f5fa8"
PRIOR = "#7fa7d4"
KDE = "#d08a1a"
DATA = "#333333"
FLOOR = "#eef4ee"
FLOOR_TEXT = "#5a7d5a"
GRID = "#e6e6e6"
LAND = "#e4e4e4"
EDGE = "#b8b8b8"
# categorical colours for the five published baselines (muted, distinguishable in grey scale by marker)
BASELINES = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]

# white -> blue, used for every density panel (data, uniform prior, statistical prior)
DENSITY = LinearSegmentedColormap.from_list(
    "density", ["#ffffff", "#dbe7f4", "#9dbfe2", "#4f86c3", "#1f5fa8", "#0d2f5e"])

TITLE_DATA = "Held-out data"
TITLE_UNIFORM = "Uniform prior"
TITLE_STAT = "Statistical prior (ours)"


def apply(font_size=8):
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True, "axes.unicode_minus": False,
        "font.size": font_size, "axes.titlesize": font_size + 0.5, "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1, "ytick.labelsize": font_size - 1, "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlepad": 4,
        "legend.frameon": False, "lines.linewidth": 1.1, "lines.markersize": 4,
        "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def clean(ax, grid=True):
    """Light horizontal grid behind the data; no top/right spines."""
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if grid:
        ax.grid(axis="y", color=GRID, lw=0.5, zorder=0)
        ax.set_axisbelow(True)


def floor_band(ax, lo, hi=1.0, label=None):
    """Shaded band marking the noise floor."""
    ax.axhspan(lo, hi, color=FLOOR, lw=0, zorder=0, label=label)


def frame(ax, color=EDGE, lw=0.6):
    """Thin uniform frame for map and density panels."""
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(color); sp.set_linewidth(lw)
