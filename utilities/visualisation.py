import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import binom, binomtest

TEXT_COLOR = "#0b0b0b"
MUTED_TEXT = "#52514e"
GRID_COLOR = "#ebe9e5"

BLUE = "#3A86FF"
ORANGE = "#FB5607"
YELLOW = "#FFBE0B"
PINK = "#FF006E"
PURPLE = "#8338EC"
GREY = "#bfbdc1"

BAR_COLOR = BLUE
SERIES_COLORS = [BLUE, ORANGE, PURPLE, YELLOW, PINK, GREY]

def _percent_lim(values_min):
    """Default value-axis range for percent charts: 0-100%, extended below 0 only if a value is negative."""
    lo = 0 if values_min >= 0 else 10 * np.floor(values_min / 10)
    return (lo, 100)


def model_family(name):
    """The family name from a model's display name: its leading letters, e.g. 'GPT-4o (2024)' -> 'GPT'."""
    return re.match(r"[A-Za-z]+", str(name)).group()


def add_family_labels(ax, names, horizontal=False):
    """Divide the categories into model families: a thin line between families and the
    family name at the top of each section (at the right edge when horizontal).

    names: the category labels in the order they are drawn (one per bar or group).
    Families must be next to each other, as in MODEL_ORDER.
    """
    families = [model_family(n) for n in names]
    start = 0
    for i in range(1, len(families) + 1):
        if i < len(families) and families[i] == families[start]:
            continue
        middle = (start + i - 1) / 2
        if horizontal:
            ax.text(1.0, middle, families[start], transform=ax.get_yaxis_transform(),
                    ha="right", va="center", color=TEXT_COLOR, fontsize=11)
        else:
            ax.text(middle, 0.97, families[start], transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=TEXT_COLOR, fontsize=11)
        if i < len(families):  # line between this family and the next
            line = ax.axhline if horizontal else ax.axvline
            line(i - 0.5, color=GREY, linewidth=1, zorder=1)
        start = i


def bar_chart(values, title, xlabel="Score", ax=None, show_values=True, value_fmt="{:.1f}%", percent=True, colors=None,
              horizontal=False, value_lim=None, average_line=True, show_families=True):
    """Bar chart of one value per category.

    values: pd.Series, index = category labels (in index order), values = bar lengths.
    Handles negative values: bars grow from a 0 baseline in either direction.
    xlabel: label for the value axis (the x-axis when horizontal, the y-axis when standing).
    percent: if True, values are fractions (0.31) shown as percentages (31%).
    value_fmt: format for the value labels; defaults to "{:.0f}%" with percent, else "{:.2f}".
    colors: optional dict {category label: colour}, e.g. MODEL_COLORS; labels not in it use BAR_COLOR.
    horizontal: True = bars grow right, categories top to bottom; False = standing columns, categories left to right.
    value_lim: optional (min, max) for the value axis, in the units shown.
               With percent=True it defaults to 0-100% (extended below 0 if any value is negative).
    average_line: if True, draw a line across the chart at the mean of the bars, labelled with its value.
    show_families: if True, separate the model families with thin lines and label each one (see add_family_labels).
    Returns the matplotlib Axes so you can tweak it further.
    """
    if percent:
        values = values * 100
    if value_fmt is None:
        value_fmt = "{:.0f}%" if percent else "{:.2f}"

    if ax is None:
        figsize = (8, 0.45 * len(values) + 1.2) if horizontal else (0.6 * len(values) + 2, 5)
        _, ax = plt.subplots(figsize=figsize)

    labels = [str(i) for i in values.index]
    pos = range(len(values))
    bar_colors = [(colors or {}).get(label, BAR_COLOR) for label in values.index]

    if horizontal:
        ax.barh(pos, values.values, height=0.6, color=bar_colors, zorder=2)
        # first item at the top
        ax.set_yticks(list(pos), labels, color=TEXT_COLOR)
        ax.invert_yaxis()
        value_axis = ax.xaxis
        ax.axvline(0, color=MUTED_TEXT, linewidth=1, zorder=3)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=1, zorder=0)
        ax.set_xlabel(xlabel, color=MUTED_TEXT)
    else:
        ax.bar(pos, values.values, width=0.6, color=bar_colors, zorder=2)
        # first item on the left; tilt long names so they fit
        ax.set_xticks(list(pos), labels, color=TEXT_COLOR, rotation=35, ha="right", rotation_mode="anchor")
        value_axis = ax.yaxis
        ax.axhline(0, color=MUTED_TEXT, linewidth=1, zorder=3)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
        ax.set_ylabel(xlabel, color=MUTED_TEXT)

    # recessive ticks, no box
    ax.tick_params(axis="both", length=0, colors=MUTED_TEXT)
    ax.tick_params(axis="y" if horizontal else "x", colors=TEXT_COLOR)
    for side in ax.spines.values():
        side.set_visible(False)

    # value at the tip of each bar, outside the bar end
    if show_values:
        span = max(values.abs().max(), 1e-9)
        pad = 0.01 * span
        for p_, v in zip(pos, values.values):
            if np.isnan(v):  # no bar for this category (e.g. hidden on purpose), so no label
                continue
            tip = v + (pad if v >= 0 else -pad)
            if horizontal:
                ax.text(tip, p_, value_fmt.format(v), va="center", ha="left" if v >= 0 else "right",
                        color=MUTED_TEXT, fontsize=9)
            else:
                ax.text(p_, tip, value_fmt.format(v), ha="center", va="bottom" if v >= 0 else "top",
                        color=MUTED_TEXT, fontsize=9)
        lo, hi = min(values.min(), 0), max(values.max(), 0)
        limits = (lo - 0.12 * span * (lo < 0), hi + 0.12 * span)
        ax.set_xlim(*limits) if horizontal else ax.set_ylim(*limits)

    # mean of all bars as a reference line across the chart
    if average_line:
        avg = values.mean()
        label = "Average=" + value_fmt.format(avg)
        if horizontal:
            ax.axvline(avg, color=TEXT_COLOR, linewidth=1.5, linestyle=(0, (4, 3)), zorder=4)
            ax.text(avg, -0.6, label, ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        else:
            ax.axhline(avg, color=TEXT_COLOR, linewidth=1.5, linestyle=(0, (4, 3)), zorder=4)
            ax.text(len(values) - 0.5, avg, label + " ", ha="right", va="bottom", color=TEXT_COLOR, fontsize=9)

    if value_lim is None and percent:
        value_lim = _percent_lim(values.min())
    if value_lim is not None:
        ax.set_xlim(*value_lim) if horizontal else ax.set_ylim(*value_lim)

    if percent:
        value_axis.set_major_formatter(PercentFormatter(decimals=0))

    if show_families:
        add_family_labels(ax, values.index, horizontal)

    ax.set_title(title, loc="center", color=TEXT_COLOR, fontsize=12, pad=10)
    return ax


def bar_chart_with_reference(values, reference, title, xlabel, colors=None,
                             value_label="Value", reference_label="Reference", percent=True, value_lim=None,
                             show_value_legend=False, show_families=True):
    """Standing bar chart of `values`, with a black marker on each bar at that category's `reference` value.

    The gap (values - reference) is printed above each bar together with the value, e.g. "87.1% (+33.8%)".

    values, reference: pd.Series with the same index (e.g. model names), as fractions (0.87) when percent=True.
    colors: optional dict {category label: colour}, like bar_chart; labels not in it use BAR_COLOR.
    value_label, reference_label: legend names for the bars and the markers.
    show_value_legend: also list the bars in the legend (default False: only the markers are listed).
    show_families: if True, separate and label the model families, as in bar_chart.
    value_lim: optional (min, max) for the value axis, in the units shown. With percent=True it defaults to
               0-100%, extended below 0 if any bar or marker is negative.
    Returns the matplotlib Axes.
    """
    reference = reference.reindex(values.index)
    gap = values - reference
    scale = 100 if percent else 1
    fmt = "{:.1%}" if percent else "{:.2f}"
    gap_fmt = "({:+.1f}%)" if percent else "({:+.2f})"

    ax = bar_chart(
        values,
        title="",  # set below, above the legend
        xlabel=xlabel,
        percent=percent,
        show_values=False,  # custom labels below show the value and the gap
        colors=colors,
        value_lim=value_lim if value_lim is not None or not percent
                  else _percent_lim(min(values.min(), reference.min()) * 100),  # keep markers in view
        average_line=False,  # the markers are the comparison here
        show_families=show_families,
    )

    # bars can reach 100%: leave room above them for the value labels and family names,
    # but keep the ticks at 100% and below
    if percent and value_lim is None:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (25 if show_families else 12))
        ax.set_yticks([t for t in ax.get_yticks() if lo <= t <= hi])

    for i, label in enumerate(values.index):
        # marker at the reference value for this category
        ax.hlines(reference[label] * scale, i - 0.3, i + 0.3, color=TEXT_COLOR, linewidth=2.5, zorder=4)
        # value and gap above the bar (or above the marker, whichever is higher)
        # (never below 0, so labels don't sit on top of negative bars)
        ax.text(i, max(values[label], reference[label], 0) * scale + 1.5 * (scale / 100 if percent else 0.01),
                fmt.format(values[label]) + "\n" + gap_fmt.format(gap[label] * scale),
                ha="center", va="bottom", fontsize=8, color=MUTED_TEXT)

    # legend for the reference markers (and the bars, if asked for)
    handles = [Line2D([], [], color=TEXT_COLOR, linewidth=2.5, label=reference_label)]
    if show_value_legend:
        handles.insert(0, Patch(color=BAR_COLOR, label=value_label))
    ax.legend(
        handles=handles,
        # 1.04 leaves a gap between the top of the plot and the legend
        loc="lower center", bbox_to_anchor=(0.5, 1.04), ncol=len(handles), frameon=False, fontsize=9,
    )
    ax.set_title(title, loc="center", color=TEXT_COLOR, fontsize=12, pad=40)  # above the legend
    return ax


def grouped_bar_chart(data, title, xlabel="Score", ax=None, colors=None, labels=None,
                      percent=False, show_values=False, value_fmt=None, horizontal=False, value_lim=None,
                      reference=None, reference_label="Reference", show_families=True):
    """Grouped bar chart: one group per row of `data`, one bar per column.

    data: pd.DataFrame, index = categories (groups), columns = series (bars within a group).
    xlabel: label for the value axis (the x-axis when horizontal, the y-axis when standing).
    colors: optional dict {column: colour}; columns not in it take SERIES_COLORS in order.
    labels: optional dict {column: legend label}; defaults to the column name.
    percent: if True, values are fractions (0.31) shown as percentages (31%).
    show_values: print each bar's value at its tip (off by default; many bars get crowded).
    horizontal: True = bars grow right, groups top to bottom; False = standing columns, groups left to right.
    value_lim: optional (min, max) for the value axis, in the units shown.
               With percent=True it defaults to 0-100% (extended below 0 if any value is negative).
    reference: optional pd.DataFrame with the same index and columns as `data`; draws a short black
               marker on each bar at its reference value (e.g. the same metric on all posts).
    reference_label: legend name for the reference markers.
    show_families: if True, separate and label the model families, as in bar_chart.
    Returns the matplotlib Axes.
    """
    if percent:
        data = data * 100
    if reference is not None:
        reference = reference.reindex(index=data.index, columns=data.columns) * (100 if percent else 1)
    if value_fmt is None:
        value_fmt = "{:.0f}%" if percent else "{:.2f}"
    colors = colors or {}
    labels = labels or {}

    n_groups, n_series = data.shape
    if ax is None:
        if horizontal:
            figsize = (9, 0.28 * n_groups * n_series + 1.6)
        else:
            figsize = (0.28 * n_groups * n_series + 2, 6)
        _, ax = plt.subplots(figsize=figsize)

    group_width = 0.8
    bar_width = group_width / n_series
    for j, col in enumerate(data.columns):
        # first series at the top (horizontal) / left (standing) of each group
        pos = [i - group_width / 2 + bar_width * (j + 0.5) for i in range(n_groups)]
        vals = data[col].values
        color = colors.get(col, SERIES_COLORS[j % len(SERIES_COLORS)])
        label = labels.get(col, str(col))
        if horizontal:
            ax.barh(pos, vals, height=bar_width * 0.85, color=color, label=label, zorder=2)
        else:
            ax.bar(pos, vals, width=bar_width * 0.85, color=color, label=label, zorder=2)
        if reference is not None:
            half = bar_width * 0.85 / 2
            for p_, r in zip(pos, reference[col].values):
                if horizontal:
                    ax.vlines(r, p_ - half, p_ + half, color=TEXT_COLOR, linewidth=2, zorder=4)
                else:
                    ax.hlines(r, p_ - half, p_ + half, color=TEXT_COLOR, linewidth=2, zorder=4)
        if show_values:
            for p_, v in zip(pos, vals):
                if horizontal:
                    ax.text(v, p_, " " + value_fmt.format(v), va="center",
                            ha="left" if v >= 0 else "right", color=MUTED_TEXT, fontsize=8)
                else:
                    ax.text(p_, v, value_fmt.format(v), ha="center",
                            va="bottom" if v >= 0 else "top", color=MUTED_TEXT, fontsize=7)

    group_names = [str(i) for i in data.index]
    if horizontal:
        ax.set_yticks(range(n_groups), group_names, color=TEXT_COLOR)
        ax.set_ylim(n_groups - 0.5, -0.5)  # first group at the top, no extra space
        value_axis = ax.xaxis
        ax.axvline(0, color=MUTED_TEXT, linewidth=1, zorder=3)
        ax.grid(axis="x", color=GRID_COLOR, linewidth=1, zorder=0)
        ax.set_xlabel(xlabel, color=MUTED_TEXT)
    else:
        ax.set_xticks(range(n_groups), group_names, color=TEXT_COLOR, rotation=35, ha="right",
                      rotation_mode="anchor")
        ax.set_xlim(-0.5, n_groups - 0.5)
        value_axis = ax.yaxis
        ax.axhline(0, color=MUTED_TEXT, linewidth=1, zorder=3)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=1, zorder=0)
        ax.set_ylabel(xlabel, color=MUTED_TEXT)

    # recessive ticks, no box
    ax.tick_params(axis="both", length=0, colors=MUTED_TEXT)
    ax.tick_params(axis="y" if horizontal else "x", colors=TEXT_COLOR)
    for side in ax.spines.values():
        side.set_visible(False)

    if value_lim is None and percent:
        lowest = np.nanmin(data.values)
        if reference is not None:
            lowest = min(lowest, np.nanmin(reference.values))  # keep markers in view
        value_lim = _percent_lim(lowest)
    if value_lim is not None:
        ax.set_xlim(*value_lim) if horizontal else ax.set_ylim(*value_lim)

    if percent:
        value_axis.set_major_formatter(PercentFormatter(decimals=0))

    if show_families:
        add_family_labels(ax, data.index, horizontal)

    handles, _ = ax.get_legend_handles_labels()
    if reference is not None:
        handles.append(Line2D([], [], color=TEXT_COLOR, linewidth=2, label=reference_label))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
              frameon=False, fontsize=9, handlelength=1, columnspacing=1.2, labelcolor=TEXT_COLOR)
    ax.set_title(title, loc="center", color=TEXT_COLOR, fontsize=12, pad=30)
    return ax


# proprietary (closed) model families; every other family is open-weight
CLOSED_FAMILIES = {"gemini", "gpt"}


def model_sort_key(name):
    """
    Open-weight families first, then closed ones.
    Families in alphabetical within each, then release year in brackets (older model first).
    Order: Gemma, Llama, Qwen | Gemini, GPT
    """
    family = re.match(r"[a-z]+", name.lower()).group()
    year = int(name[name.rindex("(") + 1 : name.rindex(")")])
    return (family in CLOSED_FAMILIES, family, year, name)
