import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable


METHODS = [
    "scHiFi", "scSAMAC", "GCLSC", "scSCCNIA", "scAGCL", "nsDCC",
    "scDCCA", "scAGCN", "scGAC", "scziDesk", "scDeepCluster", "scGNN"
]

DATASET_ORDER = [
    "Klein", "Adam", "mouse_bladder_cell", "mouse_ES_cell", "Zeisel",
    "10XPBMC", "QX_Limb_Muscle", "Tosches_turtle", "Plasschaert", "Chen",
    "QX_Spleen", "Romanov", "Wang_Lung", "Bach"
]

METRICS = ["ACC", "NMI", "ARI", "AMI"]


def parse_score(x):
    """
    Convert various cell formats to decimal scale [0,1].
    Supports:
    - 0.878
    - 87.8
    - '87.8%'
    - '--'
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        s = x.strip()
        if s in {"", "--", "-", "nan", "NaN", "None"}:
            return np.nan
        if s.endswith("%"):
            try:
                return float(s[:-1]) / 100.0
            except ValueError:
                return np.nan
        try:
            val = float(s)
        except ValueError:
            return np.nan
    else:
        try:
            val = float(x)
        except Exception:
            return np.nan

    # If the number looks like percentage, convert to [0,1]
    if abs(val) > 1.5:
        val = val / 100.0

    return val


def read_excel_records(excel_path, methods):
    df = pd.read_excel(excel_path)

    records = []
    current_dataset = None

    for _, row in df.iterrows():
        if "数据集" in row.index and pd.notna(row.get("数据集")):
            current_dataset = str(row["数据集"]).strip()

        if current_dataset is None:
            continue

        if "指标" not in row.index or pd.isna(row.get("指标")):
            continue

        metric = str(row["指标"]).strip()

        if current_dataset == "Worm neuron cell":
            continue

        rec = {"dataset": current_dataset, "metric": metric}
        for m in methods:
            rec[m] = parse_score(row.get(m, np.nan))
        records.append(rec)

    return pd.DataFrame(records)


def build_matrices(rdf, methods, dataset_order, metrics, add_overall=False):
    mats = {}
    dataset_labels = dataset_order.copy()

    for metric in metrics:
        mat = np.full((len(methods), len(dataset_order)), np.nan)

        for i, method in enumerate(methods):
            for j, dataset in enumerate(dataset_order):
                sel = rdf[
                    (rdf["dataset"] == dataset) &
                    (rdf["metric"] == metric)
                ]
                if not sel.empty and method in sel.columns:
                    mat[i, j] = sel.iloc[0][method]

        if add_overall:
            overall = np.nanmean(mat, axis=1, keepdims=True)
            mat = np.hstack([mat, overall])

        mats[metric] = mat

    if add_overall:
        dataset_labels = dataset_labels + ["Overall"]

    return mats, dataset_labels


def normalize_matrix(mat, mode="column"):
    """
    Normalize color values.
    mode:
        - 'column': each dataset column independently normalized
        - 'global': the whole panel normalized together
    Returns normalized matrix in [0,1], NaN preserved.
    """
    norm_mat = np.full_like(mat, np.nan, dtype=float)

    if mode == "global":
        valid = ~np.isnan(mat)
        if np.any(valid):
            vmin = np.nanmin(mat)
            vmax = np.nanmax(mat)
            if np.isclose(vmin, vmax):
                norm_mat[valid] = 1.0
            else:
                norm_mat[valid] = (mat[valid] - vmin) / (vmax - vmin)
        return norm_mat

    elif mode == "column":
        for j in range(mat.shape[1]):
            col = mat[:, j]
            valid = ~np.isnan(col)
            if not np.any(valid):
                continue

            cmin = np.nanmin(col)
            cmax = np.nanmax(col)

            if np.isclose(cmin, cmax):
                norm_mat[valid, j] = 1.0
            else:
                norm_mat[valid, j] = (col[valid] - cmin) / (cmax - cmin)

        return norm_mat

    else:
        raise ValueError("mode must be 'column' or 'global'")


def luminance_from_rgba(rgba):
    r, g, b, _ = rgba
    return 0.299 * r + 0.587 * g + 0.114 * b


def format_value(val, value_format="decimal"):
    if np.isnan(val):
        return "--"
    if value_format == "decimal":
        return f"{val:.3f}"
    elif value_format == "percent":
        return f"{val * 100:.1f}"
    else:
        raise ValueError("value_format must be 'decimal' or 'percent'")


def get_metric_cmaps():
    """
    Four publication-style colormaps.
    """
    cmaps = {
        "ACC": LinearSegmentedColormap.from_list(
            "acc_blue",
            ["#EAF2FB", "#B8D3F0", "#6FA8DC", "#2F73B8", "#0B3C6D"]
        ),
        "NMI": LinearSegmentedColormap.from_list(
            "nmi_teal",
            ["#EAF7F5", "#B7E3DA", "#67C5B8", "#249D8F", "#0C5C55"]
        ),
        "ARI": LinearSegmentedColormap.from_list(
            "ari_purple",
            ["#F2EEFA", "#D3C4EC", "#A987D7", "#7C4FB3", "#4A257A"]
        ),
        "AMI": LinearSegmentedColormap.from_list(
            "ami_orange_red",
            ["#FEF1E8", "#F9C9A7", "#F28C52", "#D9543A", "#8F1D1D"]
        )
    }
    return cmaps


def draw_bubble_panel(
    ax,
    raw_mat,
    color_mat,
    row_labels,
    col_labels,
    metric,
    letter=None,
    cmap=None,
    value_format="decimal",
    circle_radius=0.35,
    value_fontsize=8.5,
    tick_fontsize=11,
    title_fontsize=18,
    italic_ylabels=True,
):
    n_rows, n_cols = raw_mat.shape
    norm = Normalize(vmin=0.0, vmax=1.0)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(
        col_labels,
        rotation=35,
        ha="right",
        rotation_mode="anchor",
        fontsize=tick_fontsize,
        fontweight="semibold"
    )

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(
        row_labels,
        fontsize=tick_fontsize,
        fontweight="semibold",
        fontstyle="italic" if italic_ylabels else "normal"
    )

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#C8C8C8", linestyle=":", linewidth=0.85)

    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n_rows):
        for j in range(n_cols):
            raw_val = raw_mat[i, j]
            color_val = color_mat[i, j]

            if np.isnan(raw_val):
                circle = Circle(
                    (j, i),
                    radius=circle_radius,
                    facecolor="#F2F2F2",
                    edgecolor="#BDBDBD",
                    linewidth=0.8
                )
                ax.add_patch(circle)
                ax.text(
                    j, i, "--",
                    ha="center", va="center",
                    fontsize=value_fontsize,
                    fontweight="semibold",
                    color="#7A7A7A"
                )
                continue

            rgba = cmap(norm(color_val))
            circle = Circle(
                (j, i),
                radius=circle_radius,
                facecolor=rgba,
                edgecolor="black",
                linewidth=0.9
            )
            ax.add_patch(circle)

            lum = luminance_from_rgba(rgba)
            txt_color = "white" if lum < 0.55 else "black"

            ax.text(
                j, i, format_value(raw_val, value_format),
                ha="center", va="center",
                fontsize=value_fontsize,
                fontweight="bold",
                color=txt_color
            )

    title_text = f"{letter} {metric}" if letter is not None else metric
    ax.set_title(
        title_text,
        loc="left",
        pad=10,
        fontsize=title_fontsize,
        fontweight="bold",
        fontfamily="serif"
    )


def add_colorbar(fig, ax, cmap, label="Relative performance\n(within each column)"):
    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.015)
    cb.ax.tick_params(labelsize=8, length=2)
    cb.outline.set_linewidth(0.6)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["Low", "Mid", "High"])
    cb.set_label(label, fontsize=9)


def plot_combined_figure(
    mats,
    dataset_labels,
    methods,
    out_prefix,
    dpi=600,
    value_format="decimal",
    norm_mode="column",
    fig_width=24,
    fig_height=20,
    circle_radius=0.35,
    value_fontsize=8.5,
    tick_fontsize=11,
    title_fontsize=18,
    show_colorbar=False,
):
    cmaps = get_metric_cmaps()

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=dpi)
    axes = axes.ravel()
    letters = ["(a)", "(b)", "(c)", "(d)"]

    for ax, metric, letter in zip(axes, METRICS, letters):
        raw_mat = mats[metric]
        color_mat = normalize_matrix(raw_mat, mode=norm_mode)

        draw_bubble_panel(
            ax=ax,
            raw_mat=raw_mat,
            color_mat=color_mat,
            row_labels=methods,
            col_labels=dataset_labels,
            metric=metric,
            letter=letter,
            cmap=cmaps[metric],
            value_format=value_format,
            circle_radius=circle_radius,
            value_fontsize=value_fontsize,
            tick_fontsize=tick_fontsize,
            title_fontsize=title_fontsize,
        )

        if show_colorbar:
            add_colorbar(fig, ax, cmaps[metric])

    fig.subplots_adjust(
        left=0.08, right=0.985, top=0.95, bottom=0.10,
        wspace=0.16, hspace=0.22
    )

    out_prefix = Path(out_prefix)
    fig.savefig(out_prefix.with_name(out_prefix.name + "_combined").with_suffix(".png"),
                dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_prefix.with_name(out_prefix.name + "_combined").with_suffix(".pdf"),
                dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_single_figures(
    mats,
    dataset_labels,
    methods,
    out_prefix,
    dpi=600,
    value_format="decimal",
    norm_mode="column",
    single_width=13,
    single_height=9,
    circle_radius=0.35,
    value_fontsize=9.0,
    tick_fontsize=12,
    title_fontsize=18,
    show_colorbar=False,
):
    cmaps = get_metric_cmaps()
    letters = ["(a)", "(b)", "(c)", "(d)"]
    out_prefix = Path(out_prefix)

    for metric, letter in zip(METRICS, letters):
        fig, ax = plt.subplots(1, 1, figsize=(single_width, single_height), dpi=dpi)

        raw_mat = mats[metric]
        color_mat = normalize_matrix(raw_mat, mode=norm_mode)

        draw_bubble_panel(
            ax=ax,
            raw_mat=raw_mat,
            color_mat=color_mat,
            row_labels=methods,
            col_labels=dataset_labels,
            metric=metric,
            letter=letter,
            cmap=cmaps[metric],
            value_format=value_format,
            circle_radius=circle_radius,
            value_fontsize=value_fontsize,
            tick_fontsize=tick_fontsize,
            title_fontsize=title_fontsize,
        )

        if show_colorbar:
            add_colorbar(fig, ax, cmaps[metric])

        fig.subplots_adjust(left=0.10, right=0.96, top=0.91, bottom=0.19)

        fig.savefig(out_prefix.with_name(f"{out_prefix.name}_{metric}").with_suffix(".png"),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
        fig.savefig(out_prefix.with_name(f"{out_prefix.name}_{metric}").with_suffix(".pdf"),
                    dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--excel", default="scHiFi对比实验.xlsx")
    parser.add_argument("--out-prefix", default="clustering_bubble_heatmap")

    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--value-format", choices=["decimal", "percent"], default="decimal")
    parser.add_argument("--add-overall", action="store_true")
    parser.add_argument("--reverse-methods", action="store_true")

    # new features
    parser.add_argument("--norm-mode", choices=["column", "global"], default="column",
                        help="column: normalize each dataset column independently; global: normalize whole panel")
    parser.add_argument("--plot-mode", choices=["combined", "single", "both"], default="both",
                        help="combined: 2x2 panel; single: four separate figures; both: output both")
    parser.add_argument("--show-colorbar", action="store_true")

    # layout
    parser.add_argument("--circle-radius", type=float, default=0.35)
    parser.add_argument("--value-fontsize", type=float, default=8.5)
    parser.add_argument("--tick-fontsize", type=float, default=11)
    parser.add_argument("--title-fontsize", type=float, default=18)

    parser.add_argument("--combined-width", type=float, default=24)
    parser.add_argument("--combined-height", type=float, default=20)
    parser.add_argument("--single-width", type=float, default=13)
    parser.add_argument("--single-height", type=float, default=9)

    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white"
    })

    methods = METHODS.copy()
    if args.reverse_methods:
        methods = methods[::-1]

    rdf = read_excel_records(args.excel, methods)
    mats, dataset_labels = build_matrices(
        rdf=rdf,
        methods=methods,
        dataset_order=DATASET_ORDER,
        metrics=METRICS,
        add_overall=args.add_overall
    )

    if args.plot_mode in {"combined", "both"}:
        plot_combined_figure(
            mats=mats,
            dataset_labels=dataset_labels,
            methods=methods,
            out_prefix=args.out_prefix,
            dpi=args.dpi,
            value_format=args.value_format,
            norm_mode=args.norm_mode,
            fig_width=args.combined_width,
            fig_height=args.combined_height,
            circle_radius=args.circle_radius,
            value_fontsize=args.value_fontsize,
            tick_fontsize=args.tick_fontsize,
            title_fontsize=args.title_fontsize,
            show_colorbar=args.show_colorbar,
        )

    if args.plot_mode in {"single", "both"}:
        plot_single_figures(
            mats=mats,
            dataset_labels=dataset_labels,
            methods=methods,
            out_prefix=args.out_prefix,
            dpi=args.dpi,
            value_format=args.value_format,
            norm_mode=args.norm_mode,
            single_width=args.single_width,
            single_height=args.single_height,
            circle_radius=args.circle_radius,
            value_fontsize=max(args.value_fontsize, 9.0),
            tick_fontsize=max(args.tick_fontsize, 12),
            title_fontsize=args.title_fontsize,
            show_colorbar=args.show_colorbar,
        )


if __name__ == "__main__":
    main()