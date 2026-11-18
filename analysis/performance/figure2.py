#!/usr/bin/env python3
"""Reflow the existing four-metric bubble plot; preserve every value and color."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import compare1 as style

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'input'


def save_pub_set(fig, out_dir, stem, formats=('png', 'pdf', 'svg'), dpi=600,
                 bbox_inches='tight', close=True):
    """Export the composite figure in PNG, PDF and SVG formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = out_dir / f'{stem}.{fmt}'
        fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')
        saved.append(path)
    if close:
        plt.close(fig)
    return saved


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    (ROOT/'layout').mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white'})
    records = pd.read_csv(SOURCE / 'values.csv', float_precision='round_trip')
    metadata = json.loads((SOURCE / 'layout_metadata.json').read_text())
    methods, datasets = metadata['methods'], metadata['datasets']
    matrices = {}
    for metric in style.METRICS:
        frame = pd.read_csv(SOURCE / f'{metric}_mean.csv', index_col=0,
                            float_precision='round_trip')
        assert list(frame.index) == methods and list(frame.columns) == datasets
        mat = frame.to_numpy()
        check = records[records.metric == metric]
        np.testing.assert_array_equal(mat.ravel(), check['mean'].to_numpy())
        np.testing.assert_allclose(style.normalize_matrix(mat).ravel(),
                                   check.relative_performance, atol=1e-15, rtol=0)
        matrices[metric] = mat

    # Preserve the original panel scale; pack each panel's complete text extent.
    panel_width = 8.3
    panel_height = panel_width * len(methods) / len(datasets)
    fig = plt.figure(figsize=(24, 24), dpi=100)
    axes = [fig.add_axes([.2, .2, panel_width / 24, panel_height / 24]) for _ in range(4)]
    cmaps = style.get_metric_cmaps()
    for ax, metric, letter in zip(axes, style.METRICS, ['(a)', '(b)', '(c)', '(d)']):
        mat = matrices[metric]
        style.draw_bubble_panel(ax, mat, style.normalize_matrix(mat), methods,
                                datasets, metric, letter=letter, cmap=cmaps[metric],
                                value_format='percent', circle_radius=.35,
                                value_fontsize=8.5, tick_fontsize=11, title_fontsize=18)
        # Place the title above the method labels, below its previous baseline.
        # Explicit y disables automatic title placement above the plotting area.
        ax.set_title(f'{letter} {metric}', loc='left', x=-.015, y=.98, pad=0,
                     ha='right', va='bottom', fontsize=18,
                     fontweight='bold', fontfamily='serif')
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    margins = []
    for ax in axes:
        body = ax.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
        full = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        margins.append([body.x0-full.x0, full.x1-body.x1,
                        body.y0-full.y0, full.y1-body.y1])
    left, right, bottom, top = np.max(margins, axis=0)
    tile_w = left + panel_width + right
    tile_h = bottom + panel_height + top
    gutter, row_gap, outer = .18, .14, .08
    width = 2*tile_w + gutter + 2*outer
    height = 2*tile_h + row_gap + 2*outer
    fig.set_size_inches(width, height)
    for i, ax in enumerate(axes):
        col, row = i % 2, 1-i//2
        x = outer + col*(tile_w+gutter) + left
        y = outer + row*(tile_h+row_gap) + bottom
        ax.set_position([x/width, y/height, panel_width/width, panel_height/height])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = [ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()) for ax in axes]
    gaps = [bounds[1].x0-bounds[0].x1, bounds[3].x0-bounds[2].x1]
    assert min(gaps) >= .17, gaps
    title_checks = []
    for ax in axes:
        title_box = ax._left_title.get_window_extent(renderer)
        label_box = ax.get_yticklabels()[0].get_window_extent(renderer)
        assert title_box.y0 > label_box.y1
        assert title_box.x1 < ax.get_window_extent(renderer).x0
        title_checks.append({'title': ax._left_title.get_text(),
                             'clear_of_method_labels_and_data': True})
    assert all(len(ax.patches) == 156 for ax in axes)
    displayed = [text.get_text() for ax in axes for text in ax.texts]
    expected = [style.format_value(v, 'percent') for m in style.METRICS
                for v in matrices[m].ravel()]
    assert displayed == expected and len(displayed) == 624
    # Modest preview is made by the same Python figure backend.
    preview = ROOT / 'layout' / 'preview.png'
    fig.savefig(preview, dpi=100, bbox_inches='tight', facecolor='white')
    outputs = save_pub_set(fig, ROOT/'figures', 'clustering_bubble_heatmap_combined')
    svg = ET.parse(next(p for p in outputs if p.suffix == '.svg')).getroot()
    images = [x for x in svg.iter() if x.tag.endswith('}image')]
    texts = [x for x in svg.iter() if x.tag.endswith('}text')]
    assert not images and len(texts) >= 624
    audit = dict(date='2026-11-18', change='Move titles above method labels and lower them; tighten vertical spacing',
                 title_position_axes=[-.015, .98], title_checks=title_checks,
                 canvas_inches=[width, height], full_panel_gap_inches=gaps,
                 panel_width_inches=panel_width, panel_height_inches=panel_height,
                 values_checked=624, values_and_colors_unchanged=True,
                 svg_images=len(images), svg_editable_text_elements=len(texts), dpi=600,
                 inputs={p.name: sha(p) for p in [SOURCE/'values.csv',
                        *[SOURCE/f'{m}_mean.csv' for m in style.METRICS]]},
                 outputs={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in outputs})
    (ROOT/'layout'/'layout.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
