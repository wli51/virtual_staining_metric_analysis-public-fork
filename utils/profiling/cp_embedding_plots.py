"""Descriptive reports for the on-disk :mod:`cp_embedding` analysis contract.

Call ``render_cp_reports(output)`` after preparation, or again after embedding.
Only ``output / 'reports'`` is written; input tables, arrays and notebooks are
never modified. No fitting, array imputation, or plotting occurs on import.
"""

from __future__ import annotations

import hashlib
import json
import re
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

__all__ = ["render_cp_reports"]

STRATA = ["cell_line", "plate", "seeding_density"]
CONDITION = [*STRATA, "well"]
METRICS = {"l2": "Standardized L2", "pca_l2": "PCA L2", "cosine_distance": "Cosine distance"}
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
    "axes.labelcolor": "#334155", "text.color": "#172b4d",
    "axes.edgecolor": "#cbd5e1", "axes.spines.top": False, "axes.spines.right": False,
    "axes.facecolor": "#fafbfd", "figure.facecolor": "white", "grid.alpha": 0.22,
    "pdf.fonttype": 42, "savefig.dpi": 220,
}


def _json(path):
    return json.loads(path.read_text())


def _csv(frame, folder, name):
    frame.to_csv(folder / f"{name}.csv", index=False)
    return frame


def _save(fig, folder, name, *, close=True):
    for extension in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{extension}", bbox_inches="tight")
    if close:
        plt.close(fig)


def _rate(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def _rates(frame):
    frame = frame.copy()
    frame["missing_rate"] = _rate(frame.n_missing, frame.n_values)
    frame["inf_rate"] = _rate(frame.n_inf, frame.n_values)
    frame["nonfinite_rate"] = _rate(frame.n_missing + frame.n_inf, frame.n_values)
    return frame


def _markdown(frame):
    """Small dependency-free markdown tables (no optional tabulate dependency)."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}" if np.isfinite(value) else "NA"
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [list(frame.columns), ["---"] * len(frame.columns), *frame.itertuples(index=False)]
    return "\n".join("| " + " | ".join(map(cell, row)) + " |" for row in rows)


def _heatmap(ax, grid, title, *, label, vmax=None, cmap="magma", percent=False):
    colors = mpl.colormaps[cmap].copy()
    colors.set_bad("#e2e8f0")
    column_labels = [f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
                     for v in grid.columns]
    image = ax.imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)), aspect="auto",
                      interpolation="nearest", cmap=colors, vmin=0, vmax=vmax)
    ax.set(xticks=np.arange(len(grid.columns)), xticklabels=column_labels,
           yticks=np.arange(len(grid)), yticklabels=list(grid.index), title=title)
    ax.tick_params(axis="x", labelrotation=55, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    bar = ax.figure.colorbar(image, ax=ax, shrink=0.8, pad=0.02, label=label)
    if percent:
        bar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    return image


def _source_metadata(frame, sources):
    lookup = sources[["source_id", "sigma_px", "blur"]]
    return frame.merge(lookup, on="source_id", how="left", validate="many_to_one")


def _audit_reports(output, reports, sources):
    """Full-data denominators come from audits, never from embedded rows."""
    features = _rates(pd.read_csv(output / "feature_missingness.csv"))
    structural = pd.read_csv(output / "structural_diagnostics.csv")
    warnings = pd.read_csv(output / "warning_summary.csv")
    conditions = _rates(pd.read_csv(output / "row_missingness_by_condition.csv"))
    coverage = pd.read_csv(output / "sampling_coverage.csv")
    for name, frame in (("feature_missingness", features), ("structural_diagnostics", structural),
                        ("warning_summary", warnings), ("row_missingness_by_condition", conditions),
                        ("sampling_coverage", coverage)):
        _csv(frame, reports, name)

    totals = _rates(features.groupby(["compartment", "source_id", "scope"], observed=True)[
        ["n_values", "n_missing", "n_inf"]
    ].sum().reset_index())
    totals = _source_metadata(totals, sources).sort_values(["compartment", "sigma_px", "scope"])
    _csv(totals, reports, "full_data_missingness_by_source_scope")
    grouped = features.groupby(
        ["compartment", "source_id", "scope", "context", "family"], observed=True
    )[["n_values", "n_missing", "n_inf"]].sum().reset_index()
    grouped = _source_metadata(_rates(grouped), sources)
    _csv(grouped, reports, "family_context_missingness")
    grouped["feature_group"] = grouped[["scope", "context", "family"]].agg(" / ".join, axis=1)
    order = [f"{comp} · σ={sigma:g}" for comp in sorted(features.compartment.unique())
             for sigma in sources.sigma_px]
    grouped["source_compartment"] = [
        f"{comp} · σ={sigma:g}" for comp, sigma in zip(grouped.compartment, grouped.sigma_px)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, grouped.feature_group.nunique() * 0.25)),
                             layout="constrained")
    for ax, metric, title in zip(axes, ["missing_rate", "inf_rate"], ["Applicable NaN", "Infinity"]):
        grid = grouped.pivot(index="feature_group", columns="source_compartment", values=metric)
        _heatmap(ax, grid.reindex(columns=order), title, label="Fraction of applicable values",
                 vmax=1, percent=True)
    fig.suptitle("Full-data QC · family and channel context (geometry is audit-only)")
    _save(fig, reports, "family_context_missingness")

    worst = features.sort_values(["nonfinite_rate", "n_values"], ascending=False)
    _csv(worst, reports, "worst_features")
    ranked = _rates(features.groupby(["scope", "context", "family", "name"], observed=True)[
        ["n_values", "n_missing", "n_inf"]
    ].sum().reset_index()).sort_values("nonfinite_rate", ascending=False)
    _csv(ranked, reports, "worst_features_pooled")
    for label, subset in (("all", ranked), ("embedding_scopes", ranked.loc[ranked.scope.ne("geometry")])):
        top = subset.head(25).iloc[::-1]
        if top.empty:
            continue
        fig, ax = plt.subplots(figsize=(13, 9), layout="constrained")
        ax.barh(top.name, top.missing_rate, color="#7c3aed", label="Applicable NaN")
        ax.barh(top.name, top.inf_rate, left=top.missing_rate, color="#f97316", label="Infinity")
        ax.set(xlabel="Fraction of applicable values · pooled sources and compartments", xlim=(0, 1),
               title=f"Worst 25 features · {label.replace('_', ' ')}")
        ax.tick_params(axis="y", labelsize=8)
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
        ax.legend(loc="lower right")
        _save(fig, reports, f"worst_features_{label}")

    warning_keys = ["message", "filename", "category"]
    warning_groups = warnings.groupby(warning_keys, dropna=False).agg(
        count=("count", "sum"), n_records=("n_records", "sum"),
        n_attempts=("attempt_id", "nunique"), n_sources=("source_id", "nunique"),
        n_fovs=("fov_id", "nunique"),
    ).reset_index().sort_values("count", ascending=False)
    _csv(warning_groups, reports, "warnings_grouped_message_file_category")
    by_source = warnings.groupby(["source_id", *warning_keys], dropna=False)[
        ["count", "n_records"]
    ].sum().reset_index()
    _csv(_source_metadata(by_source, sources), reports, "warnings_by_source")
    fig, ax = plt.subplots(figsize=(14, 8), layout="constrained")
    if warning_groups.empty:
        ax.text(0.5, 0.5, "No warning emissions recorded", ha="center", transform=ax.transAxes)
    else:
        top = warning_groups.head(15).iloc[::-1]
        labels = [textwrap.fill(f"{row.category} · {Path(str(row.filename)).name}\n{row.message}", 78)
                  for row in top.itertuples()]
        positions = np.arange(len(top))
        ax.barh(positions, top["count"], color="#64748b")
        ax.set(yticks=positions, yticklabels=labels)
        ax.tick_params(axis="y", labelsize=8)
    ax.set(title="Warnings · all logged attempts, including failed/retried attempts",
           xlabel="Warning emissions (sum of count), NOT affected objects")
    _save(fig, reports, "warning_emissions")

    # Scope counts represent object-context rows; do not sum them into object counts.
    source_condition = conditions.groupby(
        ["compartment", "source_id", "cell_line"], observed=True
    )[["n_values", "n_missing", "n_inf"]].sum().reset_index()
    source_condition = _source_metadata(_rates(source_condition), sources)
    _csv(source_condition, reports, "raw_missingness_by_line_dose")
    compartments = sorted(source_condition.compartment.unique())
    fig, axes = plt.subplots(len(compartments), 2, figsize=(16, 6 * len(compartments)),
                             squeeze=False, layout="constrained")
    for row_axes, compartment in zip(axes, compartments):
        subset = source_condition.loc[source_condition.compartment.eq(compartment)]
        for ax, metric in zip(row_axes, ["missing_rate", "inf_rate"]):
            grid = subset.pivot(index="cell_line", columns="sigma_px", values=metric)
            _heatmap(ax, grid, f"{compartment} · {metric.replace('_', ' ')}",
                     label="Raw applicable value fraction (panel-scaled)",
                     vmax=max(float(grid.max().max()), 1e-6), percent=True)
            ax.set_xlabel("Gaussian blur σ (px); 0 = reference")
    fig.suptitle("Full-data source/dose-aware missingness · before feature filtering or imputation")
    _save(fig, reports, "raw_missingness_by_source_dose")

    structural_totals = structural.groupby(["compartment", "source_id"], observed=True)[
        ["n_structural", "n_not_nan"]
    ].sum().reset_index()
    _csv(_source_metadata(structural_totals, sources), reports, "structural_totals")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for comp, group in totals.groupby("compartment", observed=True):
        summary = _rates(group.groupby("sigma_px")[["n_values", "n_missing", "n_inf"]].sum())
        axes[0].plot(summary.index, summary.nonfinite_rate, "o-", label=comp)
    axes[0].set(xlabel="Gaussian blur σ (px)", ylabel="Applicable nonfinite fraction",
                title="NaN + infinity; structural padding excluded")
    axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
    axes[0].legend()
    scope_summary = structural.groupby("compartment")[["n_structural", "n_not_nan"]].sum()
    scope_summary.plot.bar(ax=axes[1], color=["#94a3b8", "#ef4444"], rot=0)
    axes[1].set(title="Structural padding is not applicable missingness", ylabel="Value slots")
    _save(fig, reports, "structural_vs_applicable")
    return totals, structural_totals, warning_groups, coverage


def _palettes(raw_rows):
    """Use exactly the category ordering and colormap sampling of notebooks 2.1/2.2."""
    combined = pd.concat(raw_rows, ignore_index=True)
    palettes = {}
    for key in STRATA:
        values = sorted(combined[key].dropna().unique())
        cmap = mpl.colormaps["viridis" if key == "seeding_density" else "tab20"]
        start, stop = (0, 1) if key == "cell_line" else (0.1, 0.9)
        colors = cmap(np.linspace(start, stop, len(values)))
        palettes[key] = {str(value): color.tolist() for value, color in zip(values, colors)}
    return palettes


def _cleanup_reports(folder, reports, rows, config):
    decisions = pd.read_csv(folder / "feature_decisions.csv")
    qc = pd.read_parquet(folder / "row_missingness.parquet")
    _csv(decisions, reports, "feature_decisions")
    _csv(qc, reports, "row_missingness")
    _csv(pd.read_csv(folder / "pair_exclusions.csv"), reports, "pair_exclusions")
    (reports / "config.json").write_text(json.dumps(config, indent=2))
    feature_summary = decisions.groupby(["scope", "family", "reason"], observed=True).agg(
        n_features=("name", "size"), n_retained=("retained", "sum")
    ).reset_index()
    _csv(feature_summary, reports, "feature_cleanup_summary")
    retention = qc.groupby([*CONDITION, "source_id", "sigma_px"], observed=True).agg(
        n_profiles_before=("row_index", "size"), n_direct_failed=("fails_object_qc", "sum"),
        n_pair_excluded=("pair_excluded", "sum"), n_missing=("n_missing", "sum"),
        n_inf=("n_inf", "sum"), n_values=("n_features", "sum"),
    ).reset_index()
    retention["n_kept"] = retention.n_profiles_before - retention.n_pair_excluded
    retention["n_excluded_due_to_other_source"] = retention.n_pair_excluded - retention.n_direct_failed
    retention["retention_rate"] = _rate(retention.n_kept, retention.n_profiles_before)
    retention["pair_exclusion_rate"] = _rate(retention.n_pair_excluded, retention.n_profiles_before)
    _csv(_rates(retention), reports, "condition_retention")
    paired = qc.loc[qc.sigma_px.eq(0)].groupby(CONDITION, observed=True).agg(
        n_pairs_before=("reference_row", "size"), n_pairs_excluded=("pair_excluded", "sum")
    ).reset_index()
    paired["n_pairs_kept"] = paired.n_pairs_before - paired.n_pairs_excluded
    paired["retention_rate"] = _rate(paired.n_pairs_kept, paired.n_pairs_before)
    _csv(paired, reports, "condition_paired_exclusions")
    imputation = rows.groupby([*CONDITION, "source_id", "sigma_px"], observed=True).agg(
        n_profiles=("row_index", "size"), n_imputed=("n_imputed", "sum"),
        n_profiles_imputed=("n_imputed", lambda x: int(x.gt(0).sum())),
        n_abs_scaled_gt10=("n_abs_scaled_gt10", "sum"),
    ).reset_index()
    imputation["n_values"] = imputation.n_profiles * config["n_features"]
    imputation["imputed_fraction"] = _rate(imputation.n_imputed, imputation.n_values)
    imputation["profiles_imputed_fraction"] = _rate(imputation.n_profiles_imputed, imputation.n_profiles)
    _csv(imputation, reports, "imputation_by_condition_dose")
    pooled = imputation.groupby(["cell_line", "sigma_px"], observed=True)[
        ["n_profiles", "n_imputed", "n_values", "n_profiles_imputed"]
    ].sum().reset_index()
    pooled["imputed_fraction"] = _rate(pooled.n_imputed, pooled.n_values)
    retained = retention.groupby(["cell_line", "sigma_px"], observed=True)[
        ["n_profiles_before", "n_kept"]
    ].sum().reset_index()
    retained["retention_rate"] = _rate(retained.n_kept, retained.n_profiles_before)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
    for ax, frame, value, title in (
        (axes[0], pooled, "imputed_fraction", "Imputed values / final retained feature values"),
        (axes[1], retained, "retention_rate", "Retained pairs / sampled pairs before QC"),
    ):
        grid = frame.pivot(index="cell_line", columns="sigma_px", values=value)
        maximum = max(float(grid.max().max()), 1e-6) if value == "imputed_fraction" else 1
        label = "Fraction (panel-scaled)" if value == "imputed_fraction" else "Fraction"
        _heatmap(ax, grid, title, label=label, vmax=maximum, percent=True)
        ax.set_xlabel("Gaussian blur σ (px)")
    fig.suptitle(f"{config['compartment']} · imputation and paired selection are part of the result")
    _save(fig, reports, "imputation_and_retention_by_dose")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    reasons = decisions.reason.value_counts().sort_values()
    ax.barh(reasons.index, reasons.values, color="#6366f1")
    ax.set(title="Reference-fitted feature cleanup · geometry is audit-only", xlabel="Expanded features")
    _save(fig, reports, "feature_cleanup")
    return paired, imputation, decisions


def _blur_palette(sources):
    """Discrete reference/low/mid/high colors shared with MorphEm notebooks."""
    ordered = sources.sort_values(["sigma_px", "source_id"])
    colors = ["#34495e", "#e3b354", "#df7852", "#a53860"]
    if len(ordered) != 4:
        colors = mpl.colormaps["magma"](np.linspace(0.15, 0.8, len(ordered))).tolist()
    return dict(zip(ordered.blur, colors))


def _blur_handles(sources):
    palette = _blur_palette(sources)
    return [Line2D([], [], marker="o", linestyle="", color=palette[source.blur],
                   label=f"{source.blur} · σ={source.sigma_px:.3g} px", markersize=5)
            for source in sources.sort_values(["sigma_px", "source_id"]).itertuples()]


def _maps(rows, reports, palettes, sources, compartment):
    # Do not let the source-major matrix order paint highest blur over references.
    rows = rows.sample(frac=1, random_state=42)
    fig, axes = plt.subplots(2, 2, figsize=(17, 15), layout="constrained")
    for ax, key in zip(axes.flat, [*STRATA, "blur"]):
        if key == "blur":
            palette = _blur_palette(sources)
            ax.scatter(rows.UMAP1, rows.UMAP2, c=[palette[v] for v in rows.blur],
                       s=2, alpha=0.45, linewidths=0, rasterized=True)
            ax.legend(handles=_blur_handles(sources), loc="upper center",
                      bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8,
                      frameon=False, title="Blur condition")
        else:
            palette = palettes[key]
            colors = [palette[v] for v in rows[key].astype(str)]
            ax.scatter(rows.UMAP1, rows.UMAP2, c=colors, s=2, alpha=0.4,
                       linewidths=0, rasterized=True)
            handles = [Line2D([], [], marker="o", linestyle="", color=color, label=value,
                              markersize=5) for value, color in palette.items()]
            ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                      ncol=min(4, len(handles)), fontsize=8, frameon=False, title=key)
        ax.set(title=key.replace("_", " "), xlabel="UMAP1", ylabel="UMAP2")
    fig.suptitle(f"{compartment} · one joint map, all retained sampled profiles\n"
                 "Category colors shared across compartments; coordinates are not")
    _save(fig, reports, "embedding_covariates_2x2")


def _select_tracks(rows, seed, compartment):
    references = rows.loc[rows.sigma_px.eq(0)]
    selected = []
    for values, group in references.groupby(STRATA, observed=True, sort=True):
        token = json.dumps([int(seed), compartment, list(values)], default=str)
        rng = np.random.default_rng(int(hashlib.sha256(token.encode()).hexdigest()[:16], 16))
        group = group.sort_values(["fov_id", "object_id", "row_index"])
        selected.extend(group.iloc[rng.choice(len(group), min(2, len(group)), replace=False)].row_index)
    tracks = rows.loc[rows.reference_row.isin(selected)].copy()
    return tracks.sort_values(["reference_row", "sigma_px", "source_id"])


def _limits(frame):
    points = frame[["UMAP1", "UMAP2"]].to_numpy()
    low, high = points.min(axis=0), points.max(axis=0)
    margin = np.maximum(high - low, 1) * 0.06
    return (low[0] - margin[0], high[0] + margin[0]), (low[1] - margin[1], high[1] + margin[1])


def _track_figure(background, tracks, title, sources, full_limits):
    palette = _blur_palette(sources)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    for ax, blurred_only in zip(axes, [False, True]):
        bg = background.loc[background.sigma_px.gt(0)] if blurred_only else background
        visible = tracks.loc[tracks.sigma_px.gt(0)] if blurred_only else tracks
        ax.scatter(bg.UMAP1, bg.UMAP2, color="#94a3b8", s=2, alpha=0.07,
                   linewidths=0, rasterized=True)
        segments = []
        for _, group in visible.groupby("reference_row", sort=False):
            points = group.sort_values(["sigma_px", "source_id"])[["UMAP1", "UMAP2"]].to_numpy()
            segments.extend(np.stack([points[:-1], points[1:]], axis=1))
        if segments:
            ax.add_collection(LineCollection(segments, colors="#334155", linewidths=0.6,
                                             alpha=0.45, rasterized=True))
        ax.scatter(visible.UMAP1, visible.UMAP2, c=[palette[v] for v in visible.blur],
                   s=12, edgecolors="white", linewidths=0.2, rasterized=True, zorder=3)
        limits = _limits(bg) if blurred_only and len(bg) else full_limits
        ax.set(xlim=limits[0], ylim=limits[1], xlabel="UMAP1", ylabel="UMAP2",
               title="Blurred-only zoom · same coordinates" if blurred_only else "Full map extent")
    fig.legend(handles=_blur_handles(sources), loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"{title}\nTwo seeded reference objects per observed biological stratum; successive doses")
    return fig


def _successive_steps(rows, folder, reports, sources):
    """All retained pairs, not just illustrative tracks; bounded array reads."""
    x = np.load(folder / "joint_profiles.npy", mmap_mode="r")
    pca = np.load(folder / "pca_scores.npy", mmap_mode="r")
    if x.ndim != 2 or pca.ndim != 2 or len(x) != len(rows) or len(pca) != len(rows):
        raise ValueError(f"Array/embedding row mismatch: {folder}")
    ordered = rows.sort_values(["reference_row", "sigma_px", "source_id"])
    previous = ordered.groupby("reference_row", sort=False).row_index.shift()
    endpoints = ordered.loc[previous.notna()].copy()
    left = previous.loc[previous.notna()].to_numpy(dtype=np.int64)
    right = endpoints.row_index.to_numpy(dtype=np.int64)
    endpoints = endpoints.rename(columns={"row_index": "to_row_index", "sigma_px": "to_sigma_px",
                                          "source_id": "to_source_id", "blur": "to_blur",
                                          **{key: f"reference_{key}" for key in METRICS}})
    endpoints["from_row_index"] = left
    lookup = rows.set_index("row_index")
    for key in ["source_id", "sigma_px", "raw_row_index", "n_imputed", "UMAP1", "UMAP2"]:
        endpoints[f"from_{key}"] = lookup.loc[left, key].to_numpy()
    endpoints["delta_sigma_px"] = endpoints.to_sigma_px - endpoints.from_sigma_px
    endpoints["to_UMAP1"], endpoints["to_UMAP2"] = endpoints.UMAP1, endpoints.UMAP2
    endpoints["delta_UMAP1"] = endpoints.UMAP1 - endpoints.from_UMAP1
    endpoints["delta_UMAP2"] = endpoints.UMAP2 - endpoints.from_UMAP2
    endpoints["umap_l2"] = np.hypot(endpoints.delta_UMAP1, endpoints.delta_UMAP2)
    for name in ["standardized_l2", "standardized_cosine_distance", "pca_step_l2"]:
        endpoints[name] = np.nan
    # Reference-distance columns in the embedding are deliberately not reused as steps.
    for start in range(0, len(endpoints), 4096):
        stop = min(start + 4096, len(endpoints))
        a = np.asarray(x[left[start:stop]], dtype=np.float64)
        b = np.asarray(x[right[start:stop]], dtype=np.float64)
        denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        cosine = np.divide(np.einsum("ij,ij->i", a, b), denominator,
                           out=np.full(len(a), np.nan), where=denominator > 0)
        ids = endpoints.index[start:stop]
        endpoints.loc[ids, "standardized_l2"] = np.linalg.norm(b - a, axis=1)
        endpoints.loc[ids, "standardized_cosine_distance"] = 1 - np.clip(cosine, -1, 1)
        endpoints.loc[ids, "pca_step_l2"] = np.linalg.norm(
            np.asarray(pca[right[start:stop]], dtype=np.float64)
            - np.asarray(pca[left[start:stop]], dtype=np.float64), axis=1)
    expected = rows.reference_row.nunique() * (len(sources) - 1)
    if len(endpoints) != expected:
        raise ValueError("Successive steps do not cover every retained object/source transition")
    endpoints.to_parquet(reports / "all_successive_steps.parquet", index=False)
    summary = endpoints.groupby(["from_source_id", "to_source_id", "from_sigma_px", "to_sigma_px"],
                                observed=True)[
        ["standardized_l2", "standardized_cosine_distance", "pca_step_l2", "umap_l2"]
    ].agg(["count", "median", "mean"]).reset_index()
    summary.columns = ["_".join(filter(None, col)) for col in summary.columns]
    _csv(summary, reports, "successive_step_summary")
    return len(endpoints)


def _trajectory_reports(rows, folder, reports, sources, seed, compartment):
    tracks = _select_tracks(rows, seed, compartment)
    tracks.to_parquet(reports / "illustrative_trajectories.parquet", index=False)
    n_steps = _successive_steps(rows, folder, reports, sources)
    limits = _limits(rows)
    fig = _track_figure(rows, tracks, f"{compartment} · global matched-object trajectories", sources, limits)
    _save(fig, reports, "trajectories_global")
    atlas_dir = reports / "cell_line_maps"
    atlas_dir.mkdir(exist_ok=True)
    with PdfPages(reports / "cell_line_trajectory_atlas.pdf") as pdf:
        for number, (line, group) in enumerate(rows.groupby("cell_line", observed=True, sort=True), 1):
            selected = tracks.loc[tracks.cell_line.eq(line)]
            fig = _track_figure(group, selected, f"{compartment} · {line}", sources, limits)
            slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(line))
            _save(fig, atlas_dir, f"{number:02d}_{slug}", close=False)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    return tracks.reference_row.nunique(), n_steps


def _distance_reports(rows, reports, sources):
    long = rows.melt(id_vars=["source_id", "sigma_px", *CONDITION, "fov_id"],
                     value_vars=list(METRICS), var_name="metric", value_name="distance")
    long["distance"] = long.distance.where(np.isfinite(long.distance))
    summary = long.groupby(["metric", "source_id", "sigma_px"], observed=True).distance.agg(
        n_total="size", n_finite="count", median="median", q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).reset_index()
    summary["n_undefined"] = summary.n_total - summary.n_finite
    _csv(summary, reports, "distance_median_iqr")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    for ax, (metric, label) in zip(axes, METRICS.items()):
        group = summary.loc[summary.metric.eq(metric)].sort_values("sigma_px")
        ax.plot(group.sigma_px, group["median"], "o-", color="#4f46e5")
        ax.fill_between(group.sigma_px.to_numpy(), group.q25.to_numpy(), group.q75.to_numpy(),
                        color="#818cf8", alpha=0.25, label="Object IQR (not CI)")
        ax.set(title=label, xlabel="Gaussian blur σ (px)", ylabel="Distance vs own reference")
        ax.grid(axis="y")
        ax.legend(frameon=False)
    fig.suptitle("Paired reference drift · retained sampled objects · median and IQR")
    _save(fig, reports, "distance_vs_sigma")

    palette = _blur_palette(sources)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout="constrained")
    ecdfs = []
    for ax, (metric, label) in zip(axes, METRICS.items()):
        for source in sources.itertuples():
            raw = rows.loc[rows.source_id.eq(source.source_id), metric].to_numpy()
            values, counts = np.unique(raw[np.isfinite(raw)], return_counts=True)
            probability = np.cumsum(counts) / counts.sum() if len(counts) else np.array([])
            ecdfs.append(pd.DataFrame({"metric": metric, "source_id": source.source_id,
                                       "sigma_px": source.sigma_px, "distance": values,
                                       "cumulative_probability": probability, "n_at_value": counts}))
            if len(values):
                ax.step(np.r_[values[0], values], np.r_[0, probability], where="post",
                        color=palette[source.blur], label=f"σ={source.sigma_px:g}")
        ax.set(title=label, xlabel="Distance vs own reference", ylabel="ECDF", ylim=(0, 1.02))
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Every dose · exact finite-distance ECDFs (undefined cosine excluded)")
    _save(fig, reports, "distance_ecdf")
    _csv(pd.concat(ecdfs, ignore_index=True), reports, "distance_ecdf")

    keys = ["source_id", "sigma_px", *CONDITION]
    fov = rows.groupby([*keys, "fov_id"], observed=True)[list(METRICS)].median().reset_index()
    fov["n_objects"] = rows.groupby([*keys, "fov_id"], observed=True).size().to_numpy()
    _csv(fov, reports, "distance_fov_medians")
    well = fov.groupby(keys, observed=True)[list(METRICS)].mean().reset_index()
    well["n_fovs"] = fov.groupby(keys, observed=True).size().to_numpy()
    _csv(well, reports, "distance_well_means_of_fov_medians")
    condition = well.groupby(["source_id", "sigma_px", *STRATA], observed=True)[
        list(METRICS)
    ].mean().reset_index()
    _csv(condition, reports, "distance_condition_well_balanced")
    # Equal weight to each well after equal weight to its FOVs; not an object-weighted mean.
    line = well.groupby(["source_id", "sigma_px", "cell_line"], observed=True)[
        list(METRICS)
    ].mean().reset_index()
    line["n_wells"] = well.groupby(["source_id", "sigma_px", "cell_line"], observed=True).size().to_numpy()
    _csv(line, reports, "distance_line_well_balanced")
    grid = line.pivot(index="cell_line", columns="sigma_px", values="l2")
    fig, ax = plt.subplots(figsize=(11, max(6, len(grid) * 0.36)), layout="constrained")
    _heatmap(ax, grid, "Blur sensitivity · FOV then well balanced", label="Reference-standardized L2")
    ax.set_xlabel("Gaussian blur σ (px); 0 = reference")
    _save(fig, reports, "distance_line_dose_balanced_heatmap")
    return summary


def _validate_embedding(rows, sources, compartment):
    required = {"row_index", "reference_row", "raw_row_index", "source_id", "sigma_px", "compartment",
                "object_id", "fov_id", "UMAP1", "UMAP2", "n_imputed", "n_abs_scaled_gt10",
                *CONDITION, *METRICS}
    if required - set(rows.columns):
        raise ValueError(f"{compartment}: missing embedding columns: {sorted(required - set(rows))}")
    if rows.empty or not np.array_equal(rows.row_index, np.arange(len(rows))):
        raise ValueError(f"{compartment}: embedding rows must be nonempty and array-aligned")
    if not np.isfinite(rows[["UMAP1", "UMAP2", "sigma_px"]].to_numpy()).all():
        raise ValueError(f"{compartment}: nonfinite map coordinates or doses")
    if set(rows.compartment) != {compartment} or set(rows.source_id) != set(sources.source_id):
        raise ValueError(f"{compartment}: compartment/source coverage mismatch")
    if rows.duplicated(["reference_row", "source_id"]).any():
        raise ValueError(f"{compartment}: duplicate object/source pairs")
    refs = rows.loc[rows.sigma_px.eq(0)].sort_values("row_index")
    if not len(refs) or not np.array_equal(refs.reference_row, refs.row_index):
        raise ValueError(f"{compartment}: invalid self-paired reference rows")
    identity = ["fov_id", "object_id", "compartment", *CONDITION]
    for source in sources.itertuples():
        group = rows.loc[rows.source_id.eq(source.source_id)].sort_values("reference_row")
        if (not group.sigma_px.eq(source.sigma_px).all()
                or not np.array_equal(group.reference_row, refs.row_index)
                or not group[identity].reset_index(drop=True).equals(refs[identity].reset_index(drop=True))):
            raise ValueError(f"{compartment}: unmatched source identities/doses")


def render_cp_reports(output):
    """Render prepared CP audits and all available compartment embedding reports.

    Parameters
    ----------
    output : str or pathlib.Path
        Prepared analysis directory, e.g. the CP pilot's ``analysis_embedding``.

    Returns
    -------
    dict
        ``report_dir`` and ``summary`` Paths, ``compartments`` with artifact counts,
        ``pending_compartments`` without embedding files, and all generated ``files``.

    Full-data QC always uses the five preparation CSVs. Embedding reports cover
    only sampled, paired-QC-retained objects. Missing embeddings are explicitly
    pending; malformed available embeddings raise rather than silently disappear.
    Repeated calls overwrite deterministic report artifacts, never source inputs.
    """
    output = Path(output).expanduser().resolve()
    config = _json(output / "analysis.json")
    sources = pd.DataFrame(config["sources"]).sort_values(["sigma_px", "source_id"])
    if (sources.empty or sources.source_id.duplicated().any() or sources.sigma_px.duplicated().any()
            or not np.isfinite(sources.sigma_px).all() or sources.sigma_px.iloc[0] != 0
            or (sources.sigma_px.iloc[1:] <= 0).any()):
        raise ValueError("Expected unique sources and increasing blur doses starting with reference")
    folders = {name: output / "compartments" / name for name in sorted(config["compartments"])}
    raw_rows = [pd.read_parquet(folder / "raw_profiles.parquet", columns=STRATA)
                for folder in folders.values()]
    if not raw_rows:
        raise ValueError("No prepared compartments declared")
    palettes = _palettes(raw_rows)
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "category_colors.json").write_text(json.dumps(palettes, indent=2))
    (reports / "blur_colors.json").write_text(json.dumps(_blur_palette(sources), indent=2))
    result = {"report_dir": reports, "summary": reports / "missingness_summary.md",
              "compartments": {}, "pending_compartments": []}
    with mpl.rc_context(STYLE):
        totals, structural, warnings, coverage = _audit_reports(output, reports, sources)
        lines = ["# CP blur profiling · missingness and embedding report", "",
                 "## Full-data audit (all profiled eligible objects, not the embedding sample)",
                 "Rates below are fractions, not percentages; NaN and infinity are disjoint counts.",
                 _markdown(totals.drop(columns="blur")), "", "### Structural padding",
                 _markdown(structural), "",
                 (f"Structural slots: **{int(structural.n_structural.sum()):,}**; unexpected non-NaN "
                  f"structural values: **{int(structural.n_not_nan.sum()):,}**."),
                 ("Structural NaNs are out-of-scope catalog padding, not applicable measurement failures. "
                 "Applicable NaN rates use n_missing / n_values; infinity is reported separately. "
                  "Geometry remains in these audits but is excluded from all embedded features."),
                 "", "### Warnings",
                 (f"**{int(warnings['count'].sum()):,} emissions** across "
                 f"**{int(warnings.n_records.sum()):,} log records**; "
                  f"**{len(warnings):,} message/file/category groups**."),
                 _markdown(warnings.head(12)),
                 ("Counts sum the recorded count field across ALL logged attempts, including failed/retried "
                 "attempts. They are neither affected-object counts nor rates. File/category/message and "
                 "source/FOV attribution are preserved; logs do not establish which feature or object failed. "
                  "Attempt counts per warning group overlap and must not be added as unique attempts."),
                 "", "### Sampling coverage", _markdown(coverage), ""]
        for compartment, folder in folders.items():
            lines.extend([f"## {compartment}", ""])
            if not (folder / "embedding_and_drift.parquet").exists():
                result["pending_compartments"].append(compartment)
                lines.append("**Pending:** embedding_and_drift.parquet is absent; no map or distance "
                             "reports were generated for this compartment.")
                continue
            local = _json(folder / "analysis.json")
            rows = pd.read_parquet(folder / "embedding_and_drift.parquet")
            _validate_embedding(rows, sources, compartment)
            destination = reports / compartment
            destination.mkdir(exist_ok=True)
            paired, imputation, decisions = _cleanup_reports(folder, destination, rows, local)
            _maps(rows, destination, palettes, sources, compartment)
            n_tracks, n_steps = _trajectory_reports(rows, folder, destination, sources,
                                                   config["sampling"]["seed"], compartment)
            distance_summary = _distance_reports(rows, destination, sources)
            before, kept = int(paired.n_pairs_before.sum()), int(paired.n_pairs_kept.sum())
            n_imputed, n_values = int(imputation.n_imputed.sum()), int(imputation.n_values.sum())
            source_imputation = imputation.groupby(["source_id", "sigma_px"], observed=True)[
                ["n_profiles", "n_imputed", "n_values", "n_profiles_imputed", "n_abs_scaled_gt10"]
            ].sum().reset_index()
            source_imputation["imputed_fraction"] = _rate(source_imputation.n_imputed, source_imputation.n_values)
            _csv(source_imputation, destination, "imputation_by_source")
            feature_reasons = decisions.groupby("reason", observed=True).agg(
                n_features=("name", "size"), n_retained=("retained", "sum")
            ).reset_index()
            n_fit = int(rows.is_umap_fit.sum()) if "is_umap_fit" in rows else None
            result["compartments"][compartment] = {
                "report_dir": destination, "n_profiles": len(rows), "n_reference_kept": kept,
                "n_illustrative_tracks": n_tracks, "n_successive_steps": n_steps,
            }
            lines.extend([
                (f"Sampled reference objects before QC: **{before:,}**; kept: **{kept:,} "
                f"({kept / before:.2%})**; excluded pairs: **{before - kept:,} "
                 f"({1 - kept / before:.2%})**. Retained profiles over all doses: **{len(rows):,}**."),
                (f"Expanded features: **{len(decisions):,}**; retained: **{int(decisions.retained.sum()):,}**. "
                 f"Imputed final feature values: **{n_imputed:,}/{n_values:,} ({n_imputed / n_values:.3%})**."),
                (f"Illustrative objects: **{n_tracks:,}** (up to two per observed cell line × plate × "
                f"density stratum, seed {config['sampling']['seed']}); all retained-object successive "
                 f"steps: **{n_steps:,}**, not just the displayed subset."),
                (f"UMAP fit profiles: **{n_fit:,}**; transformed profiles: **{len(rows) - n_fit:,}**."
                 if n_fit is not None else "UMAP fit/transform row attribution is unavailable."),
                "### Feature cleanup decisions", _markdown(feature_reasons),
                "### Saved preprocessing / PCA / UMAP settings", "```json",
                json.dumps({key: local.get(key) for key in ["preprocessing", "pca", "umap"]}, indent=2),
                "```", "### Imputation by source", _markdown(source_imputation),
                "### Paired retention by condition", _markdown(paired),
                "### Distance medians and IQRs", _markdown(distance_summary), "",
            ])
        lines.extend([
            "## Interpretation and caveats",
            ("- Feature filtering, reference medians, and standardization are fitted on reference objects "
            "only (post-pair-QC where applicable). PCA is fitted jointly on retained source profiles; "
            "UMAP is a joint fit or joint-source landmark fit with remaining rows transformed, not a "
             "reference-only embedding. Saved configuration records the actual mode."),
            ("- Pair QC uses the initially retained feature set once. A failure at ANY dose removes that "
            "object at ALL doses. Direct row failures and exclusions caused by another source are reported "
             "separately. Selection therefore depends on variants and may bias apparent blur robustness."),
            ("- Imputation replaces both NaN and infinity using reference medians. L2 and cosine distances "
            "are in reference-standardized, median-imputed feature space; PCA distances use its joint "
            "projection. Missingness indicators and geometry are excluded; no clipping or winsorization "
             "is introduced. Imputation can attenuate or distort drift; zero-norm cosine is undefined."),
            ("- Compartment maps are separately fitted: shared category colors do NOT imply comparable "
            "coordinates. All views within a compartment reuse the same UMAP1/UMAP2 coordinates. "
            "Blurred-only zooms change limits, not the embedding, and omit reference-to-first-dose segments. "
             "Faint backgrounds show all retained objects; paths connect successive doses, not independent samples."),
            ("- Median/IQR and exact finite-value ECDFs describe objects, not independent biological "
            "replicates. Balanced summaries take FOV medians, then equal-weight FOV means within each "
            "well, then equal-weight well means per cell line/dose (plate and density remain in detailed "
             "tables). No confidence intervals, hypothesis tests, or causal claims are provided."),
            ("- Full-data row audits count object-context rows per scope; summing scope row counts does "
            "not yield unique objects. Sampling coverage concerns already-profiled eligible identities, "
            "not all original cells. Embedding/step reports cover only the paired retained sample. "
             "Low drift is not proof of preserved phenotype; UMAP geometry is descriptive, not a distance assay."),
            "", "## Artifacts",
            ("Root report CSVs preserve the five full-data audit tables and add count-weighted family/context "
            "rates, worst-feature rankings, warning groups and structural totals. Each available compartment "
            "contains cleanup configuration/decisions/row QC, condition retention and exclusions, imputation "
            "tables, maps, PNG/PDF trajectory atlas, drift and balanced tables, exact ECDFs, illustrative "
            "tracks and all_successive_steps.parquet. Step tables retain both endpoint row indices and "
            "source/dose IDs, standardized/PCA/UMAP step distances and UMAP coordinate deltas; endpoint "
             "vectors are recoverable from the unchanged row-aligned input arrays."),
        ])
        result["summary"].write_text("\n\n".join(lines) + "\n")
    result["files"] = sorted(path for path in reports.rglob("*") if path.is_file())
    return result
