"""Independent CP channel fits from a completed, frozen compartment analysis.

No raw measurements, cohorts, feature QC, medians or scalers are refitted here.
Only exact joint PCA95 and full joint UMAP are fitted independently per channel.
Existing output directories must be complete and match the entire request.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.fov_ingestion.writer import digest
from utils.profiling import cp_embedding_plots as plots
from utils.profiling import embedding
from utils.profiling.cp_embedding import _hash_files
from utils.profiling.writer import atomic_json

__all__ = ["fit_cp_channels", "render_cp_channels"]

PARENT_FILES = [
    "analysis.json", "_SUCCESS", "raw_applicable.npy", "raw_profiles.parquet",
    "feature_catalog.parquet", "joint_profiles.npy", "profiles.parquet",
    "preprocessing_model.npz", "feature_decisions.csv", "pca_model.npz",
    "pca_scores.npy", "pca_spectrum.csv", "umap_model.joblib", "embedding_and_drift.parquet",
]
CHILD_FILES = [
    "joint_profiles.npy", "profiles.parquet", "feature_catalog.parquet",
    "preprocessing_model.npz", "feature_decisions.csv", "row_missingness.parquet",
    "feature_missingness.csv", "missingness_by_line_dose.csv", "pca_model.npz",
    "pca_scores.npy", "pca_spectrum.csv", "umap_model.joblib", "embedding_and_drift.parquet",
]
CAVEATS = (
    "Exact parent kept rows and reference-fitted scaling; no new QC or resampling. "
    "Channel missingness describes retained rows/features BEFORE imputation, not full raw data. "
    "Standardized L2 depends on feature dimension and reference scales; even L2/sqrt(d) "
    "is descriptive, not calibrated cross-channel information loss. Relative L2 depends on "
    "the reference norm. Cosine is measured after reference centering/scaling and median "
    "imputation, not on raw nonnegative intensities; zero-norm cosine/ratios are undefined. "
    "Imputation can attenuate/distort drift. IQRs describe objects, not confidence intervals. "
    "All-source parent pair QC can bias robustness. Independent UMAP coordinates, orientations "
    "and distances must NOT be compared across channels or compartments. Low drift does not "
    "prove preserved phenotype. Load serialized UMAP models only from trusted outputs."
)


def _json(path):
    return json.loads(path.read_text())


def _marked(folder, marker="_SUCCESS"):
    config = _json(folder / "analysis.json")
    if (folder / marker).read_text().strip() != digest(config):
        raise ValueError(f"Completion marker/config mismatch: {folder}")
    return config


def _safe_token(value):
    if not isinstance(value, str) or value in {"", ".", ".."} or Path(value).name != value:
        raise ValueError(f"Unsafe compartment/channel name: {value!r}")
    return value


def _parent_snapshot(parent):
    root = _marked(parent, "_PREPARED")
    snapshot = {"root": _hash_files(parent, ["analysis.json", "_PREPARED"]), "compartments": {}}
    if not root["compartments"] or not root["channels"]:
        raise ValueError("Parent has no compartments/channels")
    if len(set(root["channels"])) != len(root["channels"]):
        raise ValueError("Duplicate parent channels")
    for channel in root["channels"]:
        _safe_token(channel)
    for comp in sorted(root["compartments"]):
        folder = parent / "compartments" / _safe_token(comp)
        config = _marked(folder)
        if (config["compartment"] != comp or config["sources"] != root["sources"]
            or config["channels"] != root["channels"]):
            raise ValueError(f"Parent compartment provenance mismatch: {comp}")
        for key, required in (
            ("output_digests", {"joint_profiles.npy", "profiles.parquet",
                                "preprocessing_model.npz", "feature_decisions.csv"}),
            ("raw_digests", {"raw_applicable.npy", "raw_profiles.parquet", "feature_catalog.parquet"}),
        ):
            saved = config.get(key, {})
            if not required.issubset(saved) or _hash_files(folder, list(saved)) != saved:
                raise ValueError(f"Parent {key} mismatch: {comp}")
        if config.get("pca", {}).get("target_variance") != 0.95 or (
            config.get("umap", {}).get("mode") != "full"
        ):
            raise ValueError("Parent must have completed PCA95/full UMAP")
        rows = pd.read_parquet(folder / "profiles.parquet")
        embedded = pd.read_parquet(folder / "embedding_and_drift.parquet")
        plots._validate_embedding(embedded, pd.DataFrame(root["sources"]), comp)
        pd.testing.assert_frame_equal(rows, embedded[rows.columns], check_exact=True)
        raw_rows = pd.read_parquet(folder / "raw_profiles.parquet")
        raw_ids = rows.raw_row_index.to_numpy(dtype=np.int64)
        if (raw_ids < 0).any() or (raw_ids >= len(raw_rows)).any():
            raise ValueError("Invalid parent raw_row_index")
        identity = ["fov_id", "object_id", "compartment", "source_id", "sigma_px", *plots.CONDITION]
        pd.testing.assert_frame_equal(rows[identity].reset_index(drop=True),
                                      raw_rows.iloc[raw_ids][identity].reset_index(drop=True))
        snapshot["compartments"][comp] = _hash_files(folder, PARENT_FILES)
    return root, snapshot


def _channel_columns(catalog, column_indices, channel):
    """Map matrix positions through raw catalog IDs, never through column names/slices."""
    columns = np.asarray(column_indices)
    if columns.ndim != 1 or columns.dtype.kind not in "iu" or len(set(columns)) != len(columns):
        raise ValueError("Invalid preprocessing column_indices")
    indexed = catalog.set_index("column_index", verify_integrity=True)
    if not set(columns).issubset(indexed.index):
        raise ValueError("Preprocessing columns absent from catalog")
    mapped = indexed.loc[columns].reset_index()
    positions = np.flatnonzero(mapped.scope.eq("single_channel") & mapped.context.eq(channel))
    if not len(positions):
        raise ValueError(f"No retained single_channel features: {channel}")
    selected = mapped.iloc[positions].copy().reset_index(drop=True)
    selected = selected.rename(columns={"column_index": "parent_column_index"})
    selected["parent_matrix_column"] = positions
    selected["column_index"] = np.arange(len(selected))
    return positions, selected


def _channel_rows(rows, matrix, raw, catalog, channel):
    """Recompute channel diagnostics from frozen scaled values and raw row attribution."""
    stale = [key for key in rows if key in {
        "l2", "relative_l2", "cosine_distance", "distance_space", "n_imputed",
        "n_abs_scaled_gt10", "is_umap_fit",
    } or key.lower().startswith(("pca_", "umap", "cosine_", "l2_", "relative_l2_"))]
    result = rows.drop(columns=stale).copy()
    if not np.array_equal(result.row_index, np.arange(len(matrix))):
        raise ValueError("Parent matrix/row alignment mismatch")
    raw_ids = result.raw_row_index.to_numpy(dtype=np.int64)
    refs = result.reference_row.to_numpy(dtype=np.int64)
    if (raw_ids < 0).any() or (raw_ids >= len(raw)).any() or (
        (refs < 0).any() or (refs >= len(matrix)).any()
    ):
        raise ValueError("Invalid raw/reference row indices")
    result["channel"] = channel
    for key in ["l2", "relative_l2", "cosine_distance", "n_imputed", "n_abs_scaled_gt10"]:
        result[key] = 0 if key.startswith("n_") else np.nan
    missing, infinite = np.zeros(len(rows), dtype=np.int64), np.zeros(len(rows), dtype=np.int64)
    feature_missing = np.zeros((len(catalog), 2), dtype=np.int64)
    raw_columns = catalog.parent_column_index.to_numpy(dtype=np.int64)
    for start in range(0, len(rows), 4096):
        stop = min(start + 4096, len(rows))
        values = np.asarray(matrix[start:stop])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite frozen standardized matrix")
        before = raw[np.ix_(raw_ids[start:stop], raw_columns)]
        nan, inf = np.isnan(before), np.isinf(before)
        missing[start:stop], infinite[start:stop] = nan.sum(axis=1), inf.sum(axis=1)
        feature_missing += np.column_stack([nan.sum(axis=0), inf.sum(axis=0)])
        ids = result.index[start:stop]
        result.loc[ids, ["l2", "relative_l2", "cosine_distance"]] = np.column_stack(
            embedding.distances(matrix[refs[start:stop]], values)
        )
        result.loc[ids, "n_imputed"] = (nan | inf).sum(axis=1)
        result.loc[ids, "n_abs_scaled_gt10"] = (np.abs(values) > 10).sum(axis=1)
    result["distance_space"] = "parent reference-standardized median-imputed single-channel CP"
    qc = result.drop(columns=["l2", "relative_l2", "cosine_distance", "distance_space"]).copy()
    qc["n_missing"], qc["n_inf"] = missing, infinite
    qc["n_nonfinite"] = missing + infinite
    qc["n_features"] = len(catalog)
    qc["nonfinite_fraction"] = qc.n_nonfinite / len(catalog)
    features = catalog.copy()
    features["n_values"] = len(rows)
    features[["n_missing", "n_inf"]] = feature_missing
    return result, qc, features


def _validate_child(folder, request):
    config = _marked(folder)
    if config.get("channel_request") != request:
        raise ValueError(f"Channel request mismatch: {folder}")
    hashes = config.get("channel_output_digests", {})
    if set(hashes) != set(CHILD_FILES) or _hash_files(folder, CHILD_FILES) != hashes:
        raise ValueError(f"Channel output digests mismatch: {folder}")
    fit = request["fit"]
    if config["pca"]["target_variance"] != .95 or any(
        config["umap"][key] != value for key, value in fit.items() if key != "threads"
    ):
        raise ValueError(f"Saved fit settings mismatch: {folder}")
    rows = pd.read_parquet(folder / "embedding_and_drift.parquet")
    plots._validate_embedding(rows, pd.DataFrame(config["sources"]), request["compartment"])
    if not rows.channel.eq(request["channel"]).all() or not rows.is_umap_fit.all():
        raise ValueError("Expected full single-channel embedding")
    return config, rows


def _children(request):
    for comp in sorted(request["parent_snapshot"]["compartments"]):
        for channel in request["channels"]:
            yield comp, channel, {**request, "compartment": comp, "channel": channel}


def _completed(output, request):
    config = _marked(output)
    if config.get("request") != request:
        raise ValueError("Output request/parent provenance mismatch; use a fresh output")
    expected = [f"compartments/{comp}/{channel}/analysis.json"
                for comp, channel, _ in _children(request)] + ["channel_fit_summary.csv"]
    if set(config.get("output_digests", {})) != set(expected) or (
        _hash_files(output, expected) != config["output_digests"]
    ):
        raise ValueError("Root output digests mismatch")
    children = {}
    for comp, channel, child_request in _children(request):
        folder = output / "compartments" / comp / channel
        children[comp, channel] = _validate_child(folder, child_request)
    return config, children


def fit_cp_channels(parent, output, seed=42, n_neighbors=30, min_dist=.15, threads=8, reuse=True):
    """Return a summary DataFrame; write independent fits to compartments/{comp}/{channel}.

    ``parent`` is the completed 2.3 analysis_embedding directory, not its raw run.
    No real fits run on import. Incomplete directories fail, including interrupted roots;
    use a fresh output rather than mixing partial artifacts with a later request.
    """
    parent, output = Path(parent).expanduser().resolve(), Path(output).expanduser().resolve()
    if output == parent or output in parent.parents:
        raise ValueError("Output must not replace the parent or its ancestors")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in [seed, n_neighbors, threads]):
        raise ValueError("seed, n_neighbors and threads must be integers")
    if seed < 0 or n_neighbors < 2 or threads < 1 or not np.isfinite(min_dist) or not 0 <= min_dist <= 1:
        raise ValueError("Invalid fit parameters")
    root, snapshot = _parent_snapshot(parent)
    request = {
        "schema_version": 1, "parent": str(parent), "parent_snapshot": snapshot,
        "parent_digest": digest(snapshot), "channels": root["channels"],
        "fit": {"seed": seed, "n_neighbors": n_neighbors, "min_dist": min_dist,
                "threads": threads, "mode": "full"},
    }
    if output.exists():
        if not reuse or not (output / "_SUCCESS").is_file():
            raise FileExistsError(f"Incomplete/existing output: {output}; use a fresh output")
        _completed(output, request)
        return pd.read_csv(output / "channel_fit_summary.csv")
    # Preflight every mapping and cohort before creating any output or starting a fit.
    for comp, channel, _ in _children(request):
        folder = parent / "compartments" / comp
        with np.load(folder / "preprocessing_model.npz") as model:
            _channel_columns(pd.read_parquet(folder / "feature_catalog.parquet"),
                             model["column_indices"], channel)
        if len(pd.read_parquet(folder / "profiles.parquet")) <= n_neighbors:
            raise ValueError("UMAP needs more profiles than n_neighbors")
    output.mkdir(parents=True)
    atomic_json(output / "analysis.json", {"request": request, "sources": root["sources"]})
    summary = []
    for comp, channel, child_request in _children(request):
        source = parent / "compartments" / comp
        folder = output / "compartments" / comp / channel
        folder.mkdir(parents=True)
        with np.load(source / "preprocessing_model.npz") as saved:
            model = {key: saved[key].copy() for key in saved.files}
        positions, catalog = _channel_columns(
            pd.read_parquet(source / "feature_catalog.parquet"), model["column_indices"], channel
        )
        parent_x = np.load(source / "joint_profiles.npy", mmap_mode="r")
        if parent_x.shape[1] != len(model["column_indices"]):
            raise ValueError("Parent preprocessing/matrix dimension mismatch")
        matrix = np.lib.format.open_memmap(folder / "joint_profiles.npy", mode="w+",
                                           dtype=parent_x.dtype, shape=(len(parent_x), len(positions)))
        for start in range(0, len(matrix), 4096):
            matrix[start:start + 4096] = parent_x[start:start + 4096, positions]
        matrix.flush()
        rows, qc, missingness = _channel_rows(
            pd.read_parquet(source / "profiles.parquet"), matrix,
            np.load(source / "raw_applicable.npy", mmap_mode="r"), catalog, channel
        )
        rows.to_parquet(folder / "profiles.parquet", index=False)
        qc.to_parquet(folder / "row_missingness.parquet", index=False)
        missingness.to_csv(folder / "feature_missingness.csv", index=False)
        qc.groupby(["source_id", "sigma_px", "cell_line"], observed=True)[
            ["n_missing", "n_inf", "n_nonfinite", "n_features"]
        ].sum().rename(columns={"n_features": "n_values"}).to_csv(
            folder / "missingness_by_line_dose.csv"
        )
        catalog.to_parquet(folder / "feature_catalog.parquet", index=False)
        names = catalog.name.to_numpy(dtype=str)
        np.savez(folder / "preprocessing_model.npz",
                 **{key: model[key][positions] for key in
                    ["reference_medians", "mean", "std", "variance"]},
                 n_reference=model["n_reference"], feature_names=names,
                 retained_feature_names=names, feature_mask=np.ones(len(names), dtype=bool),
                 column_indices=np.arange(len(names)), parent_matrix_columns=positions,
                 parent_column_indices=catalog.parent_column_index.to_numpy(),
                 parent_digest=request["parent_digest"])
        decisions = pd.read_csv(source / "feature_decisions.csv")
        decisions = decisions.loc[decisions.scope.eq("single_channel")
                                  & decisions.context.eq(channel)].copy()
        decisions = decisions.rename(columns={"column_index": "parent_column_index"})
        decisions["column_index"] = decisions.parent_column_index.map(
            catalog.set_index("parent_column_index").column_index
        )
        decisions["decision_provenance"] = "inherited unchanged from parent QC"
        decisions.to_csv(folder / "feature_decisions.csv", index=False)
        config = {
            "channel_request": child_request, "compartment": comp, "channels": [channel],
            "sources": root["sources"], "n_profiles": len(rows), "n_features": len(names),
            "parent_digest": request["parent_digest"],
            "feature_mapping": catalog[["column_index", "parent_column_index",
                                         "parent_matrix_column", "name"]].to_dict("records"),
            "preprocessing": {"policy": "subset parent reference-fitted preprocessing; no refit",
                              "cohort": "exact parent kept rows", "clip": False,
                              "missingness_indicators": False}, "caveats": CAVEATS,
        }
        atomic_json(folder / "analysis.json", config)  # Must precede either fitting helper.
        scores, _ = embedding.fit_pca95(matrix, folder, variance=.95, threads=threads)
        embedded = embedding.embed_umap(scores, rows, folder, seed=seed, n_neighbors=n_neighbors,
                                        min_dist=min_dist, mode="full")
        plots._validate_embedding(embedded, pd.DataFrame(root["sources"]), comp)
        config = _json(folder / "analysis.json")
        config["channel_output_digests"] = _hash_files(folder, CHILD_FILES)
        atomic_json(folder / "analysis.json", config)
        (folder / "_SUCCESS").write_text(digest(config))
        _validate_child(folder, child_request)
        summary.append({"compartment": comp, "channel": channel, "profiles": len(rows),
                        "reference_objects": int(rows.sigma_px.eq(0).sum()),
                        "cell_lines": rows.cell_line.nunique(), "features": len(names),
                        "n_imputed": int(rows.n_imputed.sum()), **config["pca"]})
    result = pd.DataFrame(summary)
    result.to_csv(output / "channel_fit_summary.csv", index=False)
    files = [f"compartments/{comp}/{channel}/analysis.json"
             for comp, channel, _ in _children(request)] + ["channel_fit_summary.csv"]
    config = {"request": request, "sources": root["sources"],
              "output_digests": _hash_files(output, files), "caveats": CAVEATS}
    atomic_json(output / "analysis.json", config)
    (output / "_SUCCESS").write_text(digest(config))
    return result


def _curve_summary(rows, keys, metrics):
    frame = rows.melt(id_vars=keys, value_vars=metrics, var_name="metric", value_name="distance")
    frame["distance"] = frame.distance.where(np.isfinite(frame.distance))
    summary = frame.groupby([*keys, "metric"], observed=True).distance.agg(
        n_total="size", n_finite="count", median="median",
        q25=lambda x: x.quantile(.25), q75=lambda x: x.quantile(.75),
    ).reset_index()
    summary["n_undefined"] = summary.n_total - summary.n_finite
    return summary


def _curves(summary, reports, name, hue, palette, x="sigma_px"):
    metrics = list(summary.metric.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5),
                             squeeze=False, layout="constrained")
    for ax, metric in zip(axes.flat, metrics):
        for value, group in summary.loc[summary.metric.eq(metric)].groupby(hue, observed=True):
            group = group.sort_values(x)
            color = palette[str(value)]
            ax.plot(group[x], group["median"], "o-", label=str(value), color=color)
            ax.fill_between(group[x].to_numpy(), group.q25.to_numpy(), group.q75.to_numpy(),
                            color=color, alpha=.12)
        ax.set(title=metric, xlabel="Gaussian blur σ (px)" if x == "sigma_px" else x,
               ylabel="Median distance · object IQR (not CI)")
        ax.grid(axis="y")
    axes.flat[-1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle(name.replace("_", " "))
    plots._save(fig, reports, name)


def _direct_low_high(rows, folder):
    doses = sorted(rows.loc[rows.sigma_px.gt(0), "sigma_px"].unique())
    if not doses:
        raise ValueError("No positive blur doses")
    low = rows.loc[rows.sigma_px.eq(doses[0])].sort_values("reference_row")
    high = rows.loc[rows.sigma_px.eq(doses[-1])].sort_values("reference_row")
    if not np.array_equal(low.reference_row, high.reference_row):
        raise ValueError("Unpaired low/high endpoints")
    x = np.load(folder / "joint_profiles.npy", mmap_mode="r")
    result = high[["reference_row", "fov_id", "object_id", "cell_line", "channel",
                   *[key for key in plots.CONDITION if key != "cell_line"]]].copy()
    result["from_row_index"], result["to_row_index"] = low.row_index.to_numpy(), high.row_index.to_numpy()
    result["from_sigma_px"], result["to_sigma_px"] = doses[0], doses[-1]
    result["from_source_id"], result["to_source_id"] = low.source_id.to_numpy(), high.source_id.to_numpy()
    for key in ["l2", "relative_l2", "cosine_distance"]:
        result[key] = np.nan
    for start in range(0, len(result), 4096):
        block = result.iloc[start:start + 4096]
        result.loc[block.index, ["l2", "relative_l2", "cosine_distance"]] = np.column_stack(
            embedding.distances(x[block.from_row_index], x[block.to_row_index])
        )
    return result


def render_cp_channels(output):
    """Validate every completed fit and parent before writing any reports; never fit.

    Returns ``output``, ``report_dir``, ``summary``, ``channels`` and generated ``files`` Paths.
    Trajectory RNG uses the actual compartment (not a channel-specific title token).
    """
    output = Path(output).expanduser().resolve()
    request = _marked(output)["request"]
    root, snapshot = _parent_snapshot(Path(request["parent"]))
    if snapshot != request["parent_snapshot"]:
        raise ValueError("Parent changed since channel fitting")
    _, children = _completed(output, request)  # Preflight ALL fits before plotting.
    palettes = plots._palettes([
        pd.read_parquet(Path(request["parent"]) / "compartments" / comp / "raw_profiles.parquet",
                        columns=plots.STRATA)
        for comp in sorted(snapshot["compartments"])
    ])
    sources = pd.DataFrame(root["sources"]).sort_values(["sigma_px", "source_id"])
    reports = output / "reports"
    reports.mkdir(exist_ok=True)
    atomic_json(reports / "category_colors.json", palettes)
    atomic_json(reports / "blur_colors.json", plots._blur_palette(sources))
    result = {"output": output, "report_dir": reports,
              "summary": reports / "channel_summary.md", "channels": {}}
    pooled, steps, direct = {}, {}, {}
    with mpl.rc_context(plots.STYLE):
        for (comp, channel), (local, rows) in children.items():
            folder = output / "compartments" / comp / channel
            destination = reports / comp / channel
            destination.mkdir(parents=True, exist_ok=True)
            spectrum = pd.read_csv(folder / "pca_spectrum.csv")
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
            ax.plot(spectrum.component, spectrum.cumulative_variance, color="#245b78")
            ax.axhline(.95, color="#cd6743", linestyle="--")
            ax.axvline(local["pca"]["n_components"], color="#67757c", linestyle=":")
            ax.set(title=f"{comp} · {channel} · independent PCA", xlabel="Components",
                   ylabel="Cumulative explained variance")
            plots._save(fig, destination, "pca_variance")
            plots._maps(rows, destination, palettes, sources, f"{comp} · {channel}")
            # Keep compartment token identical across channels to preserve seeded tracks.
            tracks, n_steps = plots._trajectory_reports(
                rows, folder, destination, sources, request["fit"]["seed"], comp
            )
            plots._distance_reports(rows, destination, sources)
            enriched = rows.copy()
            enriched["l2_per_sqrt_feature"] = enriched.l2 / np.sqrt(local["n_features"])
            metrics = ["l2", "l2_per_sqrt_feature", "relative_l2", "cosine_distance"]
            line = _curve_summary(enriched, ["cell_line", "source_id", "sigma_px"], metrics)
            line.to_csv(destination / "distance_line_median_iqr.csv", index=False)
            _curves(line, destination, "per_line_dose_curves", "cell_line", palettes["cell_line"])
            pooled.setdefault(comp, []).append(enriched)
            step = pd.read_parquet(destination / "all_successive_steps.parquet")
            step["channel"] = channel
            step["standardized_l2_per_sqrt_feature"] = step.standardized_l2 / np.sqrt(local["n_features"])
            steps.setdefault(comp, []).append(step)
            endpoints = _direct_low_high(rows, folder)
            endpoints.to_parquet(destination / "low_high_direct.parquet", index=False)
            direct.setdefault(comp, []).append(endpoints)
            qc = pd.read_parquet(folder / "row_missingness.parquet")
            imputation = qc.groupby(["source_id", "sigma_px", "cell_line"], observed=True)[
                ["n_missing", "n_inf", "n_imputed", "n_abs_scaled_gt10", "n_features"]
            ].sum().rename(columns={"n_features": "n_values"}).reset_index()
            imputation["imputed_fraction"] = imputation.n_imputed / imputation.n_values
            imputation.to_csv(destination / "imputation_by_line_dose.csv", index=False)
            result["channels"][f"{comp}/{channel}"] = {
                "output": folder, "report_dir": destination, "n_illustrative_tracks": tracks,
                "n_successive_steps": n_steps,
            }
        for comp, reference_frames in pooled.items():
            destination = reports / comp
            condition_tables = {
                channel: pd.read_csv(destination / channel / "distance_condition_well_balanced.csv")
                for channel in request["channels"]
            }
            maximum = max(float(frame.l2.max()) for frame in condition_tables.values())
            for channel, frame in condition_tables.items():
                doses = sources.loc[sources.sigma_px.gt(0), "sigma_px"]
                fig, axes = plt.subplots(1, len(doses), figsize=(16, 8), squeeze=False,
                                         layout="constrained")
                wells = pd.read_csv(
                    destination / channel / "distance_well_means_of_fov_medians.csv"
                )
                for ax, dose in zip(axes.flat, doses):
                    # Equal weight to wells across plates after within-well FOV balancing.
                    grid = wells.loc[wells.sigma_px.eq(dose)].groupby(
                        ["cell_line", "seeding_density"], observed=True
                    ).l2.mean().unstack("seeding_density")
                    plots._heatmap(ax, grid, f"σ={dose:.3g} px",
                                   label="Well-balanced standardized L2", vmax=maximum)
                    ax.set_xlabel("Seeding density")
                fig.suptitle(f"{comp} · {channel} · shared L2 scale across channels")
                plots._save(fig, destination / channel, "condition_density_heatmap")
            colors = dict(zip(request["channels"],
                              mpl.colormaps["tab10"](np.linspace(0, .9, len(request["channels"]))).tolist()))
            atomic_json(destination / "channel_colors.json", colors)
            for name, frame, metrics, x, extra in (
                ("channel_reference_dose", pd.concat(reference_frames, ignore_index=True),
                 ["l2", "l2_per_sqrt_feature", "relative_l2", "cosine_distance"], "sigma_px", []),
                ("channel_successive_steps", pd.concat(steps[comp], ignore_index=True),
                 ["standardized_l2", "standardized_l2_per_sqrt_feature",
                  "standardized_cosine_distance"], "to_sigma_px", ["from_sigma_px"]),
            ):
                keys = ["channel", x, *extra]
                summary = _curve_summary(frame, keys, metrics)
                summary.to_csv(destination / f"{name}_median_iqr.csv", index=False)
                _curves(summary, destination, name, "channel", colors, x=x)
                by_line = _curve_summary(frame, [*keys, "cell_line"], metrics)
                by_line.to_csv(destination / f"{name}_by_line_median_iqr.csv", index=False)
            endpoints = pd.concat(direct[comp], ignore_index=True)
            endpoints.to_parquet(destination / "channel_low_high_direct.parquet", index=False)
            summary = _curve_summary(endpoints, ["channel", "cell_line", "from_sigma_px", "to_sigma_px"],
                                     ["l2", "relative_l2", "cosine_distance"])
            summary.to_csv(destination / "channel_low_high_direct_median_iqr.csv", index=False)
            grid = summary.loc[summary.metric.eq("cosine_distance")].pivot(
                index="cell_line", columns="channel", values="median"
            ).reindex(columns=request["channels"])
            fig, ax = plt.subplots(figsize=(11, max(6, len(grid) * .36)), layout="constrained")
            plots._heatmap(ax, grid, f"{comp} · direct lowest→highest positive blur",
                           label="Median standardized-space cosine distance", vmax=2)
            plots._save(fig, destination, "channel_low_high_direct_cosine_heatmap")
    result["summary"].write_text(
        "# CP channel embeddings\n\n" + CAVEATS + "\n\n"
        "Each channel report contains maps, global trajectories, one PNG/PDF per observed cell "
        "line (18 for the completed pilot), a combined PDF atlas, exact ECDFs, dose curves and "
        "all-object successive-step tables. Atlas titles identify compartment/line; their parent "
        "directory identifies channel. Track RNG is shared across channels within each compartment. "
        "Low→high means the lowest POSITIVE dose directly to the highest, not sums of steps. "
        "Compartment comparison reports use only frozen standardized feature-space distances, "
        "never independent UMAP coordinates. All tables include undefined-distance counts.\n\n"
        + plots._markdown(pd.read_csv(output / "channel_fit_summary.csv")) + "\n"
    )
    result["files"] = sorted(path for path in reports.rglob("*") if path.is_file())
    return result
