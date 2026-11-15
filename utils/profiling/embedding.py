"""Joint, channel-concatenated profiles and bounded-memory exact covariance PCA.

Large arrays live on disk. A row is one reference-anchored nucleus context in one
source, not one channel. No standardization, whitening, or L2 normalization is
applied: distances retain the original MorphEm feature geometry.
"""

import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.writer import digest

KEYS = ["object_id", "compartment", "crop_id"]
STRATA = ["cell_line", "seeding_density", "plate"]


def stratified_anchors(anchors, cells_per_stratum, seed=42):
    """Uniform within-stratum sampling, capped equally; rare strata retain all cells.

    Sorting before seeded sampling makes selection independent of input row order.
    Each returned identity is later retained in every source, never sampled separately.
    """
    if not isinstance(cells_per_stratum, int) or cells_per_stratum < 1:
        raise ValueError("cells_per_stratum must be a positive integer")
    if anchors.duplicated(["fov_id", *KEYS]).any():
        raise ValueError("Duplicate anchor identities")
    if anchors[STRATA].isna().any().any():
        raise ValueError("Missing sampling strata")
    ordered = anchors.sort_values([*STRATA, "fov_id", *KEYS])
    rng = np.random.default_rng(seed)
    selected = []
    for _, group in ordered.groupby(STRATA, observed=True, sort=True):
        ids = rng.choice(len(group), min(cells_per_stratum, len(group)), replace=False)
        selected.append(group.iloc[ids])
    if not selected:
        raise ValueError("No eligible anchors")
    return pd.concat(selected, ignore_index=True).sort_values(["fov_id", *KEYS])


def inspect_run(root, fov_limit=None):
    """Validate completion, source identity, complete manifest, and biological metadata."""
    root = Path(root).resolve()
    spec = json.loads((root / "run.json").read_text())
    if (root / "_SUCCESS").read_text().strip() != digest(spec):
        raise ValueError("Profiling completion marker does not match run.json")
    manifest = pd.read_parquet(root / "manifest.parquet")
    expected = pd.MultiIndex.from_product(
        [spec["fov_ids"], [s["source_id"] for s in spec["sources"]]],
        names=["fov_id", "source_id"],
    )
    manifest = manifest.set_index(["fov_id", "source_id"], verify_integrity=True)
    if set(manifest.index) != set(expected):
        raise ValueError("Manifest does not cover exactly every FOV/source")
    if not manifest.n_rows.eq(manifest.n_eligible * len(spec["channels"])).all():
        raise ValueError("Expected one observation per eligible nucleus and channel")
    sources = []
    reference = None
    for i, source in enumerate(spec["sources"]):
        collection = MicroscopyCollection(root / source["uri"])
        if collection.fingerprint != source["fingerprint"]:
            raise ValueError("Source fingerprint changed")
        if i == 0:
            reference = collection
            sigma = 0.0
            if collection.fingerprint != spec["reference_fingerprint"]:
                raise ValueError("Wrong reference source")
        else:
            transform = collection.spec["transform"]
            if (
                transform["name"] != "gaussian_blur"
                or transform["mask_policy"] != "fixed_reference"
                or collection.spec["reference_fingerprint"] != spec["reference_fingerprint"]
            ):
                raise ValueError("Expected fixed-reference Gaussian blur")
            sigma = float(transform["parameters"]["sigma_px"])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Variant sigma must be positive and finite")
        sources.append({**source, "sigma_px": sigma})
    sources = sorted(sources, key=lambda source: source["sigma_px"])
    for i, source in enumerate(sources):
        source["blur"] = "Reference" if i == 0 else f"Blur {i} (σ={source['sigma_px']:.2f})"
    if reference is None:
        raise ValueError("No reference source")
    metadata = reference.metadata()
    required = ["fov_id", "plate", "well", "cell_line", "seeding_density"]
    metadata = metadata[required].copy().set_index("fov_id", verify_integrity=True)
    metadata["seeding_density"] = pd.to_numeric(metadata.seeding_density, errors="raise")
    fovs = sorted(spec["fov_ids"])
    if fov_limit is not None:
        if not isinstance(fov_limit, int) or fov_limit < 1:
            raise ValueError("fov_limit must be None or a positive integer")
        fovs = fovs[:fov_limit]
    metadata = metadata.loc[fovs]
    if metadata.isna().any().any():
        raise ValueError("Missing biological metadata; resolve it before plotting")
    manifest = manifest.loc[fovs]
    counts = manifest.n_eligible.unstack("source_id")
    if not counts.eq(counts.iloc[:, 0], axis=0).all().all():
        raise ValueError("Eligible nucleus counts differ across sources")
    return spec, sources, metadata, manifest


def _read_cell_profiles(root, row, source, spec, fid):
    table = pd.read_parquet(Path(root) / row.feature_uri)
    if len(table) != row.n_rows:
        raise ValueError("Shard row count differs from manifest")
    for column, value in {
        "profile_run_id": spec["profile_run_id"],
        "feature_set_id": spec["feature_set_id"],
        "source_id": source["source_id"],
        "fov_id": fid,
    }.items():
        if not table[column].eq(value).all():
            raise ValueError(f"Shard provenance mismatch: {column}")
    if not all(list(cs) == [c] for cs, c in zip(table.channels, table.channel)):
        raise ValueError("Expected single-channel observations")
    if not all(list(ids) == [i] for ids, i in zip(
        table.reference_image_ids, table.reference_image_id
    )):
        raise ValueError("Inconsistent reference image attribution")
    if set(table.channel) != set(spec["channels"]) and len(table):
        raise ValueError("Unexpected channels")
    blocks, image_ids = [], []
    index = None
    dimension = len(spec["profiler"]["feature_names"])
    for channel in spec["channels"]:
        block = table.loc[table.channel.eq(channel)].set_index(KEYS, verify_integrity=True)
        block = block.sort_index()
        if index is None:
            index = block.index
        if not block.index.equals(index):
            raise ValueError("Channels have different nucleus/crop identities")
        matrix = np.stack(block.features) if len(block) else np.empty((0, dimension))
        if matrix.shape != (len(block), dimension) or not np.isfinite(matrix).all():
            raise ValueError("Invalid feature vectors")
        blocks.append(matrix)
        image_ids.append(block.reference_image_id.to_numpy())
    if len(index) != row.n_eligible:
        raise ValueError("Cell count differs from manifest")
    return index, np.concatenate(blocks, axis=1), np.stack(image_ids, axis=1)


def distances(reference, variant):
    """Original-space distances; undefined zero-norm ratios/cosines remain NaN."""
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(variant, dtype=np.float64)
    norm_a, norm_b = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    l2 = np.linalg.norm(b - a, axis=1)
    relative = np.divide(l2, norm_a, out=np.full(len(a), np.nan), where=norm_a > 0)
    denominator = norm_a * norm_b
    cosine = np.divide(
        np.sum(a * b, axis=1), denominator,
        out=np.full(len(a), np.nan), where=denominator > 0,
    )
    return l2, relative, 1 - np.clip(cosine, -1, 1)


def assemble_profiles(root, output, fov_limit=None, *, cells_per_stratum=None, seed=42):
    """Strictly pair all channels/sources and write a row-indexed disk-backed matrix.

    Use a new directory for every analysis; never silently reuse stale partial data.
    """
    root, output = Path(root), Path(output)
    spec, sources, metadata, manifest = inspect_run(root, fov_limit)
    selected = None
    coverage = None
    selected_by_fov = None
    if cells_per_stratum is not None:
        anchors = pd.read_parquet(root / "anchors.parquet", columns=["fov_id", *KEYS, "eligible"])
        anchors = anchors.loc[anchors.eligible & anchors.fov_id.isin(metadata.index)].copy()
        counts = anchors.groupby("fov_id").size().reindex(metadata.index, fill_value=0)
        expected = manifest.xs(sources[0]["source_id"], level="source_id").n_eligible
        if not counts.eq(expected.reindex(counts.index)).all():
            raise ValueError("Eligible anchor catalog differs from feature manifest")
        anchors = anchors.merge(metadata.reset_index(), on="fov_id", validate="many_to_one")
        available = anchors.groupby(STRATA, observed=True).size().rename("available_anchors")
        strata_index = pd.MultiIndex.from_frame(metadata[STRATA].drop_duplicates())
        available = available.reindex(strata_index, fill_value=0)
        if available.eq(0).any():
            raise ValueError(f"Conditions with no eligible anchors: {available[available.eq(0)]}")
        selected = stratified_anchors(anchors, cells_per_stratum, seed)
        coverage = available.to_frame().join(
            selected.groupby(STRATA, observed=True).size().rename("selected_anchors")
        )
        coverage["sampling_fraction"] = coverage.selected_anchors / coverage.available_anchors
        coverage = coverage.join(
            selected.groupby(STRATA, observed=True).fov_id.nunique().rename("selected_fovs")
        )
        selected_by_fov = {
            fid: pd.MultiIndex.from_frame(group[KEYS])
            for fid, group in selected.groupby("fov_id", sort=False)
        }
    output.mkdir(parents=True, exist_ok=False)
    if selected is not None:
        selected.to_parquet(output / "sampled_anchors.parquet", index=False)
        coverage.to_csv(output / "sampling_coverage.csv")
    channels = spec["channels"]
    dimension = len(spec["profiler"]["feature_names"])
    n = int(manifest.n_eligible.sum()) if selected is None else len(selected) * len(sources)
    if n < 3:
        raise ValueError("Not enough profiles")
    x = np.lib.format.open_memmap(
        output / "joint_profiles.npy", mode="w+", dtype="float32",
        shape=(n, len(channels) * dimension),
    )
    frames, offset = [], 0
    fovs_to_read = metadata.index if selected_by_fov is None else list(selected_by_fov)
    for fi, fid in enumerate(fovs_to_read):
        reference_index, reference, reference_images = _read_cell_profiles(
            root, manifest.loc[(fid, sources[0]["source_id"])], sources[0], spec, fid
        )
        keep = np.ones(len(reference), dtype=bool)
        if selected_by_fov is not None:
            keep = reference_index.isin(selected_by_fov[fid])
            if keep.sum() != len(selected_by_fov[fid]):
                raise ValueError("Sampled anchor not found in reference feature shard")
        reference_selected = reference[keep]
        reference_rows = np.arange(offset, offset + len(reference_selected))
        for si, source in enumerate(sources):
            if si == 0:
                index, matrix, images = reference_index, reference, reference_images
            else:
                index, matrix, images = _read_cell_profiles(
                    root, manifest.loc[(fid, source["source_id"])], source, spec, fid
                )
            if not index.equals(reference_index) or not np.array_equal(images, reference_images):
                raise ValueError(f"Unmatched nucleus/crop/reference image: {fid}")
            index, matrix = index[keep], matrix[keep]
            end = offset + len(matrix)
            x[offset:end] = matrix
            frame = index.to_frame(index=False)
            frame["row_index"] = np.arange(offset, end)
            frame["reference_row"] = reference_rows
            frame["fov_id"] = fid
            for name, value in metadata.loc[fid].items():
                frame[name] = value
            for name in ["source_id", "variant_id", "blur", "sigma_px"]:
                frame[name] = source[name]
            frame["l2"], frame["relative_l2"], frame["cosine_distance"] = distances(
                reference_selected, matrix
            )
            for ci, channel in enumerate(channels):
                sl = slice(ci * dimension, (ci + 1) * dimension)
                frame[f"cosine_{channel}"] = distances(reference_selected[:, sl], matrix[:, sl])[2]
            frames.append(frame)
            offset = end
        if (fi + 1) % 100 == 0 or fi + 1 == len(fovs_to_read):
            print(f"Assembled {fi + 1:,}/{len(fovs_to_read):,} FOVs; {offset:,} profiles", flush=True)
    if offset != n:
        raise ValueError("Final matrix count mismatch")
    x.flush()
    rows = pd.concat(frames, ignore_index=True)
    rows.to_parquet(output / "profiles.parquet", index=False)
    config = {
        "profile_root": str(root.resolve()), "profile_run_id": spec["profile_run_id"],
        "feature_set_id": spec["feature_set_id"], "run_digest": digest(spec),
        "channels": channels, "feature_names_per_channel": spec["profiler"]["feature_names"],
        "sources": sources, "fov_ids": metadata.index.tolist(), "n_profiles": n,
        "selected_fov_ids": list(fovs_to_read),
        "sampling": {
            "cells_per_stratum": cells_per_stratum, "seed": seed, "strata": STRATA,
            "policy": "all anchors" if selected is None else "uniform within strata, capped",
            "available_profiles": int(manifest.n_eligible.sum()),
        },
        "representation": "channel concatenation; mean-centered only; no whitening",
        "packages": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "umap-learn"]},
    }
    (output / "analysis.json").write_text(json.dumps(config, indent=2))
    return rows, x


def fit_pca95(x, output, variance=0.95, batch_size=4096, threads=8):
    """Exact all-row covariance PCA, using stable batch-merged centered moments.

    Memory is O(batch_size * features + features**2), not O(rows * features).
    The saved mean/components implement (X - mean) @ components.T.
    """
    output = Path(output)
    if not 0 < variance <= 1 or len(x) < 2 or batch_size < 1:
        raise ValueError("Invalid PCA settings")
    count = 0
    mean = np.zeros(x.shape[1], dtype=np.float64)
    scatter = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    with threadpool_limits(limits=threads):
        for start in range(0, len(x), batch_size):
            block = np.array(x[start:start + batch_size], dtype=np.float64)
            block_mean = block.mean(axis=0)
            delta = block_mean - mean
            total = count + len(block)
            block -= block_mean
            scatter += block.T @ block + np.outer(delta, delta) * (count * len(block) / total)
            mean += delta * (len(block) / total)
            count = total
        eigenvalues, eigenvectors = np.linalg.eigh(scatter / (count - 1))
        eigenvalues = np.maximum(eigenvalues[::-1], 0)
        if eigenvalues.sum() <= 0:
            raise ValueError("Zero total variance")
        ratio = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(ratio)
        k = min(len(ratio), int(np.searchsorted(cumulative, variance)) + 1)
        components = eigenvectors[:, ::-1][:, :k].T.copy()
        # Fix arbitrary signs for reproducible saved projections.
        signs = np.sign(components[np.arange(k), np.abs(components).argmax(axis=1)])
        components *= signs[:, None]
        scores = np.lib.format.open_memmap(
            output / "pca_scores.npy", mode="w+", dtype="float32", shape=(len(x), k)
        )
        for start in range(0, len(x), batch_size):
            scores[start:start + batch_size] = (x[start:start + batch_size] - mean) @ components.T
        scores.flush()
    np.savez(
        output / "pca_model.npz", mean=mean, components=components,
        explained_variance=eigenvalues[:k], explained_variance_ratio=ratio[:k],
        n_samples=count, target_variance=variance,
    )
    spectrum = pd.DataFrame({
        "component": np.arange(1, len(ratio) + 1), "explained_variance": eigenvalues,
        "explained_variance_ratio": ratio, "cumulative_variance": cumulative,
        "retained": np.arange(len(ratio)) < k,
    })
    spectrum.to_csv(output / "pca_spectrum.csv", index=False)
    print(f"PCA: {k}/{x.shape[1]} components retain {cumulative[k - 1]:.3%} variance")
    return scores, spectrum


def embed_umap(scores, rows, output, *, mode="full", landmark_cells=25000,
               seed=42, n_neighbors=30, min_dist=0.15, batch_size=20000):
    """Fit one joint map. Optional landmarks retain every source for sampled cells.

    In landmark mode every remaining profile is transformed, not discarded.
    Coordinates carry is_umap_fit to distinguish fitted vs projected profiles.
    """
    import umap

    if mode not in {"full", "landmark"} or landmark_cells < 1:
        raise ValueError("Choose full or landmark mode with positive landmark_cells")
    if not np.array_equal(rows.row_index, np.arange(len(scores))):
        raise ValueError("Rows must align with PCA scores")
    output = Path(output)
    fit_mask = np.ones(len(rows), dtype=bool)
    if mode == "landmark":
        refs = rows.loc[rows.sigma_px.eq(0)]
        # Round-robin allocation across biological strata, random within each stratum.
        shuffled = refs.sample(frac=1, random_state=seed).copy()
        strata = ["cell_line", "plate", "seeding_density"]
        shuffled["rank"] = shuffled.groupby(strata, observed=True).cumcount()
        chosen = shuffled.sort_values("rank", kind="stable").head(landmark_cells).row_index
        fit_mask = rows.reference_row.isin(chosen).to_numpy()
    fit_ids = np.flatnonzero(fit_mask)
    if len(fit_ids) <= n_neighbors:
        raise ValueError("UMAP needs more fit profiles than n_neighbors")
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean",
        random_state=seed, transform_seed=seed, n_jobs=1, low_memory=True,
    )
    coords = np.empty((len(rows), 2), dtype=np.float32)
    print(f"UMAP {mode}: fitting {len(fit_ids):,}; embedding all {len(rows):,} profiles", flush=True)
    coords[fit_ids] = reducer.fit_transform(np.asarray(scores[fit_ids]))
    pending = np.flatnonzero(~fit_mask)
    for start in range(0, len(pending), batch_size):
        ids = pending[start:start + batch_size]
        coords[ids] = reducer.transform(np.asarray(scores[ids]))
    if not np.isfinite(coords).all():
        raise ValueError("Nonfinite UMAP coordinates")
    joblib.dump(reducer, output / "umap_model.joblib")
    result = rows.copy()
    result[["UMAP1", "UMAP2"]] = coords
    result["is_umap_fit"] = fit_mask
    pca_l2 = np.empty(len(rows))
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        ref_ids = rows.reference_row.iloc[start:stop].to_numpy()
        pca_l2[start:stop] = np.linalg.norm(scores[start:stop] - scores[ref_ids], axis=1)
    result["pca_l2"] = pca_l2
    result.to_parquet(output / "embedding_and_drift.parquet", index=False)
    config = json.loads((output / "analysis.json").read_text())
    config["umap"] = {
        "mode": mode, "n_fit": len(fit_ids), "landmark_cells": landmark_cells,
        "seed": seed, "n_neighbors": n_neighbors, "min_dist": min_dist,
        "metric": "euclidean", "transform_batch_size": batch_size,
    }
    with np.load(output / "pca_model.npz") as pca:
        config["pca"] = {
            "n_components": scores.shape[1],
            "target_variance": float(pca["target_variance"]),
            "retained_variance": float(pca["explained_variance_ratio"].sum()),
        }
    (output / "analysis.json").write_text(json.dumps(config, indent=2))
    (output / "_SUCCESS").write_text(digest(config))
    return result
