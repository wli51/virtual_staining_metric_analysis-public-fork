"""Structural-aware CP auditing and reference-fitted, paired embedding inputs.

``prepare_cp`` reads *every* shard once, auditing all applicable measurements and
assembling a smaller source-major matrix for each compartment. Sampling uses the
already-profiled eligible anchors, not the original collection's object universe.
``clean_cp_compartment`` writes ``joint_profiles.npy`` and ``profiles.parquet``
for :mod:`utils.profiling.embedding`'s fit_pca95/embed_umap. PCA/UMAP themselves
are joint fits; only feature selection, imputation and scaling are reference fits.

Cleaning's on-disk input contract is raw_applicable.npy (rows x expanded features),
feature_catalog.parquet (column_index, name, scope, family, feature_index, context),
raw_profiles.parquet (row_index, reference_row, fov_id, compartment, object_id,
source_id, sigma_px, plus biological metadata), and analysis.json (sources).
NaN and infinity are counted separately in audits, both invalid for processing.
Raw arrays are never modified. No missingness indicators or clipping are embedded.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

from utils.fov_ingestion.collection import MicroscopyCollection
from utils.fov_ingestion.writer import digest
from utils.profiling.embedding import distances
from utils.profiling.writer import atomic_json

STRATA = ["cell_line", "seeding_density", "plate"]
BIOLOGY = ["cell_line", "plate", "well", "seeding_density"]
IDENTITY = ["fov_id", "compartment", "object_id"]
SCOPES = {"geometry": 0, "single_channel": 1, "channel_pair": 2}


def _file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _hash_files(folder, names):
    return {name: _file_digest(Path(folder) / name) for name in names}


def _inspect(root):
    spec = json.loads((root / "run.json").read_text())
    if (root / "_SUCCESS").read_text().strip() != digest(spec):
        raise ValueError("Profiling completion marker does not match run.json")
    if spec["profiler"]["name"] != "cp_measure":
        raise ValueError("Expected a CP run")
    fovs, channels = spec["fov_ids"], spec["channels"]
    sources = spec["sources"]
    if (
        not fovs or len(set(fovs)) != len(fovs)
        or not channels or len(set(channels)) != len(channels)
        or not sources or len({s["source_id"] for s in sources}) != len(sources)
        or len({s["fingerprint"] for s in sources}) != len(sources)
    ):
        raise ValueError("Empty or duplicate run identities")
    manifest = pd.read_parquet(root / "manifest.parquet")
    manifest = manifest.set_index(["fov_id", "source_id"], verify_integrity=True)
    expected = pd.MultiIndex.from_product([fovs, [s["source_id"] for s in sources]])
    if set(manifest.index) != set(expected) or manifest.feature_uri.duplicated().any():
        raise ValueError("Manifest must uniquely cover exactly every source/FOV")
    if (manifest[["n_rows", "n_eligible", "n_nuclei"]] < 0).any().any():
        raise ValueError("Negative manifest counts")
    catalog = pd.read_parquet(root / "feature_catalog.parquet").sort_values("feature_index")
    declared = pd.DataFrame(spec["profiler"]["feature_catalog"]).sort_values("feature_index")
    pd.testing.assert_frame_equal(catalog.reset_index(drop=True), declared.reset_index(drop=True),
                                  check_dtype=False)
    if (
        not np.array_equal(catalog.feature_index, np.arange(len(catalog)))
        or catalog.feature_name.duplicated().any() or set(catalog.scope) - set(SCOPES)
        or catalog.feature_name.tolist() != spec["profiler"]["feature_names"]
    ):
        raise ValueError("Invalid CP feature catalog")
    compartments = spec["profiler"]["settings"]["compartments"]
    if not compartments or len(set(compartments)) != len(compartments) or (
        set(compartments) - {"nuclei", "cells"}
    ):
        raise ValueError("Invalid compartments")
    collections, enriched = {}, []
    metadata = None
    for i, source in enumerate(sources):
        collection = MicroscopyCollection(root / source["uri"])
        if collection.fingerprint != source["fingerprint"]:
            raise ValueError("Source fingerprint changed")
        if collection.spec["dataset_id"] != source["dataset_id"]:
            raise ValueError("Source dataset provenance mismatch")
        current = collection.metadata().set_index("fov_id", verify_integrity=True)
        current = current.loc[fovs, BIOLOGY].copy()
        current["seeding_density"] = pd.to_numeric(current.seeding_density, errors="raise")
        if current.isna().any().any():
            raise ValueError("Missing biological metadata")
        if i == 0:
            if source["variant_id"] != "reference" or (
                collection.fingerprint != spec["reference_fingerprint"]
            ):
                raise ValueError("First source must be the reference")
            metadata, sigma = current, 0.0
        else:
            pd.testing.assert_frame_equal(metadata, current, check_dtype=False)
            transform = collection.spec["transform"]
            if (
                transform["name"] != "gaussian_blur"
                or transform["mask_policy"] != "fixed_reference"
                or collection.spec["reference_fingerprint"] != spec["reference_fingerprint"]
            ):
                raise ValueError("Expected fixed-reference Gaussian blur")
            sigma = float(transform["parameters"]["sigma_px"])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Invalid variant sigma")
        collections[source["source_id"]] = collection
        enriched.append({**source, "sigma_px": sigma})
    enriched.sort(key=lambda s: (s["sigma_px"], s["source_id"]))
    for i, source in enumerate(enriched):
        source["blur"] = "Reference" if i == 0 else f"Blur {i} (σ={source['sigma_px']:.2f})"
    return spec, enriched, metadata, manifest, catalog, collections


def _expanded_catalog(catalog, channels):
    contexts = [("geometry", ())]
    contexts += [("single_channel", (c,)) for c in channels]
    contexts += [("channel_pair", pair) for pair in combinations(channels, 2)]
    records, blocks = [], {}
    for scope, context in contexts:
        subset = catalog.loc[catalog.scope.eq(scope)]
        if subset.empty:
            continue
        start = len(records)
        for feature in subset.to_dict("records"):
            records.append({**feature, "column_index": len(records),
                            "context": "|".join(context) if context else "geometry",
                            "channels": list(context),
                            "name": "|".join([*context, feature["feature_name"]])
                            if context else "geometry|" + feature["feature_name"]})
        blocks[context] = (scope, subset.feature_index.to_numpy(), slice(start, len(records)))
    expanded = pd.DataFrame(records)
    if expanded.empty or expanded.name.duplicated().any():
        raise ValueError("Empty or ambiguous expanded feature names")
    return expanded, blocks


def _cohorts(anchors, metadata, compartments, cap, seed):
    if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
        raise ValueError("cells_per_stratum must be a positive integer")
    if anchors.duplicated(["fov_id", "object_id"]).any():
        raise ValueError("Duplicate nuclear anchors")
    if set(anchors.fov_id) - set(metadata.index):
        raise ValueError("Anchor FOV outside the run")
    eligible = anchors.loc[anchors.eligible.eq(True)].copy()
    if "selected" in eligible and not eligible.selected.eq(True).all():
        raise ValueError("Eligible anchors must have been selected for profiling")
    all_objects, selected, coverage = {}, {}, []
    strata = metadata[STRATA].drop_duplicates().sort_values(STRATA)
    for compartment in compartments:
        if compartment == "nuclei":
            objects = eligible[["fov_id", "object_id"]].copy()
        else:
            objects = eligible[["fov_id", "matched_cell_id"]].dropna().rename(
                columns={"matched_cell_id": "object_id"}
            ).drop_duplicates()
        objects["compartment"] = compartment
        objects = objects.merge(metadata.reset_index(), on="fov_id", validate="many_to_one")
        if objects.object_id.isna().any():
            raise ValueError("Missing object identity")
        objects = objects.sort_values(IDENTITY).reset_index(drop=True)
        all_objects[compartment] = objects
        chosen = []
        groups = objects.groupby(STRATA, observed=True, sort=True)
        for values in strata.itertuples(index=False, name=None):
            group = groups.get_group(values) if values in groups.groups else objects.iloc[:0]
            # Independent per-stratum RNG: stable even when another stratum is added.
            token = digest([int(seed), compartment, list(values)])
            rng = np.random.default_rng(int(token[:16], 16))
            keep = group.iloc[rng.choice(len(group), min(cap, len(group)), replace=False)]
            chosen.append(keep)
            coverage.append(dict(zip(STRATA, values)) | {
                "compartment": compartment, "available_objects": len(group),
                "selected_objects": len(keep), "available_fovs": group.fov_id.nunique(),
                "selected_fovs": keep.fov_id.nunique(), "empty": len(group) == 0,
                "rare": len(group) < cap,
                "sampling_fraction": len(keep) / len(group) if len(group) else np.nan,
            })
        selected[compartment] = pd.concat(chosen, ignore_index=True).sort_values(
            IDENTITY
        ).reset_index(drop=True)
    return all_objects, selected, pd.DataFrame(coverage)


def _warnings(root, output, spec):
    """Preserve full records, including failed/retried attempts, without object claims."""
    keys = ["message", "category", "filename", "lineno", "source_id", "fov_id", "attempt_id"]
    counts, records = Counter(), Counter()
    source_ids, fovs = {s["source_id"] for s in spec["sources"]}, set(spec["fov_ids"])
    with (output / "warning_records.jsonl").open("w") as target:
        for path in sorted((root / "warnings").glob("*.jsonl")):
            with path.open() as stream:
                for line in stream:
                    event = json.loads(line)
                    if event["source_id"] not in source_ids or event["fov_id"] not in fovs:
                        raise ValueError("Warning provenance outside the run")
                    if event.get("profile_run_id", spec["profile_run_id"]) != spec["profile_run_id"]:
                        raise ValueError("Warning belongs to another run")
                    count = event["count"]
                    if not isinstance(count, int) or count < 1:
                        raise ValueError("Invalid warning emission count")
                    key = tuple(event.get(k) for k in keys)
                    counts[key] += count
                    records[key] += 1
                    target.write(json.dumps({**event, "warning_log": str(path.relative_to(root))}) + "\n")
    frame = pd.DataFrame([
        dict(zip(keys, key)) | {"count": count, "n_records": records[key]}
        for key, count in counts.items()
    ], columns=keys + ["count", "n_records"])
    frame.to_csv(output / "warning_summary.csv", index=False)
    return {"n_records": sum(records.values()), "emissions": sum(counts.values()),
            "interpretation": "warning emissions, not affected objects; includes all attempts"}


def prepare_cp(root, output, cells_per_stratum=200, seed=42):
    """Audit all raw data; write paired sampled matrices; return (spec, sources, reports).

    ``output`` is normally ``Path(root) / 'analysis_embedding'`` and must not exist.
    Each compartment has an independent capped uniform sample. Every sampled
    identity is retained in every source. Empty/rare observed biological strata
    are reported, not invented or silently replaced by other conditions.
    A failed validation leaves diagnostic partial outputs but no _PREPARED marker.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    spec, sources, metadata, manifest, catalog, collections = _inspect(root)
    anchors = pd.read_parquet(root / "anchors.parquet")
    compartments = spec["profiler"]["settings"]["compartments"]
    all_objects, selected, coverage = _cohorts(
        anchors, metadata, compartments, cells_per_stratum, seed
    )
    features, contexts = _expanded_catalog(catalog, spec["channels"])
    for fid in spec["fov_ids"]:
        group = anchors.loc[anchors.fov_id.eq(fid)]
        units = manifest.xs(fid, level="fov_id")
        expected_rows = sum(int(objects.fov_id.eq(fid).sum()) for objects in all_objects.values())
        expected_rows *= len(contexts)
        if not (units.n_eligible.eq(int(group.eligible.sum())).all()
                and units.n_nuclei.eq(len(group)).all() and units.n_rows.eq(expected_rows).all()):
            raise ValueError(f"Manifest/anchor count mismatch: {fid}")
    output.mkdir(parents=True, exist_ok=False)
    coverage.to_csv(output / "sampling_coverage.csv", index=False)
    warnings_report = _warnings(root, output, spec)
    source_ids = [s["source_id"] for s in sources]
    source_rank = {sid: i for i, sid in enumerate(source_ids)}
    arrays, seen, positions, raw_rows, folders = {}, {}, {}, {}, {}
    config = {
        "profile_root": str(root), "profile_run_id": spec["profile_run_id"],
        "feature_set_id": spec["feature_set_id"], "run_digest": digest(spec),
        "sources": sources, "channels": spec["channels"], "fov_ids": spec["fov_ids"],
        "sampling": {"cells_per_stratum": cells_per_stratum, "seed": seed, "strata": STRATA,
                     "policy": "uniform within profiled eligible identities per compartment"},
        "input_digests": _hash_files(root, ["run.json", "manifest.parquet", "anchors.parquet",
                                            "feature_catalog.parquet"]),
        "shard_manifest_digest": digest(manifest.reset_index().to_json()),
        "preprocessing": {"status": "raw; geometry audit-only", "fit_population": "reference"},
        "target": "paired compartment-specific CP object profiles",
    }
    for compartment in compartments:
        folder = output / "compartments" / compartment
        folder.mkdir(parents=True)
        folders[compartment] = folder
        objects = selected[compartment]
        n = len(objects)
        arrays[compartment] = np.lib.format.open_memmap(
            folder / "raw_applicable.npy", mode="w+", dtype="float32",
            shape=(len(sources) * n, len(features)),
        )
        arrays[compartment][:] = np.nan
        seen[compartment] = np.zeros((len(sources) * n, len(contexts)), dtype=bool)
        positions[compartment] = pd.MultiIndex.from_frame(objects[IDENTITY])
        frames = []
        for si, source in enumerate(sources):
            frame = objects.copy()
            frame["crop_id"] = None  # Full-object profiles have no crop identity.
            frame["row_index"] = si * n + np.arange(n)
            frame["reference_row"] = np.arange(n)
            for name in ["source_id", "variant_id", "sigma_px", "blur"]:
                frame[name] = source[name]
            frames.append(frame)
        raw_rows[compartment] = pd.concat(frames, ignore_index=True)
        raw_rows[compartment].to_parquet(folder / "raw_profiles.parquet", index=False)
        features.to_parquet(folder / "feature_catalog.parquet", index=False)
    # O(sources * compartments * expanded features), independent of raw row count.
    audit = { (sid, comp): np.zeros((len(features), 3), dtype=np.int64)
             for sid in source_ids for comp in compartments }
    row_reports, structural, provenance = [], [], []
    reference = collections[source_ids[0]]
    reference_masks = reference.fovs.set_index("fov_id").mask_set_id
    # Avoid rescanning whole image registries for each of the thousands of shards.
    registries = {
        sid: {fid: group.set_index("channel", verify_integrity=True).loc[spec["channels"]]
              for fid, group in collection.images.loc[
                  collection.images.role.eq("intensity")
                  & collection.images.fov_id.isin(spec["fov_ids"])
              ].groupby("fov_id", sort=False)}
        for sid, collection in collections.items()
    }
    expected_by_fov = {
        comp: {fid: set(group.object_id) for fid, group in objects.groupby("fov_id")}
        for comp, objects in all_objects.items()
    }
    for mi, ((fid, sid), record) in enumerate(manifest.iterrows()):
        source = sources[source_rank[sid]]
        path = (root / record.feature_uri).resolve()
        if not path.is_relative_to(root / "shards"):
            raise ValueError("Shard path is outside the profiling run")
        done = json.loads((path.parent / "complete.json").read_text())
        for key in ["fingerprint", "source_id", "fov_id", "feature_uri", "n_rows", "n_eligible"]:
            value = record[key] if key in record.index else {"source_id": sid, "fov_id": fid}[key]
            if done[key] != value:
                raise ValueError(f"Shard commit/manifest mismatch: {key}")
        # A shard is a bounded FOV/source unit; never concatenate the full dataset.
        table = pq.read_table(path).to_pandas()
        if len(table) != record.n_rows:
            raise ValueError("Shard row count differs from manifest")
        expected_provenance = {
            "profile_run_id": spec["profile_run_id"], "feature_set_id": spec["feature_set_id"],
            "fov_id": fid, "source_id": sid, "dataset_id": source["dataset_id"],
            "variant_id": source["variant_id"],
            "mask_set_id": reference_masks.loc[fid],
        }
        for name, value in expected_provenance.items():
            if not table[name].eq(value).all():
                raise ValueError(f"Shard provenance mismatch: {name}")
        if table.crop_id.notna().any() or table.object_id.isna().any():
            raise ValueError("CP needs full-object, non-null identities and null crop_id")
        table["context_key"] = table.channels.map(tuple)
        if table.duplicated(["compartment", "object_id", "context_key"]).any():
            raise ValueError("Duplicate object/context observation")
        if set(table.compartment) - set(compartments) or set(table.context_key) - set(contexts):
            raise ValueError("Unexpected compartment/channel context")
        registry = registries[sid][fid]
        ref_registry = registries[source_ids[0]][fid]
        if registry.reference_image_id.tolist() != ref_registry.image_id.tolist():
            raise ValueError("Source/reference image registry mismatch")
        actual_groups = {(comp, ctx): group for (comp, ctx), group in table.groupby(
            ["compartment", "context_key"], sort=False
        )}
        for compartment in compartments:
            expected_ids = expected_by_fov[compartment].get(fid, set())
            for ci, (context, (scope, indices, columns)) in enumerate(contexts.items()):
                group = actual_groups.get((compartment, context), table.iloc[:0])
                group = group.set_index("object_id", verify_integrity=True).sort_index()
                if set(group.index) != expected_ids:
                    raise ValueError(f"Absent/extra object context: {fid}/{sid}/{compartment}/{context}")
                if not len(group):
                    continue
                for field, expected_ids_list in [
                    ("image_ids", registry.loc[list(context)].image_id.tolist()),
                    ("reference_image_ids", ref_registry.loc[list(context)].image_id.tolist()),
                ]:
                    if not group[field].map(lambda v, ids=expected_ids_list: list(v) == ids).all():
                        raise ValueError(f"Image provenance mismatch: {field}")
                for field, expected_value in [
                    ("channel", context[0] if len(context) == 1 else None),
                    ("image_id", registry.loc[context[0], "image_id"] if len(context) == 1 else None),
                    ("reference_image_id", ref_registry.loc[context[0], "image_id"]
                     if len(context) == 1 else None),
                ]:
                    valid = group[field].isna() if expected_value is None else group[field].eq(expected_value)
                    if not valid.all():
                        raise ValueError(f"Scalar context attribution mismatch: {field}")
                values = np.stack(group.features)
                if values.shape != (len(group), len(catalog)):
                    raise ValueError("Invalid raw feature vector shape")
                outside = np.ones(len(catalog), dtype=bool)
                outside[indices] = False
                structural.append({"fov_id": fid, "source_id": sid,
                                   "compartment": compartment, "scope": scope,
                                   "context": "|".join(context) or "geometry",
                                   "n_structural": int(len(group) * outside.sum()),
                                   "n_not_nan": int((~np.isnan(values[:, outside])).sum())})
                applicable = values[:, indices]
                missing, infinite = np.isnan(applicable), np.isinf(applicable)
                audit[sid, compartment][columns] += np.column_stack([
                    np.full(len(indices), len(group)), missing.sum(axis=0), infinite.sum(axis=0)
                ])
                row_reports.append({"fov_id": fid, "source_id": sid,
                                    "compartment": compartment, "scope": scope,
                                    **metadata.loc[fid].to_dict(), "n_rows": len(group),
                                    "n_values": int(applicable.size),
                                    "n_missing": int(missing.sum()), "n_inf": int(infinite.sum()),
                                    "n_rows_nonfinite": int((missing | infinite).any(axis=1).sum()),
                                    "sum_row_missing_fraction": float(missing.mean(axis=1).sum()),
                                    "sum_row_nonfinite_fraction": float((missing | infinite).mean(axis=1).sum())})
                identities = pd.MultiIndex.from_arrays([
                    np.repeat(fid, len(group)), np.repeat(compartment, len(group)), group.index
                ])
                pos = positions[compartment].get_indexer(identities)
                keep = pos >= 0
                destination = source_rank[sid] * len(selected[compartment]) + pos[keep]
                arrays[compartment][destination, columns] = applicable[keep]
                if seen[compartment][destination, ci].any():
                    raise ValueError("Sampled context written more than once")
                seen[compartment][destination, ci] = True
        provenance.append({"fov_id": fid, "source_id": sid, "n_rows": len(table),
                           "n_objects": table[["compartment", "object_id"]].drop_duplicates().shape[0]})
        if (mi + 1) % 100 == 0 or mi + 1 == len(manifest):
            print(f"CP audit/assembly: {mi + 1:,}/{len(manifest):,} shards", flush=True)
    audit_frames = []
    for (sid, compartment), counts in audit.items():
        frame = features.drop(columns="channels").copy()
        frame[["n_values", "n_missing", "n_inf"]] = counts
        frame["source_id"], frame["compartment"] = sid, compartment
        frame["missing_rate"] = frame.n_missing / frame.n_values.replace(0, np.nan)
        frame["nonfinite_rate"] = (frame.n_missing + frame.n_inf) / frame.n_values.replace(0, np.nan)
        audit_frames.append(frame)
    pd.concat(audit_frames, ignore_index=True).to_csv(output / "feature_missingness.csv", index=False)
    rows = pd.DataFrame(row_reports)
    measures = ["n_rows", "n_values", "n_missing", "n_inf", "n_rows_nonfinite",
                "sum_row_missing_fraction", "sum_row_nonfinite_fraction"]
    group_keys = ["compartment", "scope", "source_id", "fov_id", *BIOLOGY]
    rows = rows.groupby(group_keys, observed=True, dropna=False)[measures].sum().reset_index()
    rows.to_csv(output / "row_missingness_by_fov.csv", index=False)
    rows.groupby(["compartment", "scope", "source_id", *BIOLOGY], observed=True)[measures].sum().to_csv(
        output / "row_missingness_by_condition.csv"
    )
    structural_frame = pd.DataFrame(structural)
    structural_frame.to_csv(output / "structural_diagnostics.csv", index=False)
    pd.DataFrame(provenance).to_csv(output / "provenance_counts.csv", index=False)
    if structural_frame.n_not_nan.sum():
        raise ValueError("Non-NaN values outside catalog scope; see structural_diagnostics.csv")
    reports = {"sampling": coverage, "warnings": warnings_report, "output": str(output),
               "compartments": {}}
    for compartment, folder in folders.items():
        arrays[compartment].flush()
        if not seen[compartment].all():
            raise ValueError(f"Missing sampled object/context/source: {compartment}")
        raw_rows[compartment].to_parquet(folder / "profiles.parquet", index=False)
        local = {**config, "compartment": compartment,
                 "n_profiles": len(raw_rows[compartment]), "n_features": len(features),
                 "selected_fov_ids": selected[compartment].fov_id.unique().tolist(),
                 "raw_shape": list(arrays[compartment].shape),
                 "raw_digests": _hash_files(folder, ["raw_applicable.npy", "raw_profiles.parquet",
                                                     "feature_catalog.parquet"])}
        atomic_json(folder / "analysis.json", local)
        (folder / "_PREPARED").write_text(digest(local))
        reports["compartments"][compartment] = {"folder": str(folder), "n_profiles": local["n_profiles"]}
    atomic_json(output / "analysis.json", {**config, "warnings": warnings_report,
                                          "compartments": reports["compartments"]})
    (output / "_PREPARED").write_text(digest(json.loads((output / "analysis.json").read_text())))
    return spec, sources, reports


def clean_cp_compartment(folder, max_feature_missing=0.2, max_object_missing=0.3):
    """Reference-only feature filtering/median/scaler with all-source pairwise QC.

    Feature missingness includes infinity. QC uses the initially retained feature
    set, once, before post-QC zero-reference-variance removal (no iterative QC).
    Pair selection deliberately uses variants; model statistics never do. Returned
    ``(rows, joint_memmap)`` can be passed directly to PCA/UMAP helpers. Distances
    are in the standardized, median-imputed space, not original measurements.
    """
    if not 0 <= max_feature_missing <= 1 or not 0 <= max_object_missing <= 1:
        raise ValueError("Missingness thresholds must be in [0,1]")
    folder = Path(folder)
    config = json.loads((folder / "analysis.json").read_text())
    if (folder / "_CLEANED").exists() or (folder / "joint_profiles.npy").exists():
        raise FileExistsError("Cleaning outputs already exist; use a fresh prepared compartment")
    if (folder / "_PREPARED").exists() and (
        (folder / "_PREPARED").read_text().strip() != digest(config)
    ):
        raise ValueError("Prepared configuration changed")
    inputs = ["raw_applicable.npy", "raw_profiles.parquet", "feature_catalog.parquet"]
    hashes = _hash_files(folder, inputs)
    if "raw_digests" in config and hashes != config["raw_digests"]:
        raise ValueError("Prepared cleaning inputs changed")
    raw = np.load(folder / "raw_applicable.npy", mmap_mode="r")
    features = pd.read_parquet(folder / "feature_catalog.parquet").sort_values("column_index")
    rows = pd.read_parquet(folder / "raw_profiles.parquet")
    if raw.shape != (len(rows), len(features)) or not np.array_equal(
        features.column_index, np.arange(len(features))
    ) or not np.array_equal(rows.row_index, np.arange(len(rows))):
        raise ValueError("Raw rows/columns do not align with metadata")
    if rows.duplicated([*IDENTITY, "source_id"]).any() or rows[IDENTITY].isna().any().any():
        raise ValueError("Duplicate or missing object identity")
    source_ids = [s["source_id"] for s in config["sources"]]
    if len(set(source_ids)) != len(source_ids) or set(rows.source_id) != set(source_ids):
        raise ValueError("Cleaning source coverage mismatch")
    refs = rows.loc[rows.source_id.eq(source_ids[0])]
    ref_ids = refs.row_index.to_numpy(dtype=np.int64)
    if len(refs) < 2 or not refs.sigma_px.eq(0).all() or not np.array_equal(
        refs.reference_row, ref_ids
    ):
        raise ValueError("Need at least two self-paired reference objects")
    ref_keys = pd.MultiIndex.from_frame(refs[IDENTITY])
    for sid in source_ids:
        group = rows.loc[rows.source_id.eq(sid)]
        positions = ref_keys.get_indexer(pd.MultiIndex.from_frame(group[IDENTITY]))
        if len(group) != len(refs) or (positions < 0).any() or not np.array_equal(
            group.reference_row, ref_ids[positions]
        ):
            raise ValueError("Missing or incorrectly paired source object")
    reference = np.array(raw[ref_ids], dtype=np.float64)
    reference[~np.isfinite(reference)] = np.nan
    missing_rate = np.isnan(reference).mean(axis=0)
    # Compute only finite columns to avoid all-NaN reduction warnings.
    observed = np.sum(np.isfinite(reference), axis=0)
    variance = np.zeros(len(features))
    valid_columns = observed > 0
    variance[valid_columns] = np.nanvar(reference[:, valid_columns], axis=0)
    retain = (~features.scope.eq("geometry").to_numpy() & (missing_rate <= max_feature_missing)
              & (observed > 0) & np.isfinite(variance) & (variance > 0))
    decisions = features.copy()
    decisions["reference_missing_rate"] = missing_rate
    decisions["reference_finite_variance"] = variance
    decisions["retained_before_pair_qc"] = retain
    decisions["reason"] = np.select(
        [features.scope.eq("geometry"), missing_rate > max_feature_missing, ~valid_columns,
         ~np.isfinite(variance) | (variance <= 0)],
        ["geometry_audit_only", "reference_missingness", "no_finite_reference", "zero_reference_variance"],
        default="retained",
    )
    if not retain.any():
        decisions.to_csv(folder / "feature_decisions.csv", index=False)
        raise ValueError("No non-geometry reference-variable features survive")
    selected_columns = np.flatnonzero(retain)
    # The balanced sample is modest; raw data stays read-only and disk-backed.
    values = np.array(raw[:, selected_columns], dtype=np.float64)
    invalid = ~np.isfinite(values)
    values[invalid] = np.nan
    qc = rows.copy()
    qc["n_missing"] = np.isnan(np.asarray(raw[:, selected_columns])).sum(axis=1)
    qc["n_inf"] = np.isinf(np.asarray(raw[:, selected_columns])).sum(axis=1)
    qc["n_nonfinite"] = invalid.sum(axis=1)
    qc["n_features"] = len(selected_columns)
    qc["nonfinite_fraction"] = invalid.mean(axis=1)
    qc["fails_object_qc"] = qc.nonfinite_fraction.gt(max_object_missing)
    excluded = set(qc.loc[qc.fails_object_qc, "reference_row"])
    qc["pair_excluded"] = qc.reference_row.isin(excluded)
    qc.to_parquet(folder / "row_missingness.parquet", index=False)
    exclusions = qc.loc[qc.pair_excluded].copy()
    exclusions["reason"] = np.where(exclusions.fails_object_qc, "row_missingness", "paired_source_failed")
    exclusions.to_csv(folder / "pair_exclusions.csv", index=False)
    rates = qc.groupby("source_id", sort=False).agg(
        n_profiles=("row_index", "size"), n_missing=("n_missing", "sum"),
        n_inf=("n_inf", "sum"), n_nonfinite=("n_nonfinite", "sum"),
        n_failed_rows=("fails_object_qc", "sum"), n_excluded_rows=("pair_excluded", "sum"),
    )
    rates["n_values"] = rates.n_profiles * len(selected_columns)
    rates["nonfinite_rate"] = rates.n_nonfinite / rates.n_values
    rates.to_csv(folder / "source_missingness_after_feature_retention.csv")
    keep = ~qc.pair_excluded.to_numpy()
    kept_refs = ref_ids[~refs.reference_row.isin(excluded).to_numpy()]
    if len(kept_refs) < 2:
        raise ValueError("Fewer than two reference objects survive paired QC")
    post_reference = values[kept_refs]
    post_observed = np.isfinite(post_reference).sum(axis=0)
    post_variance = np.zeros(len(selected_columns))
    post_variance[post_observed > 0] = np.nanvar(post_reference[:, post_observed > 0], axis=0)
    post_keep = (post_observed > 0) & np.isfinite(post_variance) & (post_variance > 0)
    decisions.loc[selected_columns, "post_qc_reference_finite_variance"] = post_variance
    decisions.loc[selected_columns[~post_keep], "reason"] = "zero_reference_variance_after_pair_qc"
    retain[selected_columns[~post_keep]] = False
    decisions["retained"] = retain
    decisions.to_csv(folder / "feature_decisions.csv", index=False)
    if not post_keep.any():
        raise ValueError("No reference-variable features after paired QC")
    selected_columns = selected_columns[post_keep]
    medians = np.nanmedian(post_reference[:, post_keep], axis=0)
    ref_imputed = np.where(np.isfinite(post_reference[:, post_keep]), post_reference[:, post_keep], medians)
    scaler = StandardScaler().fit(ref_imputed)
    values = values[keep][:, post_keep]
    imputed_mask = ~np.isfinite(values)
    values = np.where(imputed_mask, medians, values)
    scaled = scaler.transform(values)
    if not np.isfinite(scaled).all():
        raise ValueError("Nonfinite standardized values")
    clean = np.lib.format.open_memmap(
        folder / "joint_profiles.npy", mode="w+", dtype="float64", shape=scaled.shape
    )
    clean[:] = scaled
    clean.flush()
    result = rows.loc[keep].copy().reset_index(drop=True)
    mapping = pd.Series(np.arange(len(result)), index=result.row_index)
    result["raw_row_index"] = result.row_index
    result["reference_row"] = result.reference_row.map(mapping).astype("int64")
    result["row_index"] = np.arange(len(result))
    result["ref_row"] = result.reference_row
    result["n_imputed"] = imputed_mask.sum(axis=1)
    result["n_abs_scaled_gt10"] = (np.abs(scaled) > 10).sum(axis=1)
    result["l2"], result["relative_l2"], result["cosine_distance"] = distances(
        scaled[result.reference_row.to_numpy()], scaled
    )
    result["distance_space"] = "reference-standardized median-imputed CP (no indicators)"
    result.to_parquet(folder / "profiles.parquet", index=False)
    np.savez(folder / "preprocessing_model.npz", feature_mask=retain,
             column_indices=selected_columns, feature_names=features.name.to_numpy(dtype=str),
             retained_feature_names=features.name.iloc[selected_columns].to_numpy(dtype=str),
             reference_medians=medians, mean=scaler.mean_, std=scaler.scale_,
             variance=scaler.var_, n_reference=len(kept_refs))
    summary = result.groupby("source_id", sort=False).agg(
        n_profiles=("row_index", "size"), n_imputed=("n_imputed", "sum"),
        n_abs_scaled_gt10=("n_abs_scaled_gt10", "sum"),
    )
    summary["n_values"] = summary.n_profiles * scaled.shape[1]
    summary["imputed_fraction"] = summary.n_imputed / summary.n_values
    summary["abs_scaled_gt10_fraction"] = summary.n_abs_scaled_gt10 / summary.n_values
    summary.to_csv(folder / "cleaning_by_source.csv")
    config.update({
        "n_profiles": len(result), "n_features": scaled.shape[1],
        "representation": "reference-standardized median-imputed CP; geometry excluded",
        "preprocessing": {
            "max_feature_missing": max_feature_missing, "max_object_missing": max_object_missing,
            "fit_population": "kept reference objects only", "invalid": "NaN or infinity",
            "pair_qc": "any source fails on initially retained features; single pass",
            "n_reference_before": len(refs), "n_reference_kept": len(kept_refs),
            "n_pairs_excluded": len(excluded), "reference_imputed_values": int(
                result.loc[result.source_id.eq(source_ids[0]), "n_imputed"].sum()),
            "variant_imputed_values": int(result.loc[~result.source_id.eq(source_ids[0]), "n_imputed"].sum()),
            "clip": False, "winsorize": False, "missingness_indicators": False,
            "abs_scaled_gt10_fraction": float((np.abs(scaled) > 10).mean()),
            "distance_caveat": "L2, relative L2 and cosine are in standardized imputed space",
        },
        "cleaning_input_digests": hashes,
        "output_digests": _hash_files(folder, ["joint_profiles.npy", "profiles.parquet",
                                              "preprocessing_model.npz", "feature_decisions.csv"]),
    })
    atomic_json(folder / "analysis.json", config)
    (folder / "_CLEANED").write_text(digest(config))
    return result, clean
