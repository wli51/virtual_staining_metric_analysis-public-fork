"""Streaming feature-table writer; separate from the image dataset writer."""
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def feature_schema(dimension):
    return pa.schema([
        ('profile_run_id', pa.string()), ('feature_set_id', pa.string()),
        ('source_id', pa.string()), ('dataset_id', pa.string()), ('variant_id', pa.string()),
        ('fov_id', pa.string()), ('mask_set_id', pa.string()),
        ('object_id', pa.string()), ('compartment', pa.string()), ('crop_id', pa.string()),
        ('channels', pa.list_(pa.string())), ('channel', pa.string()),
        ('image_ids', pa.list_(pa.string())), ('reference_image_ids', pa.list_(pa.string())),
        ('image_id', pa.string()), ('reference_image_id', pa.string()),
        ('features', pa.list_(pa.float32(), dimension)),
    ])


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


@contextmanager
def run_lock(output, spec):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.profile.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(fd, str(os.getpid()).encode())
    finally:
        os.close(fd)
    try:
        config = output / 'run.json'
        if config.exists():
            if json.loads(config.read_text()) != spec:
                raise ValueError('Profiling output has different inputs/configuration; use a new directory')
        else:
            if [p for p in output.iterdir() if p != lock]:
                raise ValueError('Nonempty output without run.json')
            atomic_json(config, spec)
        (output / '_SUCCESS').unlink(missing_ok=True)
        yield
    finally:
        lock.unlink()


def feature_table(observations, values, context, *, common, schema, description):
    """Validate plugin identities and attach authoritative image provenance."""
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(observations), schema.field('features').type.list_size):
        raise ValueError('Profiler output shape does not match declared feature names')
    if np.isinf(values).any() or (not description.get('allow_nan', False) and np.isnan(values).any()):
        raise ValueError('Nonfinite profiler output')
    objects = context.objects.set_index('object_id')
    anchors = context.anchors.loc[context.anchors.eligible.eq(True)].set_index('crop_id')
    images = context.image_rows.set_index('channel')
    records = []
    for obs in observations.to_dict('records'):
        oid, compartment, crop_id = obs['object_id'], obs['compartment'], obs['crop_id']
        if oid not in objects.index or objects.loc[oid, 'compartment'] != compartment:
            raise ValueError('Profiler returned an unknown reference object/compartment')
        if crop_id is not None:
            if crop_id not in anchors.index or anchors.loc[crop_id, 'object_id'] != oid:
                raise ValueError('Profiler returned an invalid crop attribution')
        elif description['input_scope'] == 'patch':
            raise ValueError('Patch features require crop identity')
        channels = obs['channels']
        if not isinstance(channels, (list, tuple)) or (not channels and not description.get('allows_channel_free', False)) or len(channels) != len(set(channels)):
            raise ValueError('Each observation requires a distinct channel list')
        if set(channels) - set(images.index):
            raise ValueError('Profiler returned unknown channels')
        rows = images.loc[list(channels)]
        image_ids, reference_ids = rows.image_id.tolist(), rows.reference_image_id.tolist()
        records.append({**common, 'object_id': oid, 'compartment': compartment, 'crop_id': crop_id,
                        'channels': list(channels), 'channel': channels[0] if len(channels) == 1 else None,
                        'image_ids': image_ids, 'reference_image_ids': reference_ids,
                        'image_id': image_ids[0] if len(channels) == 1 else None,
                        'reference_image_id': reference_ids[0] if len(channels) == 1 else None})
    metadata_schema = pa.schema(list(schema)[:-1])
    table = pa.Table.from_pylist(records, schema=metadata_schema)
    vectors = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), type=pa.float32()), values.shape[1])
    return table.append_column(schema.field('features'), vectors)


def merge_parquet(paths, destination):
    schema = pa.unify_schemas([pq.read_schema(p) for p in paths], promote_options='permissive')
    destination = Path(destination)
    tmp = destination.with_name(destination.name + '.tmp')
    with pq.ParquetWriter(tmp, schema, compression='zstd') as writer:
        for path in paths:
            writer.write_table(pq.read_table(path).cast(schema))
    tmp.replace(destination)
