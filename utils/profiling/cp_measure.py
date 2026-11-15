"""CellProfiler-style object measurements using the pinned cp-measure API.

Full intensity planes preserve the package's FOV-level operations. Only selected
objects enter its measurement masks; labels are compacted without changing the
canonical object IDs. No segmentation or per-object intensity scaling is done.
"""
import warnings
from functools import partial
from importlib.metadata import version
from itertools import combinations

import numpy as np
import pandas as pd

from .api import FeatureBatch

GEOMETRY = ('sizeshape', 'zernike', 'feret')


def compact_labels(mask, labels):
    """Explicit 1..N relabeling in supplied object order (including gapped IDs)."""
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(np.unique(labels)) or (labels <= 0).any():
        raise ValueError('Expected distinct positive labels')
    # Avoid allocating a lookup indexed by potentially huge original label IDs.
    order = np.argsort(labels)
    sorted_labels = labels[order]
    positions = np.searchsorted(sorted_labels, mask)
    result = np.zeros(mask.shape, dtype=np.int32)
    if len(labels):
        valid = positions < len(labels)
        valid &= sorted_labels[np.minimum(positions, len(labels) - 1)] == mask
        result[valid] = order[positions[valid]] + 1
    if set(np.unique(result)) - {0} != set(range(1, len(labels) + 1)):
        raise ValueError('Selected reference label is absent from mask')
    return result


def _overlap_with_undefined(first, second, mask):
    """Keep upstream overlap semantics, but its Python 0/0 is undefined, not fatal."""
    from cp_measure.core.measurecolocalization import get_correlation_overlap_ind
    from scipy.ndimage import find_objects

    names = ('Correlation_Overlap', 'Correlation_K_1', 'Correlation_K_2')
    result = {name: [] for name in names}
    for label, region in enumerate(find_objects(mask), 1):
        if region is None:
            raise ValueError('Overlap requires contiguous labels')
        try:
            measured = get_correlation_overlap_ind(first[region], second[region], mask[region] == label)
        except ZeroDivisionError:
            # Upstream explicitly converts squared sums to Python float, so
            # np.errstate cannot handle the zero denominator for empty signal.
            measured = dict.fromkeys(names, np.nan)
        for name in names:
            result[name].append(measured[name])
    return result


class CPMeasureProfiler:
    def __init__(self, *, compartments=('nuclei', 'cells'), families=None,
                 texture_scales=(3,), gray_levels=256, include_pairs=True):
        import cp_measure
        from cp_measure.bulk import get_core_measurements, get_correlation_measurements

        if version('cp-measure') != '0.3.0':
            raise ValueError('This adapter is validated with cp-measure==0.3.0')
        cp_measure.set_accelerator(None)
        self.compartments = tuple(compartments)
        if not self.compartments or len(set(self.compartments)) != len(self.compartments) or set(self.compartments) - {'nuclei', 'cells'}:
            raise ValueError('Select nuclei and/or cells compartments')
        scales = tuple(texture_scales)
        if not scales or len(set(scales)) != len(scales) or any(s < 1 for s in scales) or not 2 <= gray_levels <= 256:
            raise ValueError('Invalid texture scales/gray levels')
        core = get_core_measurements(legacy=True, sanitize=False)
        pairs = {**get_correlation_measurements(sanitize=False), 'overlap': _overlap_with_undefined}
        pairs['costes'] = partial(pairs['costes'], scale_max=65535)
        available = set(core) | set(pairs)
        requested = available if families is None else set(families)
        if not requested or requested - available:
            raise ValueError(f'Unknown/empty families; available: {sorted(available)}')
        self.functions = {}
        for family, fn in core.items():
            if family not in requested:
                continue
            scope = 'geometry' if family in GEOMETRY else 'single_channel'
            if family == 'texture':
                for scale in scales:
                    self.functions[f'texture_s{scale}'] = (scope, family, partial(fn, scale=scale, gray_levels=gray_levels))
            else:
                self.functions[family] = (scope, family, fn)
        if include_pairs:
            self.functions.update({k: ('channel_pair', k, f) for k, f in pairs.items() if k in requested})
        if not self.functions:
            raise ValueError('No enabled measurements')
        self.settings = {'compartments': list(self.compartments), 'families': sorted(requested),
                         'texture_scales': list(scales), 'gray_levels': gray_levels,
                         'include_pairs': include_pairs}
        # Discover the exact scalar schema from the installed, pinned functions.
        mask = np.zeros((64, 64), np.int32)
        mask[8:26, 9:27] = 1
        mask[35:55, 32:55] = 2
        pixels = np.random.default_rng(42).uniform(.05, .8, mask.shape)
        self.catalog, self.keys = [], {}
        for key, (scope, family, fn) in self.functions.items():
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                result = fn(pixels, pixels[::-1].copy(), mask) if scope == 'channel_pair' else fn(mask, pixels)
            self.keys[key] = sorted(result)
            for name in self.keys[key]:
                self.catalog.append({'feature_index': len(self.catalog), 'feature_name': f'{key}::{name}',
                                     'family': family, 'measurement': name, 'scope': scope,
                                     'texture_scale': int(key[9:]) if key.startswith('texture_s') else None})
        self.indices = {r['feature_name']: r['feature_index'] for r in self.catalog}

    def describe(self):
        return {'name': 'cp_measure', 'version': 'cp-adapter-v1', 'input_scope': 'full_fov',
                'requires_masks': True, 'allow_nan': True, 'allows_channel_free': True,
                'feature_names': [r['feature_name'] for r in self.catalog],
                'feature_catalog': self.catalog, 'settings': self.settings,
                'dependencies': {p: version(p) for p in ('cp-measure', 'centrosome', 'scikit-image', 'numpy', 'scipy', 'mahotas')},
                'normalization': 'raw / 65535, require finite values in [0,1]; no clipping',
                'geometry_units': 'pixels; no physical spacing conversion',
                'object_selection': 'eligible reference nuclei and unique best-matched cells',
                'intensity_percentiles': 'legacy=True',
                'overlap_zero_denominator': 'all three coefficients NaN for the affected object',
                'undefined_values': 'NaN; structural NaN outside the row scope; infinity converted to NaN with warning',
                'excluded': {'multimask_overlap': 'image-level segmentation comparison, not a per-cell feature',
                             'neighbors': 'upstream 0.3.0 has singleton failure and neighbor-ID inconsistencies; not silently approximated',
                             'skeleton': 'not implemented by cp-measure'}}

    def profile_fov(self, context, batch_size):
        anchors = context.anchors.loc[context.anchors.eligible.eq(True)]
        if anchors.empty:
            return
        images = {}
        for channel, raw in context.images.items():
            image = np.asarray(raw, dtype=np.float64) / 65535.
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError(f'{context.fov_id}/{channel}: CP requires intensities in [0,65535]; no implicit clipping')
            images[channel] = image
        for compartment in self.compartments:
            ids = anchors.object_id if compartment == 'nuclei' else anchors.matched_cell_id.dropna().unique()
            objects = context.objects.loc[context.objects.compartment.eq(compartment) & context.objects.object_id.isin(ids)].sort_values('object_id')
            if objects.empty:
                continue
            if set(ids) - set(objects.object_id):
                raise ValueError('A selected anchor references a missing object')
            mask = compact_labels(context.masks[compartment], objects.label.to_numpy())
            scopes = [('geometry', [])]
            scopes += [('single_channel', [c]) for c in images]
            scopes += [('channel_pair', list(pair)) for pair in combinations(images, 2)]
            for scope, channels in scopes:
                functions = [(key, fn) for key, (kind, _, fn) in self.functions.items() if kind == scope]
                if not functions:
                    continue
                values = np.full((len(objects), len(self.catalog)), np.nan, dtype=np.float64)
                for key, fn in functions:
                    try:
                        if scope == 'channel_pair':
                            result = fn(images[channels[0]], images[channels[1]], mask)
                        else:
                            result = fn(mask, images[channels[0]] if channels else None)
                    except Exception as error:
                        raise RuntimeError(f'CP failed: FOV={context.fov_id}, compartment={compartment}, channels={channels}, family={key}') from error
                    if sorted(result) != self.keys[key]:
                        raise ValueError(f'CP schema changed in {key}')
                    for name in self.keys[key]:
                        vector = np.asarray(result[name], dtype=np.float64)
                        if vector.shape != (len(objects),):
                            raise ValueError(f'CP object alignment/shape mismatch in {key}/{name}: {vector.shape}')
                        values[:, self.indices[f'{key}::{name}']] = vector
                # CP can legitimately produce undefined divisions for empty signal.
                with np.errstate(over='ignore', invalid='ignore'):
                    values = values.astype(np.float32)
                if np.isinf(values).any():
                    warnings.warn(f'{context.fov_id}/{compartment}/{channels}: infinite CP values recorded as NaN', stacklevel=2)
                    values[np.isinf(values)] = np.nan
                for start in range(0, len(objects), batch_size):
                    block = objects.iloc[start:start + batch_size]
                    observations = pd.DataFrame({'object_id': block.object_id.to_numpy(),
                                                 'compartment': compartment, 'crop_id': None,
                                                 'channels': [channels.copy() for _ in range(len(block))]})
                    yield FeatureBatch(observations, values[start:start + batch_size])
