"""Deterministic reference-object anchors, shared across every image variant."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from utils.fov_ingestion.writer import digest


@dataclass(frozen=True)
class CropPolicy:
    size: int = 128

    def descriptor(self):
        if not isinstance(self.size, int) or self.size < 2:
            raise ValueError('crop size must be an integer >= 2')
        return {**asdict(self), 'version': 'nucleus-crop-v1', 'border': 'exclude',
                'center': 'rounded_centroid_else_nearest_instance_pixel',
                'rounding': 'floor(x+0.5)', 'masked': False}


ANCHOR_COLUMNS = [
    'fov_id', 'mask_set_id', 'object_id', 'compartment', 'label', 'crop_id',
    'centroid_y', 'centroid_x', 'center_y', 'center_x', 'y0', 'x0', 'y1', 'x1',
    'eligible', 'exclusion_reason', 'matched_cell_id', 'nucleus_cell_overlap',
    'nuclei_in_crop', 'cell_foreground_fraction',
]


def make_anchors(fov_id, objects, relationships, nuclei, cells, policy):
    descriptor = policy.descriptor()
    if nuclei.ndim != 2 or cells.shape != nuclei.shape:
        raise ValueError('Reference mask shapes differ')
    nuclear = objects.loc[objects.compartment.eq('nuclei')].sort_values('object_id')
    if nuclear.object_id.duplicated().any() or nuclear.label.duplicated().any():
        raise ValueError('Duplicate nuclear identities')
    if set(np.unique(nuclei)) - {0} != set(nuclear.label):
        raise ValueError('Reference nuclear labels and object table differ')
    best = relationships.loc[relationships.is_best_match.eq(True)] if len(relationships) else relationships
    if len(best) and best.nucleus_id.duplicated().any():
        raise ValueError('Ambiguous best nucleus-cell relation')
    links = {r['nucleus_id']: r for r in best.to_dict('records')}
    h, w = nuclei.shape
    rows = []
    for obj in nuclear.itertuples(index=False):
        cy, cx = int(np.floor(obj.centroid_y + .5)), int(np.floor(obj.centroid_x + .5))
        if not (0 <= cy < h and 0 <= cx < w and nuclei[cy, cx] == obj.label):
            # Concave instances can have a centroid outside their foreground.
            yy, xx = np.nonzero(nuclei[obj.y0:obj.y1, obj.x0:obj.x1] == obj.label)
            yy, xx = yy + obj.y0, xx + obj.x0
            if not len(yy):
                raise ValueError(f'Missing nucleus {obj.object_id}')
            index = np.argmin((yy - obj.centroid_y)**2 + (xx - obj.centroid_x)**2)
            cy, cx = int(yy[index]), int(xx[index])
        y0, x0 = cy - policy.size // 2, cx - policy.size // 2
        y1, x1 = y0 + policy.size, x0 + policy.size
        eligible = y0 >= 0 and x0 >= 0 and y1 <= h and x1 <= w
        link = links.get(obj.object_id, {})
        rows.append({
            'fov_id': fov_id, 'mask_set_id': obj.mask_set_id, 'object_id': obj.object_id,
            'compartment': 'nuclei', 'label': int(obj.label),
            'crop_id': 'crop_' + digest([obj.object_id, descriptor, [y0, x0, y1, x1]])[:24],
            'centroid_y': obj.centroid_y, 'centroid_x': obj.centroid_x,
            'center_y': cy, 'center_x': cx, 'y0': y0, 'x0': x0, 'y1': y1, 'x1': x1,
            'eligible': eligible, 'exclusion_reason': '' if eligible else 'crop_crosses_fov_boundary',
            'matched_cell_id': link.get('cell_id'),
            'nucleus_cell_overlap': link.get('nucleus_overlap_fraction'),
            'nuclei_in_crop': int(np.count_nonzero(np.unique(nuclei[y0:y1, x0:x1]))) if eligible else None,
            'cell_foreground_fraction': float(np.mean(cells[y0:y1, x0:x1] > 0)) if eligible else None,
        })
    return pd.DataFrame(rows, columns=ANCHOR_COLUMNS)


def crop_batch(image, anchors):
    return np.stack([image[r.y0:r.y1, r.x0:r.x1] for r in anchors.itertuples(index=False)]).astype(np.float32)
