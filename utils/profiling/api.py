"""Profiler contract supports patch embeddings and full-FOV object measurements."""
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


@dataclass
class FOVContext:
    fov_id: str
    images: dict[str, np.ndarray]  # Raw intensity planes, never implicitly normalized.
    image_rows: pd.DataFrame
    masks: dict[str, np.ndarray]   # Canonical reference cells/nuclei, uint32.
    objects: pd.DataFrame
    relationships: pd.DataFrame
    anchors: pd.DataFrame         # Includes eligibility flags and excluded nuclei.


@dataclass
class FeatureBatch:
    # Required fields: object_id, compartment, crop_id (None for object-wide CP),
    # channels (list[str], usually length 1 for MorphEm).
    observations: pd.DataFrame
    values: np.ndarray            # N x D. D and column meaning declared by describe().


class Profiler(Protocol):
    def describe(self) -> dict:
        """JSON: name, version, input_scope, feature_names, requires_masks, etc."""
        ...

    def profile_fov(self, context: FOVContext, batch_size: int) -> Iterable[FeatureBatch]:
        ...
