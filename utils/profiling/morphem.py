"""Pinned MorphEm inference, one channel per reference-nucleus-centered crop."""
from __future__ import annotations

import importlib.metadata
import re

from .api import FeatureBatch
from .crops import crop_batch

DEFAULT_REVISION = '0e8d58787421f83f975634d72420d85c5dfc9c2c'


def preprocess(patches, *, device):
    import torch
    import torch.nn.functional as F

    # Continuous 8-bit-equivalent scale, NOT uint8 quantization. This keeps the
    # documented InstanceNorm epsilon meaningful in the model card's units.
    x = torch.as_tensor(patches, dtype=torch.float32, device=device).unsqueeze(1)
    x = x / 65535.0 * 255.0
    x = F.instance_norm(x, eps=1e-7)
    return F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False, antialias=True)


class MorphEmProfiler:
    def __init__(self, *, revision=DEFAULT_REVISION, device='cuda', cache_dir=None,
                 local_files_only=False):
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Use a full immutable Hugging Face commit hash for revision')
        self.revision, self.device = revision, device
        self.cache_dir, self.local_files_only = cache_dir, local_files_only
        self._model = None

    def describe(self):
        return {
            'name': 'morphem', 'version': 'nucleus-patch-morphem-v1',
            'input_scope': 'patch', 'requires_masks': False, 'allow_nan': False,
            'feature_names': [f'cls_{i:03d}' for i in range(384)],
            'model_id': 'CaicedoLab/MorphEm', 'revision': self.revision,
            'output': 'forward_features.x_norm_clstoken',
            'preprocessing': {'scale': 'raw / 65535 * 255, no quantization or clipping',
                              'normalization': 'per-patch instance_norm', 'eps': 1e-7,
                              'resize': [224, 224], 'interpolation': 'bilinear',
                              'align_corners': False, 'antialias': True,
                              'saturation_noise': False, 'mask_pixels': False},
            'precision': 'float32, no autocast', 'device': self.device,
            'software': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy')},
        }

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                raise RuntimeError('CUDA requested but unavailable; use --device cpu explicitly')
            # Disable TF32 for this FP32 comparison pilot; do not enable AMP.
            if self.device.startswith('cuda'):
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            model, loading = AutoModel.from_pretrained(
                'CaicedoLab/MorphEm', revision=self.revision, code_revision=self.revision,
                trust_remote_code=True, cache_dir=self.cache_dir,
                local_files_only=self.local_files_only, output_loading_info=True,
            )
            if loading.get('missing_keys') or loading.get('mismatched_keys'):
                raise ValueError(f'Incomplete MorphEm encoder checkpoint: {loading}')
            if any(not key.startswith('head.') for key in loading.get('unexpected_keys', [])):
                raise ValueError(f'Unexpected non-head checkpoint weights: {loading}')
            self._model = model.float().to(self.device).eval()
            if self._model.config.embed_dim != 384:
                raise ValueError('Unexpected MorphEm embedding dimension')
        return self._model

    def profile_fov(self, context, batch_size):
        import torch
        eligible = context.anchors.loc[context.anchors.eligible.eq(True)]
        if eligible.empty:
            return
        model = self._load()
        for channel, image in context.images.items():
            for start in range(0, len(eligible), batch_size):
                anchors = eligible.iloc[start:start + batch_size]
                patches = crop_batch(image, anchors)
                with torch.inference_mode():
                    inputs = preprocess(patches, device=self.device)
                    features = model.forward_features(inputs)['x_norm_clstoken']
                    values = features.float().cpu().numpy()
                observations = anchors[['object_id', 'compartment', 'crop_id']].reset_index(drop=True)
                observations['channels'] = [[channel] for _ in range(len(observations))]
                yield FeatureBatch(observations, values)
