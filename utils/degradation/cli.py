"""Reference collection -> callable transform -> immutable derived collection."""
import argparse
from pathlib import Path

import numpy as np

from utils.degradation.degrade import degrade_collection
from utils.degradation.transforms import gaussian_pilot_specs, gaussian_spec
from utils.fov_ingestion.collection import MicroscopyCollection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--channels', nargs='+', help='Default: all intensity channels, including BF')
    parser.add_argument('--sigmas', nargs='+', type=float, help='Override pilot sigma values in pixels')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--sample-seed', type=int, default=42)
    parser.add_argument('--fov-ids-file', help='One FOV ID per line; optional frozen cohort')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    ref = MicroscopyCollection(args.reference)
    ids = (Path(args.fov_ids_file).read_text().splitlines() if args.fov_ids_file else list(ref.fovs.fov_id))
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        ids = list(np.random.default_rng(args.sample_seed).choice(ids, size=min(args.limit, len(ids)), replace=False))
    variants = ([(f'sigma_{sigma:g}', gaussian_spec(sigma)) for sigma in args.sigmas]
                if args.sigmas else gaussian_pilot_specs())
    # Sequential variants bound memory; workers parallelize FOVs within each variant.
    for level, transform in variants:
        destination = Path(args.output_root) / f'{level}_{transform.variant_id}'
        print(f'{level}: {transform.parameters} -> {destination}', flush=True)
        results = degrade_collection(ref, destination, transform, channels=args.channels,
                                     fov_ids=ids, workers=args.workers, dry_run=args.dry_run)
        print(results.to_string(index=False))


if __name__ == '__main__':
    main()
