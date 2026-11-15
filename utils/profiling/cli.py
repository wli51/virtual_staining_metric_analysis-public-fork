import argparse
import json
from pathlib import Path

import numpy as np

from utils.fov_ingestion.collection import MicroscopyCollection

from .compare import write_comparisons
from .crops import CropPolicy
from .morphem import DEFAULT_REVISION, MorphEmProfiler
from .runner import profile_collections
from .sampling import SamplingPolicy


def main():
    parser = argparse.ArgumentParser(description='Reference-anchored feature profiling')
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('morphem', 'cp'):
        profile = sub.add_parser(command)
        profile.add_argument('--reference', required=True)
        profile.add_argument('--variants', nargs='*', default=[])
        profile.add_argument('--output', required=True)
        profile.add_argument('--channels', nargs='+')
        profile.add_argument('--crop-size', type=int, default=128)
        profile.add_argument('--batch-size', type=int, default=128)
        profile.add_argument('--fov-ids-file')
        profile.add_argument('--limit', type=int)
        profile.add_argument('--sample-seed', type=int, default=42)
        profile.add_argument('--dry-run', action='store_true')
        sampling = profile.add_mutually_exclusive_group()
        sampling.add_argument('--max-instances-per-stratum', type=int)
        sampling.add_argument('--sampling-plan')
        profile.add_argument('--strata', nargs='+', default=['cell_type', 'seeding_density', 'plate'])
        profile.add_argument('--instance-seed', type=int, default=42)
        profile.add_argument('--sampling-allocation', choices=['fov_balanced', 'uniform_instances'], default='fov_balanced')
        if command == 'morphem':
            profile.add_argument('--device', default='cuda')
            profile.add_argument('--revision', default=DEFAULT_REVISION)
            profile.add_argument('--cache-dir')
            profile.add_argument('--local-files-only', action='store_true')
        else:
            profile.add_argument('--compartments', nargs='+', default=['nuclei', 'cells'])
            profile.add_argument('--families', nargs='+')
            profile.add_argument('--texture-scales', nargs='+', type=int, default=[3])
            profile.add_argument('--gray-levels', type=int, default=256)
            profile.add_argument('--no-channel-pairs', action='store_true')
            profile.add_argument('--workers', type=int, default=1, help='CP worker processes, one FOV per task')
            profile.add_argument('--threads-per-worker', type=int, default=1, help='Native numerical threads per CP worker')
    compare = sub.add_parser('compare')
    compare.add_argument('--run', required=True)
    compare.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'compare':
        print(f'Wrote {write_comparisons(args.run, args.output)} matched observations')
        return
    reference = MicroscopyCollection(args.reference)
    variants = [MicroscopyCollection(path) for path in args.variants]
    common = set.intersection(*(set(s.spec['items']) for s in [reference, *variants]))
    ids = Path(args.fov_ids_file).read_text().splitlines() if args.fov_ids_file else sorted(common)
    if args.limit is not None:
        if args.limit < 1:
            parser.error('--limit must be positive')
        ids = np.random.default_rng(args.sample_seed).choice(ids, size=min(args.limit, len(ids)), replace=False).tolist()
    if args.command == 'morphem':
        profiler = MorphEmProfiler(revision=args.revision, device=args.device,
                                  cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    else:
        from .cp_measure import CPMeasureProfiler
        profiler = CPMeasureProfiler(compartments=args.compartments, families=args.families,
                                     texture_scales=args.texture_scales, gray_levels=args.gray_levels,
                                     include_pairs=not args.no_channel_pairs)
    policy = None
    if args.max_instances_per_stratum is not None:
        policy = SamplingPolicy(args.max_instances_per_stratum, tuple(args.strata),
                                args.instance_seed, args.sampling_allocation)
    result = profile_collections(reference, variants, args.output, profiler, channels=args.channels,
                                 fov_ids=ids, crop_policy=CropPolicy(args.crop_size),
                                 batch_size=args.batch_size, dry_run=args.dry_run,
                                 sampling_policy=policy, sampling_plan=args.sampling_plan,
                                 workers=getattr(args, 'workers', 1),
                                 threads_per_worker=getattr(args, 'threads_per_worker', 1))
    print(json.dumps(result, indent=2) if args.dry_run else result.to_string(index=False))
