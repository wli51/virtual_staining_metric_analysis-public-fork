#!/usr/bin/env bash
# Run in the existing project environment; extra arguments override defaults.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Defaults include every intensity channel, including brightfield.
# Append --channels <names...> to restrict channels, or --dry-run to plan only.
exec fov-degrade \
  --reference "${REFERENCE_ROOT:-/mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/pedcp90_ref}" \
  --output-root "${OUTPUT_ROOT:-/mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/pedcp90_blur}" \
  --sample-seed "${SAMPLE_SEED:-42}" \
  --workers "${WORKERS:-2}" \
  "$@"
