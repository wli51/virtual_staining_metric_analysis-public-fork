#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Use the active project environment; append --dry-run to validate without inference.
"${PYTHON:-python}" -u -m utils.profiling morphem \
  --reference /mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/pedcp90_ref \
  --variants \
    /mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/pedcp90_blur/low_variant_e4ce06526116c8402e0c1ba6 \
    /mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/pedcp90_blur/mid_variant_24691e5aa5a1482e9a1dd9bd \
    /mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/pedcp90_blur/high_variant_7dc34a55b1434ab75449b809 \
  --output /mnt/hdd20tb2/virtual_staining_metric_analysis_large_outputs/morphem_blur_pilot \
  --channels OrigAGP OrigDNA OrigER OrigMito OrigRNA \
  --crop-size 128 \
  --batch-size 128 \
  --device cuda \
  "$@"
