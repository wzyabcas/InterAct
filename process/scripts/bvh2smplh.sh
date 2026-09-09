#!/usr/bin/env bash
set -euo pipefail

export OUTPUT_DIR="/media/volume/sxu1/jianqi/code/sept27/InterAct/data/raw/output_process"
export PATH_SMPLH="/media/volume/sxu1/jianqi/code/sept27/InterAct/data/raw/smplh/"
export HUMOTO_OBJECT_DIR="/media/volume/sxu1/jianqi/code/sept27/InterAct/data/raw/humoto_objects_0805"
export PATH_BETAS="/media/volume/sxu1/jianqi/code/sept27/InterAct/data/human_betas_1005.npy"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export REPO_ROOT="$REPO_ROOT"

cd "$REPO_ROOT"

echo "Current working directory: $(pwd)"
echo "PYTHONPATH: $PYTHONPATH"

python process/bvh2smplh.py --number 0