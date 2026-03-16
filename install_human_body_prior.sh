#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR/_human_body_prior_build"
HBP_BRANCH="staging"
HBP_REPO="https://github.com/nghorbani/human_body_prior.git"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

git clone --depth 1 --branch "$HBP_BRANCH" "$HBP_REPO"
cd human_body_prior

pip install --no-deps --ignore-requires-python .

python -c "from human_body_prior.tools import tgm_conversion as tgm; print('tgm_conversion OK:', hasattr(tgm, 'angle_axis_to_rotation_matrix'))"

cd "$SCRIPT_DIR"
rm -rf "$WORK_DIR"

echo "Done."
