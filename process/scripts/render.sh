# #!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATH_PROJECT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

export HUMOTO_DATASET_DIR="/media/volume/sxu1/jianqi/code/sept27/InterAct/data/raw/up_bone_humoto"
export HUMOTO_OBJECT_DIR="/media/volume/sxu1/jianqi/code/sept27/InterAct/data/raw/humoto_objects_0805"

OUTPUT_DIR="/media/volume/sxu1/jianqi/code/sept27/InterAct/data/raw/output_process"

for seq_path in "$HUMOTO_DATASET_DIR"/*; do
    # Skip anything that is not a directory
    [[ -d "$seq_path" ]] || continue

    # Get only the sequence name
    seq_name="$(basename "$seq_path")"

    echo "========================================"
    echo "Processing: $seq_name"
    echo "========================================"

    "${HUMOTO_PYTHON:-python}" \
        "/media/volume/sxu1/jianqi/code/sept27/humoto/render_humoto_pytorch3d.py" \
        -d "$seq_name" \
        -o "$OUTPUT_DIR" \
        -y \
        -u \
        -b 20
done