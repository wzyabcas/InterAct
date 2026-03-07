#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR/_pointnet2_build"
CONDA_ENV="interact"
CUDA_HOME="/usr/local/cuda-12.8"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch/pointnet2_ops_lib

sed -i '/from torch.utils.cpp_extension import/i \
import torch.utils.cpp_extension\ntorch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None' setup.py

sed -i 's/os.environ\["TORCH_CUDA_ARCH_LIST"\] = .*/os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5;8.0;8.6;8.9;9.0"/' setup.py

CUDA_HOME="$CUDA_HOME" conda run -n "$CONDA_ENV" pip install .

cd "$SCRIPT_DIR"
rm -rf "$WORK_DIR"

echo "Done."
