import argparse
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import numpy as np
import torch
import json

np.set_printoptions(precision=6, suppress=True)
torch.set_printoptions(precision=6, sci_mode=False)


# 1. human_pos.npz：查看所有键、每个数组的形状和数值
human_pos_path = Path("/work/nvme/bdeg/ziyin/humoto/human_model/human_pos.npz")
human_pos_data = np.load(human_pos_path, allow_pickle=False)
json_human_pos_path = Path(__file__).resolve().parents[1] / "results" / "human_pos.json"
json_human_pos_path.parent.mkdir(parents=True, exist_ok=True)
print("lens of human_pos_data:", len(human_pos_data.files))
with json_human_pos_path.open("w", encoding="utf-8") as f:
    json.dump({name: human_pos_data[name].tolist() for name in human_pos_data.files}, f, indent=4)

    
# 2. human_betas.npy：直接得到一个数组
betas = np.load("/work/nvme/bdeg/ziyin/humoto/human_model/human_betas.npy", allow_pickle=False)
path_human_betas = Path(__file__).resolve().parents[1] / "results" / "human_betas.json"
path_human_betas.parent.mkdir(parents=True, exist_ok=True)
print("lens of human_betas:", betas.shape)
with path_human_betas.open("w", encoding="utf-8") as f:
    json.dump(betas.tolist(), f, indent=4)

# 2'. test_human_betas.npy：直接得到一个数组
betas = np.load("../data/test_human_betas.npy", allow_pickle=False)
path_human_betas = Path(__file__).resolve().parents[1] / "results" / "test_human_betas.json"
path_human_betas.parent.mkdir(parents=True, exist_ok=True)
print("lens of human_betas:", betas.shape)
with path_human_betas.open("w", encoding="utf-8") as f:
    json.dump(betas.tolist(), f, indent=4)

# 3. 查看某个动作的两份 .pt 文件
sequence_dir = Path(
    "../data/output_process/add_ingredients_from_deep_plate_to_mixing_bowl_with_left_hand-900"
)
# "add_ingredients_from_deep_plate_to_mixing_bowl_with_left_hand-900"
# adjusting_shelf_leveling_screw-686
for filename in [
    "human_joints_mixamo.pt",
    "human_pose_params_matrix.pt",
]:
    data = torch.load(
        sequence_dir / filename,
        map_location="cpu",
        weights_only=True,
    )
    
    # save data to json
    path_pt_file = Path(__file__).resolve().parents[1] / "results" / f"{filename}.json" 
    print("lens of {}: {}".format(filename, len(data)))
    print(len(data["mixamorig:Hips"]), len(data["mixamorig:HeadTop_End"]))
    print(f"shape:({len(data)},{len(data['mixamorig:Hips'])},{len(data['mixamorig:Hips'][0])})")
    with path_pt_file.open("w", encoding="utf-8") as f:
        json.dump({name: data[name].tolist() for name in data.keys()}, f, indent=4)

