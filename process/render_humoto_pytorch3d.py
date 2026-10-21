import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from human_model.human_model import HumanModelDifferentiable, HUMAN_MODEL_DIR
from human_model.bone_names import MIXAMO_BONE_NAMES
import cv2
import imageio.v2 as imageio
import argparse
import numpy as np
from utils.rotation_helper import quaternion_to_matrix
from utils.pytorch3d_render_helper import *
from utils.load_humoto import *
from utils.np_torch_conversion import *

try:
    HUMOTO_OBJECT_DIR = os.environ.get('HUMOTO_OBJECT_DIR')
except:
    HUMOTO_OBJECT_DIR = None

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True,
                    help="The folder containing the PKL file to process")
parser.add_argument("-o", "--output_folder", type=str, default='',
                    help="Output root: saves the video here and pose files in a sequence subfolder. If omitted, all files are saved beside the source PKL file.")
parser.add_argument("-m", "--object_model", type=str, default=HUMOTO_OBJECT_DIR,
                    help="The path to the object model OBJ file. Default is the HUMOTO_OBJECT_DIR environment variable.")
parser.add_argument("-y", "--y_up", action='store_true',
                    help="Whether to render the sequence in y up coordinate system.")
parser.add_argument("-b", "--render_batch_size", type=int, default=50,
                    help="The batch size to render the sequence.")
parser.add_argument("-u", "--up_bone", action='store_true',
                    help="Whether to use the up bone version.")
parser.add_argument("-t", "--include_text", action='store_true',
                    help="Whether to include the text metadata.")
args = parser.parse_args()

pkl_folder_path = args.dir
output_folder = args.output_folder
folder = pkl_folder_path.split('/')[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = (900, 1600)

GROUND = create_chessboard_mesh(board_size=10, square_size=1.0, device=device, y_up=args.y_up)

COLORS = get_light_colors()

# load the sequence data
sequence_data = load_one_humoto_sequence(args.dir, 
                                         include_text=args.include_text,
                                         y_up=args.y_up, 
                                         object_model=True,
                                         object_model_path=args.object_model, 
                                         object_modality=['mesh', 'pc'],
                                         pose_params=True,
                                         bone_names=MIXAMO_BONE_NAMES
                                         )

human_pose_params = dict_to_torch(sequence_data['armature_pose_params'], device=device)
object_pose_params = dict_to_torch(sequence_data['object_pose_params'], device=device)
objects_meshes = dict_to_torch(sequence_data['object_models'], device=device)

# setup the human model
y_up = 'yup' if args.y_up else 'zup'
bone_model = 'up_bone' if args.up_bone else 'mixamo_bone'
humoto_model_path = f'human_model_{bone_model}_{y_up}.json'
human_model = HumanModelDifferentiable(character_data_path=os.path.join(HUMAN_MODEL_DIR, humoto_model_path), device=device)
human_pose_params_matrix = {bone_name: quaternion_to_matrix(human_pose_params[bone_name]) for bone_name in human_pose_params}
human_verts, human_joints = human_model(human_pose_params_matrix)
human_faces = human_model.triangulated_faces_torch

# Save the pose data in the format expected by InterAct.
pose_output_folder = os.path.join(output_folder, folder) if output_folder else pkl_folder_path
os.makedirs(pose_output_folder, exist_ok=True)
torch.save(
    {name: value.detach().cpu() for name, value in human_joints.items()},
    os.path.join(pose_output_folder, "human_joints_mixamo.pt"),
)
torch.save(
    {name: value.detach().cpu() for name, value in human_pose_params_matrix.items()},
    os.path.join(pose_output_folder, "human_pose_params_matrix.pt"),
)
np.savez(
    os.path.join(pose_output_folder, "obj_pose.npz"),
    **sequence_data['object_pose_params'],
)
print(f"Pose files saved to {pose_output_folder}")

# transform the object verts
object_transformed = {}
object_to_render = {}
for i, obj in enumerate(object_pose_params):
    object_transformed[obj] = get_transformed_object(objects_meshes[obj], object_pose_params[obj])
    object_to_render[obj] = (object_transformed[obj]['mesh'][0], object_transformed[obj]['mesh'][1], torch.tensor(COLORS[i], device=device, dtype=torch.float32))

print("Start rendering...")

frame_images = render_sequence(human_joints, human_verts, human_faces, object_to_render, image_size=IMAGE_SIZE, render_batch_size=args.render_batch_size, ground=GROUND, y_up=args.y_up, device=device)

# write the video
if not output_folder or output_folder == '':
    output_folder = pkl_folder_path
output_path = os.path.join(output_folder, f"{folder}.mp4")

video_writer = imageio.get_writer(
    output_path, fps=30, codec='libx264', pixelformat='yuv420p',
    macro_block_size=1, ffmpeg_params=['-movflags', '+faststart'],
)

print(f"Writing video to {output_path}")

for frame in frame_images:
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if args.include_text:
        text = sequence_data['text']['short_script']
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    video_writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

video_writer.close()
print(f"Video saved to {output_path}")
