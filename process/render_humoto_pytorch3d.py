import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.human_model.human_model import HumanModelDifferentiable, HUMAN_MODEL_DIR
from data.human_model.bone_names import MIXAMO_BONE_NAMES
import cv2
import argparse
from text2interaction.utils.rotation_helper import quaternion_to_matrix
from text2interaction.utils.pytorch3d_render_helper import *
from text2interaction.utils.load_humoto import *
from text2interaction.utils.np_torch_conversion import *
from PIL import Image
import imageio

try:
    HUMOTO_OBJECT_DIR = os.environ.get('HUMOTO_OBJECT_DIR')
except:
    HUMOTO_OBJECT_DIR = None

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, required=True,
                    help="The folder containing the PKL file to process")
parser.add_argument("-o", "--output_folder", type=str, default='',
                    help="The folder to save the rendered video, if not specified, the rendered video will be saved in the same folder as the original PKL file")
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
parser.add_argument("-n", "--number",type=int, default=0,
                    help="Whether to include the text metadata.")


args = parser.parse_args()

# bp='/projects/bbsg/ziyin/HUMOTO/up_bone_humoto'
bp = HUMOTO_DATASET_DIR
LISTS = sorted(os.listdir(bp))
# L = len(LISTS)//20+1
# LISTS = LISTS[args.number*L:min(args.number*L+L,len(LISTS))]
print(args.output_folder)
from tqdm import tqdm
objss=[]
# LISTS=['carrying_low_chair_with_left_arm-847','carrying_low_chair_with_right_hand-599','carrying_mixing_bowl_with_both_hands-134','carrying_mug_with_left_hand-125']
# LISTS=['add_ingredients_from_deep_plate_to_mixing_bowl_with_left_hand-900','add_ingredients_from_deep_plate_to_mixing_bowl_with_right_hand-639']
# LISTS = ['adding_dressing_with_right_hand_to_mixing_bowl_on_table-662']
# LISTS=['carrying_mug_with_left_hand-757','carrying_mug_with_left_hand-125']
ats=['spoon.001', 'knife.001', 'working_chair.001', 'deep_plate.001', 'clothes_hanger.001', 'laptop.001', 'organizer_small.001', 'screwdriver.001', 'notebook.001', 'spoon.002', 'organizer_medium.001', 'mug.001', 'spoon.003', 'clothes_hanger.002']
for nn in tqdm(LISTS[:]):
    args.dir =nn
    pkl_folder_path = args.dir
    output_folder = args.output_folder
    # print(nn,args.dir)
    # print(fuck)
    folder = pkl_folder_path.split('/')[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    IMAGE_SIZE = (900, 1600)

    GROUND = create_chessboard_mesh(board_size=10, square_size=1.0, device=device, y_up=args.y_up)

    COLORS = get_light_colors()

    # load the sequence data
    sequence_data,obj_names_orig = load_one_humoto_sequence(args.dir, 
                                            include_text=args.include_text,
                                            y_up=args.y_up, 
                                            object_model=True,
                                            object_model_path=args.object_model, 
                                            object_modality=['mesh', 'pc'],
                                            pose_params=True,
                                            bone_names=MIXAMO_BONE_NAMES
                                            )
    # weirds= []
    # for n in obj_names_orig:
    #     if '.' in n:
    #         weirds.append(n)
    # if len(weirds) ==0:
    #     continue
    # print(weirds)

    human_pose_params = dict_to_torch(sequence_data['armature_pose_params'], device=device)
    object_pose_params = dict_to_torch(sequence_data['object_pose_params'], device=device)
    objects_meshes = dict_to_torch(sequence_data['object_models'], device=device)
    # for i, obj in enumerate(object_pose_params):
    #     if '.' in obj:
    #         objss.append(obj)
    # continue
            # print(obj,object_pose_params[obj][0])
    
    
    # print(objects_meshes.keys(),object_pose_params,keys(),'KEYS')

    # setup the human model
    y_up = 'yup' if args.y_up else 'zup'
    bone_model = 'up_bone' if args.up_bone else 'mixamo_bone'
    humoto_model_path = f'human_model_{bone_model}_{y_up}.json'
    # print(humoto_model_path,'HUMOTO_P')
    
    human_model = HumanModelDifferentiable(character_data_path=os.path.join(HUMAN_MODEL_DIR, humoto_model_path), device=device)
    human_pose_params_matrix = {bone_name: quaternion_to_matrix(human_pose_params[bone_name]) for bone_name in human_pose_params}
    human_verts, human_joints = human_model(human_pose_params_matrix)
    from IPython import embed; embed()
    os.makedirs(os.path.join(output_folder,args.dir),exist_ok=True)
    # print(os.path.join(output_folder,args.dir))
    torch.save(human_joints,os.path.join(output_folder,args.dir,'human_joints_mixamo.pt'))
    torch.save(human_pose_params_matrix,os.path.join(output_folder,args.dir,'human_pose_params_matrix.pt'))
    # print(sequence_data['object_pose_params'].keys())
    np.savez(os.path.join(output_folder,args.dir,'obj_pose.npz'),**sequence_data['object_pose_params'])
    print(os.path.join(output_folder,args.dir))
    # for key,value in human_joints.items():
    #     print(key,value.shape)
    # X=np.array(list(human_joints.keys()))
    # print(fuck)
    # np.save(X,'./joints_dict.')
#     human_faces = human_model.triangulated_faces_torch

# #     # transform the object verts
#     object_transformed = {}
#     object_to_render = {}
#     for i, obj in enumerate(object_pose_params):
        
#         object_transformed[obj] = get_transformed_object(objects_meshes[obj], object_pose_params[obj])
#         object_to_render[obj] = (object_transformed[obj]['mesh'][0], object_transformed[obj]['mesh'][1], torch.tensor(COLORS[i], device=device, dtype=torch.float32))

# # #     # print("Start rendering...")
# # # print(list(set(objss)),'OBJSS')
#     frame_images = render_sequence(human_joints, human_verts, human_faces, object_to_render, image_size=IMAGE_SIZE, render_batch_size=args.render_batch_size, ground=GROUND, y_up=args.y_up, device=device)
#     output_path = os.path.join(output_folder, f"{folder}.mp4")
#     imageio.mimsave(output_path, frame_images, fps=30)
#     print(f"Video saved to {output_path}")


# # write the video
# if not output_folder or output_folder == '':
#     output_folder = pkl_folder_path
# # output_path = os.path.join(output_folder, f"{folder}.mp4")

# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# output_path = os.path.join(output_folder, f"{folder}.avi").replace('-','_')
# # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# height, width, _ = frame_images[0].shape
# video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
# img = Image.fromarray(frame_images[0])
# img.save("output.png")
# print(f"Writing video to {output_path}")

# for frame in frame_images:
#     print(frame.shape)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     if args.include_text:
#         text = sequence_data['text']['short_script']
#         cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
#     video_writer.write(frame)

# video_writer.release()
# print(f"Video saved to {output_path}")

