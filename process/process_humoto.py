
import os
import os.path
import numpy as np
import smplx
import logging
from tqdm import tqdm
from typing import List

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from process.utils import ObjectSequence, HumanSequence, Processor, Debugger

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

dict_path = {
    'PATH_RAW': './data/humoto/raw',
    'MOTION_PATH_RAW_HUMAN': './data/humoto/raw/smplh', # where we load human motion sequences
    'MOTION_PATH_RAW_OBJECT': './data/humoto/raw/output_process', # where we load object motion sequences
    'OBJECT_PATH_RAW': './data/humoto/raw/humoto_objects_0805', # where we load object meshes
    
    'MOTION_PATH': './data/humoto/sequences',   # where we save processed human+object sequences
    'OBJECT_PATH': './data/humoto/objects', 
    
    'MODEL_PATH': './models',
}

if not os.path.isdir(dict_path['PATH_RAW']):
    raise FileNotFoundError(f"Raw data path does not exist: {dict_path['PATH_RAW']}. Please check your paths.")

visualize_raw_data = False

# The processor can memorize the paths. If so, it would later try to load by memory first.
# In the interest of brevity, please send the paths to the processor when you create it.
procer = Processor(dict_path)  

sequence_names = [
    os.path.splitext(filename)[0]
    for filename in os.listdir(dict_path['MOTION_PATH_RAW_HUMAN'])
    if filename.endswith('.npz')
]

# optional: filter sequence_names if visualize_raw_data is True
if visualize_raw_data:
    sequence_names = ['walking_back_and_forth_while_pushing_the_clothes_rack_with_right_hand_then_left_hand-594']

smpl_model_male = smplx.create(dict_path['MODEL_PATH'], model_type='smplh',
                          gender="male",
                          use_pca=False,
                          num_betas=10,
                          flat_hand_mean=True,
                          ext='pkl')

smpl_model_female = smplx.create(dict_path['MODEL_PATH'], model_type='smplh',
                          gender="female",
                          use_pca=False,
                          num_betas=10,
                          flat_hand_mean=True,
                          ext='pkl')

smpl = {'male': smpl_model_male, 'female': smpl_model_female}

if not visualize_raw_data:
    # create dirs for objects and sequences, and make sure they are empty
    # If we are visualizing, we might have already run this code and created the directories, so we don't want to delete them.
    if os.path.isdir(dict_path['MOTION_PATH']):
        raise FileExistsError(
            f"Directory {dict_path['MOTION_PATH']} already exists.\nPlease delete it."
        )
    if os.path.isdir(dict_path['OBJECT_PATH']):
        raise FileExistsError(
            f"Directory {dict_path['OBJECT_PATH']} already exists.\nPlease delete it."
        )
    os.makedirs(dict_path['MOTION_PATH'], exist_ok=False)
    os.makedirs(dict_path['OBJECT_PATH'], exist_ok=False)
    logging.info(f"Created directory: {dict_path['OBJECT_PATH']}")
    logging.info(f"Created directory: {dict_path['MOTION_PATH']}")

# PROCESSING LOOP
success = []; failure = []
for i, name in tqdm(enumerate(sequence_names)):
    try:
        logging.info(f"Processing sequence: {name}")

        sequence_dir = os.path.join(procer.motion_path, name)
        if not visualize_raw_data:
            os.makedirs(sequence_dir, exist_ok=False)
        else: os.makedirs(sequence_dir, exist_ok=True)

        human_sequence:HumanSequence = procer._load_human_sequence(dict_path['MOTION_PATH_RAW_HUMAN'], name, False)
        objs_sequence:List[ObjectSequence] = procer._load_objects_sequence(dict_path['MOTION_PATH_RAW_OBJECT'], dict_path['OBJECT_PATH_RAW'], name)
        
        logging.debug("rotating the whole HOI, to make sure the coordinate system is y-up ...")
        human_sequence, objs_sequence = procer._rotate_HOI_at_given_angle(human_sequence, objs_sequence, angle=0) # humoto is already y-up.
        logging.debug("lifting/descending the whole HOI ...")
        human_sequence, objs_sequence, diff_fix = procer._make_sure_HOI_on_ground(human_sequence, objs_sequence, smpl)
        logging.debug(f"The minimum y value of the human and object vertices is {diff_fix}.")
        logging.debug("visualizing the body, objects, and axes ...")
        # if i < 5:
        #     Debugger.visualize_body_obj_and_axes(
        #         body_verts_faces=human_sequence.get_body_verts_faces_smpl(smpl),
        #         obj_verts_faces=ObjectSequence.merge_object_meshes(objs_sequence),
        #         save_path=f"/media/volume/sxu1/jianqi/code/test/InterAct/results/raw_visualize_{name}.mp4",
        #         axis_length=0.5,
        #         multi_angle=True,
        #         h=512,
        #         w=512,
        #     )

        for obj_sequence in objs_sequence:
            dict_obj_sequence = {
                "angles": obj_sequence.angles,
                "trans": obj_sequence.trans,
                "mesh": obj_sequence.mesh,
                "mesh_name": obj_sequence.mesh_name,
                "instance_name": obj_sequence.instance_name,
            }
            object_sequence_path = os.path.join(
                sequence_dir,
                f"object_{obj_sequence.instance_name}.npz",
            )
            np.savez(object_sequence_path, **dict_obj_sequence)
        dict_human_sequence = {
            "poses": human_sequence.poses,
            "betas": human_sequence.betas[0],
            "trans": human_sequence.trans,
            "gender": human_sequence.gender
        }
        np.savez(os.path.join(sequence_dir, 'human.npz'), **dict_human_sequence)

        logging.info(f"Saved human.npz + {len(objs_sequence)} object(s) for {name}")
        logging.debug(f"objects instance name: {[obj.instance_name for obj in objs_sequence]}")
        logging.debug("\n------------------------------------------------\nnext sequence:\n")
        success.append(name)
    except Exception as exc:
        logging.error(f"Error processing sequence: {name}")
        failure.append((name, repr(exc)))

logging.info(f"Successfully processed {len(success)} sequences: {success}")
logging.info(f"Failed to process {len(failure)} sequences: {failure}")

