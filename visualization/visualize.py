import json
import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from copy import copy

import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('./text2interaction')
from render.mesh_viz import visualize_body_obj
from human_body_prior.body_model.body_model import BodyModel



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

dataset = sys.argv[1].upper()

MOTION_PATH = './data/{}/sequences_canonical'.format(dataset.lower())
OBJECT_PATH = './data/{}/objects'.format(dataset.lower())
MODEL_PATH = './models'


data_name = os.listdir(MOTION_PATH)
######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl')

smplh10 = {'male': smplh_model_male, 'female': smplh_model_female}
######################################## smplx 10 ########################################
smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                        gender = 'male',
                        use_pca=False,
                        ext='pkl')
                           
smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="female",
                        use_pca=False,
                        ext='pkl')

smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl')

smplx10 = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}
######################################## smplx 12 ########################################
smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="male",
                          num_pca_comps=12,
                          ext='pkl')

smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="female",
                          num_pca_comps=12,
                          ext='pkl')

smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="neutral",
                          num_pca_comps=12,
                          ext='pkl')

smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
######################################## smplh 16 ########################################
SMPLH_PATH = MODEL_PATH+'/smplh'
surface_model_male_fname = os.path.join(SMPLH_PATH,'male', "model.npz")
surface_model_female_fname = os.path.join(SMPLH_PATH, "female","model.npz")
surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 

smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}
######################################## smplx 16 ########################################
SMPLX_PATH = MODEL_PATH+'/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16 = {'male': smplx16_model_male, 'female': smplx16_model_female, 'neutral': smplx16_model_neutral}
########################################################################################
results_folder = "./results"
os.makedirs(results_folder, exist_ok=True)

######################################## Visualize SMPL ########################################
def visualize_smpl(name, MOTION_PATH, model_type, num_betas, num_pca_comps=None):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
        
    frame_times = poses.shape[0]
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                transl=torch.from_numpy(trans).float(),) 
        elif model_type == 'smplx':
            if num_pca_comps == 12:
                smpl_model = smplx12[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                    global_orient=torch.from_numpy(poses[:, :3]).float(),
                                    left_hand_pose=torch.from_numpy(poses[:, 66:78]).float(),
                                    right_hand_pose=torch.from_numpy(poses[:, 78:90]).float(),
                                    jaw_pose=torch.zeros(frame_times, 3).float(),
                                    leye_pose=torch.zeros(frame_times, 3).float(),
                                    reye_pose=torch.zeros(frame_times, 3).float(),
                                    expression=torch.zeros(frame_times, 10).float(),
                                    betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                    transl=torch.from_numpy(trans).float(),)
            else:
                smpl_model = smplx10[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                    global_orient=torch.from_numpy(poses[:, :3]).float(),
                                    left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                    right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                    jaw_pose = torch.zeros([frame_times,3]).float(),
                                    reye_pose = torch.zeros([frame_times,3]).float(),
                                    leye_pose = torch.zeros([frame_times,3]).float(),
                                    expression = torch.zeros([frame_times,10]).float(),
                                    betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                    transl=torch.from_numpy(trans).float(),)
        verts = to_cpu(smplx_output.vertices)
        faces = smpl_model.faces
    elif num_betas == 16: 
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
                            pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
                            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
                            root_orient=torch.from_numpy(poses[:, :3]).float(), 
                            trans=torch.from_numpy(trans).float())
        verts = to_cpu(smplx_output.v)
        faces = smpl_model.f
    
    return verts, faces



######################################## Visualize GRAB ########################################
def visualize_grab(name, MOTION_PATH):
    """
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, vtemp, trans, gender = f['poses'], f['vtemp'], f['trans'], str(f['gender'])
    n_comps = 24
    T = len(poses)

    smpl_model = smplx.create( 
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=gender,
        num_pca_comps=n_comps,
        v_template = vtemp,
        batch_size=T)

    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                            global_orient=torch.from_numpy(poses[:, :3]).float(),
                            left_hand_pose=torch.from_numpy(poses[:, 66:90]).float(),
                            right_hand_pose=torch.from_numpy(poses[:, 90:114]).float(),
                            transl=torch.from_numpy(trans).float(),)
    verts = to_cpu(smplx_output.vertices)
    faces = smpl_model.faces

    return verts, faces

# visualize surface motion of smpl model
for k, name in tqdm(enumerate(data_name)):
    print(name)
    try:
        if dataset == 'GRAB':
            verts, faces = visualize_grab(name, MOTION_PATH)
        elif dataset == 'BEHAVE':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
        elif dataset == 'NEURALDOME' or dataset == 'IMHD':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
        elif dataset == 'CHAIRS':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
        elif dataset == 'INTERCAP':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10, 12)
        elif dataset == 'OMOMO':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 16)

        with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
            obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

        mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
        obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

        angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
        obj_verts = (obj_verts)[None, ...]
        obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
        rend_video_path = os.path.join(results_folder, '{}_{}_{}.mp4'.format(dataset, name, obj_name))
        visualize_body_obj(verts, faces, obj_verts, obj_faces, save_path=rend_video_path, show_frame=True, multi_angle=True)
    except Exception as e:
        print(e)
        continue

