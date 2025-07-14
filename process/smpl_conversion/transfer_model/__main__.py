# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

import os
from copy import copy

import os.path as osp
import sys
import pickle
import trimesh
import pickle

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm
import shutil

from smplx import build_layer

from .config import parse_args
from .data import build_dataloader
from .transfer_model import run_fitting
from .utils import read_deformation_transfer, np_mesh_to_o3d
from human_body_prior.body_model.body_model import BodyModel

import smplx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



MODEL_PATH = '../../models'


######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl')

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl')

smplh10 = {'male': smplh_model_male, 'female': smplh_model_female, 'neutral': smplh_model_neutral}
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
######################################## smplx 10 pca 12 ########################################
smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="male",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')

smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="female",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')
smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="neutral",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')
smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
######################################## smplh 16 ########################################
SMPLH_PATH = MODEL_PATH+'/smplh'
surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
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
def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 10 PCA 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
        
    frame_times = poses.shape[0]
    if frame_times > 400:
        return None,None,None,None,None
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
            if use_pca:
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
        verts = (smplx_output.vertices)
        faces = smpl_model.faces.astype(np.int32)
        faces = torch.from_numpy(faces).unsqueeze(0).repeat(frame_times, 1, 1)
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
        verts = (smplx_output.v)
        faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)
    
    return verts, faces , None, None,gender

######################################## utils for GRAB ########################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

    points = points.reshape(-1,3)
    n_points = points.shape[0]

    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

        spheres.append(sphs)

    spheres = Mesh.concatenate_meshes(spheres)
    return spheres

class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 process = False,
                 visual = None,
                 wireframe=False,
                 smooth = False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process = process)
            vertices = mesh.vertices
            faces= mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices*vscale

        if faces is None:
            mesh = points2sphere(vertices)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self,color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self,vc, vertex_ids = None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self,fc, face_ids = None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)
def DotDict(in_dict):
    
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

sbj_info = {}
def load_sbj_verts(sbj_id, seq_data, data_root_folder = '../../data/grab'):
    
    mesh_path = os.path.join(data_root_folder,seq_data.body.vtemp)
    if sbj_id in sbj_info:
        sbj_vtemp = sbj_info[sbj_id]
    else:
        sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
        sbj_info[sbj_id] = sbj_vtemp
    return sbj_vtemp
######################################## Visualize GRAB ########################################
def visualize_grab(name, MOTION_PATH):
    """
    vertices: (N, 10475, 3)
    """
    motion_file = os.path.join(MOTION_PATH,name,'motion.npz')
    seq_data = parse_npz(motion_file)
    n_comps = seq_data['n_comps']
    gender = seq_data['gender']
    sbj_id = seq_data['sbj_id']
    T = seq_data.n_frames
    if T>400:
        return None,None,None,None,None
    # sbj_vtemp = load_sbj_verts(sbj_id, seq_data, os.path.dirname(MOTION_PATH))
    sbj_vtemp = load_sbj_verts(sbj_id, seq_data)

    smpl_model = smplx.create( 
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=gender,
        num_pca_comps=n_comps,
        v_template = sbj_vtemp,
        batch_size=T).cuda()
    sbj_parms = params2torch(seq_data.body.params)

    smplx_output = smpl_model(**sbj_parms)
    verts = (smplx_output.vertices)
    faces = smpl_model.faces.astype(np.int32)

    return verts, torch.from_numpy(faces).unsqueeze(0).repeat(T,1,1),None, None,gender

def get_variables(
    batch_size,
    NB = 10,
    dtype = torch.float32
):
    var_dict = {}

    device =torch.device('cuda:0')
    if NB == 10:
        var_dict.update({
            'transl': torch.zeros(
                [batch_size, 3], device=device, dtype=dtype),
            'global_orient': torch.zeros(
                [batch_size, 1, 3], device=device, dtype=dtype),
            'body_pose': torch.zeros(
                [batch_size, 21, 3],
                device=device, dtype=dtype),
            'betas': torch.zeros([batch_size, 10],
                                 dtype=dtype, device=device),
        })
        var_dict.update(
            left_hand_pose=torch.zeros(
                [batch_size, 15, 3], device=device,
                dtype=dtype),
            right_hand_pose=torch.zeros(
                [batch_size, 15, 3], device=device,
                dtype=dtype),
        )
        # pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
        #                     pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
        #                     betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
        #                     root_orient=torch.from_numpy(poses[:, :3]).float(), 
        #                     trans=torch.from_numpy(trans).float())
    else:
        var_dict.update(
            pose_body = torch.zeros(
                [batch_size, 63],
                device=device, dtype=dtype),
            pose_hand=torch.zeros(
                [batch_size, 90], device=device,
                dtype=dtype),
            betas = torch.zeros([batch_size, 16],
                                 dtype=dtype, device=device),
            root_orient= torch.zeros(
                [batch_size, 3], device=device, dtype=dtype),
            transl= torch.zeros(
                [batch_size, 3], device=device, dtype=dtype)
        )
        

    

    
    # Toggle gradients to True
    for key, val in var_dict.items():
        val.requires_grad_(True)

    return var_dict

def main() -> None:
    exp_cfg = parse_args()

    if torch.cuda.is_available() and exp_cfg["use_cuda"]:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if exp_cfg["use_cuda"]:
            if input("use_cuda=True and GPU is not available, using CPU instead,"
                     " would you like to continue? (y/n)") != "y":
                sys.exit(3)
    dataset = exp_cfg.dataset
    numbers =exp_cfg.numbers
    MOTION_PATH = os.path.join('../../data',dataset,'sequences_canonical')
    
    MOTION_PATH_OUT = os.path.join('../../data_smplh',dataset,'sequences_canonical')
    os.makedirs(MOTION_PATH_OUT,exist_ok=True)
    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)
    L = len(os.listdir(MOTION_PATH))
    
    
    
    for name in tqdm((os.listdir(MOTION_PATH))):
        try:
            verts, joints, faces, poses,gender = None, None ,None, None,None
            tbase = os.path.join(MOTION_PATH_OUT,name)
            
            if os.path.isfile(os.path.join(tbase,'human.npz')):
                continue
            if dataset.upper() == 'GRAB':
                verts, faces, joints, poses, gender = visualize_grab(name, MOTION_PATH)
                NB = 10
              
            elif dataset.upper() == 'BEHAVE' or dataset.upper() =='BEHAVE_CORRECT':
                verts,  faces, joints, poses,gender = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
                NB = 10
                
            elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
                verts, faces, joints, poses,gender = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
                NB = 16
                
            elif dataset.upper() == 'CHAIRS':
                verts,  faces, joints,poses,gender = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
                NB = 10
                
            elif dataset.upper() == 'INTERCAP' or dataset.upper() =='INTERCAP_CORRECT':
                verts,  faces, joints, poses,gender = visualize_smpl(name, MOTION_PATH, 'smplx', 10, True)
                NB = 10
                
            elif dataset.upper() == 'OMOMO' or dataset.upper() == 'OMOMO_CORRECT':
                verts,  faces, joints, poses,gender = visualize_smpl(name, MOTION_PATH, 'smplx', 16)
                # body_model = smplh16[gender]
                NB = 16
            
                
            if not gender:
                continue
           
            batch = {}
            batch['vertices'] = verts.detach().to(device)
            batch['faces'] = faces.detach().to(device)
            # mask_ids = None
            mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
            mask_ids = None
            if osp.exists(mask_ids_fname):
                # logger.info(f'Loading mask ids from: {mask_ids_fname}')
                mask_ids = np.load(mask_ids_fname)
                mask_ids = torch.from_numpy(mask_ids).to(device=device)
            
            NB = 10
            model_path = exp_cfg.body_model.folder
            exp_cfg.body_model.gender = gender
            body_model = build_layer(model_path, **exp_cfg.body_model)
            body_model = body_model.to(device)
            var_dict_original = None
            var_dict = run_fitting(
                exp_cfg, batch, body_model, def_matrix, mask_ids,var_dict_original)
            var_dict['gender'] = np.array(gender)
            os.makedirs(tbase,exist_ok=True)
            np.save(os.path.join(tbase,'human.npz'),**var_dict)
            
            logger.info((f'finish: {name}'))
        except Exception as e:
            print(e)
    logger.info(f'END: {exp_cfg.numbers},{dataset}')
    


if __name__ == '__main__':
    main()
