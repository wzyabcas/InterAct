import json
import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation
import gc  


MOTION_PATH = './data/behave/sequences'
OBJECT_PATH = './data/behave/objects'
MODEL_PATH = './models'
data_name = os.listdir(MOTION_PATH)

smpl_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                          gender="male",
                          use_pca=False,
                          ext='pkl')

smpl_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                          gender="female",
                          use_pca=False,
                          ext='pkl')

smpl = {'male': smpl_model_male, 'female': smpl_model_female}


for k, name in tqdm(enumerate(data_name)):
    print(name)
    with np.load(os.path.join(MOTION_PATH, name, 'object_fit_all.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, frame_times = f['angles'], f['trans'], f['frame_times']
    with np.load(os.path.join(MOTION_PATH, name, 'smpl_fit_all.npz'), allow_pickle=True) as f:
        poses, betas, trans = f['poses'], f['betas'], f['trans']
    
    frame_times = frame_times.shape[0]
    info_file = os.path.join(MOTION_PATH, name, 'info.json')
    info = json.load(open(info_file))
    gender = info['gender']
    obj_name = info['cat']
    
    smpl_model = smpl[gender]
    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                              betas=torch.from_numpy(betas).float(),
                              transl=torch.from_numpy(trans).float(),)
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
    rotvecs = poses[:, :3]
    rotations = Rotation.from_rotvec(rotvecs)
    rotation_matrix_x = Rotation.from_euler('x', -np.pi, degrees=False)
    # Apply the rotation to the batch of rotations
    rotated_rotations = rotation_matrix_x * rotations
    # Convert the rotated rotations back to rotation vectors
    poses[:, :3] = rotated_rotations.as_rotvec()

    trans = rotation_matrix_x.apply(trans)

    rotvecs2 = obj_angles
    rotations2 = Rotation.from_rotvec(rotvecs2)

    # Apply the rotation to the batch of rotations
    rotated_rotations2 = rotation_matrix_x * rotations2
    # Convert the rotated rotations back to rotation vectors
    obj_angles = rotated_rotations2.as_rotvec()
    obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis)

    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                              betas=torch.from_numpy(betas).float(),
                              transl=torch.from_numpy(trans).float(),)
    
    verts = smplx_output.vertices.detach().numpy()
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
    faces = smpl_model.faces
    
    obj_trans = pelvis + obj_trans_delta

    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
    mesh_obj.vertices = (obj_verts - obj_verts.mean(axis=0, keepdims=True))

    mesh_obj.export(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"))

    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = mesh_obj.vertices[None, ...]
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    diff_fix = min(verts[:30, ..., 1].min(), obj_verts[:30, ..., 1].min())
    obj_trans[..., 1] -= diff_fix
    trans[..., 1] -= diff_fix

    obj = {
        'angles': obj_angles,
        'trans': obj_trans,
        'name': obj_name,
    }
    human = {
        'poses': poses,
        'betas': betas[0],
        'trans': trans,
        'gender': gender,
    }

    np.savez(os.path.join(MOTION_PATH, name, 'object.npz'), **obj)
    np.savez(os.path.join(MOTION_PATH, name, 'human.npz'), **human)

    del smplx_output, verts, pelvis, obj_verts, angle_matrix
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()

