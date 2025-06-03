import os
import os.path
import numpy as np
import torch
from copy import copy
import smplx
from render.mesh_utils import Mesh
import trimesh
from scipy.spatial.transform import Rotation


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def DotDict(in_dict):

    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def to_cuda(v, dtype = torch.float32):
    return torch.from_numpy(v).type(dtype).to(device)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

def load_sbj_verts(sbj_id, seq_data):
    mesh_path = os.path.join(body_root_folder,seq_data.body.vtemp)
    if sbj_id in sbj_info:
        sbj_vtemp = sbj_info[sbj_id]
    else:
        sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
        sbj_info[sbj_id] = sbj_vtemp
    return sbj_vtemp

sbj_info = {}
data_root_folder = './data/grab/raw'
motion_folder = './data/grab/raw/grab'
model_folder = './models'
body_root_folder = './data/grab/raw/'
OBJECT_PATH_RAW = './data/grab/raw/tools/object_meshes'
OBJECT_PATH = './data/grab/objects'
MOTION_PATH = './data/grab/sequences'

# process objects
if not os.path.exists(OBJECT_PATH):
    for obj_ply in os.listdir(OBJECT_PATH_RAW):
        obj_name = obj_ply.split('.')[0]
        obj_path = os.path.join(OBJECT_PATH_RAW, obj_ply)
        obj_mesh = trimesh.load(obj_path)
        os.makedirs(os.path.join(OBJECT_PATH, obj_name), exist_ok=True)
        obj_mesh.export(os.path.join(OBJECT_PATH, obj_name, f"{obj_name}.obj"))

# process sequences
for sub_id in ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10']:
    motion_names = os.listdir(os.path.join(motion_folder,sub_id))
    for name in motion_names:
        sub_name = name.split('.')[0]
        obj_name = sub_name.split('_')[0]
        motion_file = os.path.join(motion_folder,sub_id,name)
        if not os.path.isfile(motion_file):
            continue
        print(sub_id,sub_name)
        seq_data = parse_npz(motion_file)
        n_comps = seq_data['n_comps']
        gender = seq_data['gender']
        sbj_id = seq_data['sbj_id']
        obj= seq_data['object']
        body = seq_data['body']
        obj_params = obj['params']
        body_params = body['params']
        obj_angles = obj_params['global_orient']
        obj_trans = obj_params['transl']
        global_orient = body_params['global_orient']
        transl = body_params['transl']
        body_pose = body_params['body_pose']
        left_hand_pose = body_params['left_hand_pose']
        right_hand_pose = body_params['right_hand_pose']

        T = seq_data.n_frames

        sbj_vtemp = load_sbj_verts(sbj_id, seq_data)

        smpl_model = smplx.create( 
            model_path=model_folder,
            model_type='smplx',
            gender=gender,
            num_pca_comps=n_comps,
            v_template = sbj_vtemp,
            batch_size=T).cuda()
        sbj_parms = params2torch(seq_data.body.params)
        smplx_output = smpl_model(**sbj_parms)

        pelvis = to_cpu(smplx_output.joints)[:, 0, :]
        rotvecs = to_cpu(sbj_parms['global_orient'])
        rotations = Rotation.from_rotvec(rotvecs)
        rotation_matrix_x = Rotation.from_euler('x', -np.pi/2, degrees=False)
        # Apply the rotation to the batch of rotations
        rotated_rotations = rotation_matrix_x * rotations
        # Convert the rotated rotations back to rotation vectors
        global_orient = rotated_rotations.as_rotvec()
        sbj_parms['global_orient'] = to_cuda(global_orient)
        trans = rotation_matrix_x.apply(to_cpu(sbj_parms['transl']))
        sbj_parms['transl'] = to_cuda(trans)
        
        rotvecs2 = obj_angles
        rotations2 = Rotation.from_rotvec(rotvecs2)

        # Apply the rotation to the batch of rotations
        rotated_rotations2 = rotation_matrix_x * rotations2
        # Convert the rotated rotations back to rotation vectors
        obj_angles = rotated_rotations2.as_rotvec()
        obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis)



        smplx_output = smpl_model(**sbj_parms)
        verts = to_cpu(smplx_output.vertices) 
        pelvis = to_cpu(smplx_output.joints)[:, 0, :]
        faces = smpl_model.faces
        obj_trans = pelvis + obj_trans_delta
        mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
        obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

        angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
        obj_verts = (obj_verts)[None, ...]
        obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
        diff_fix = verts[:30, ..., 1].min()
        trans[..., 1] -= diff_fix
        obj_trans[..., 1] -= diff_fix
       
        save_name = f"{sub_id}_{sub_name}"
        poses = np.concatenate([global_orient, body_pose, left_hand_pose, right_hand_pose], axis=1)
        # desample from 120 to 30
        poses = poses[::4]
        trans = trans[::4]
        obj_angles = obj_angles[::4]
        obj_trans = obj_trans[::4]
        human = {
            'poses': poses,
            'trans': trans,
            'vtemp': sbj_vtemp,
            'gender': gender,
        }
       
        obj = {
            'angles': obj_angles,
            'trans': obj_trans,
            'name': obj_name,
        }
        os.makedirs(os.path.join(MOTION_PATH, save_name), exist_ok=True)
        np.savez(os.path.join(MOTION_PATH, save_name ,'human.npz'), **human)
        np.savez(os.path.join(MOTION_PATH, save_name , 'object.npz'), **obj)
        print(f"Saved {save_name}")

