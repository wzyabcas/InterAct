import sys
import os
import shutil
sys.path.append(os.getcwd())
from phc.utils import torch_utils
import torch
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from uhc.smpllib.smpl_mujoco import SMPLH_BONE_ORDER_NAMES, SMPLH_SEGMENT, smplx_vert_segmentation, smpl_vert_segmentation
from uhc.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState
import trimesh
import smplx
from human_body_prior.body_model.body_model import BodyModel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_cpu(tensor):
    """Convert tensor to CPU tensor, handling both PyTorch tensors and numpy arrays."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    elif isinstance(tensor, np.ndarray):
        return torch.from_numpy(tensor).cpu()
    else:
        return torch.tensor(tensor).cpu()
import torch.nn.functional as F
import argparse
from copy import copy

MODEL_PATH = "../models"

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

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
    # print(params.keys())
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

sbj_info = {}
def load_sbj_verts(sbj_id, seq_data, data_root_folder = '../data/grab/'):

    mesh_path = os.path.join(data_root_folder,seq_data.body.vtemp)
    if sbj_id in sbj_info:
        sbj_vtemp = sbj_info[sbj_id]
    else:
        sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
        sbj_info[sbj_id] = sbj_vtemp
    return sbj_vtemp

def generate_urdf(dataset_name, object_path, object_name, output_dir="../intermimic/data/assets/objects"):
    template = '''<?xml version="1.0" ?>
<robot name="{object_name}.urdf">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="baseLink">
    <contact>
      <lateral_friction value="0.9"/>
      <rolling_friction value="0.5"/>
      <stiffness value="30000"/>
      <damping value="1000"/>
    </contact>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="hoi/{object_path}/{object_name}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
       <material name="mat">
        <color rgba="0.7 0.8 0.9 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="hoi/{object_path}/{object_name}.obj" scale="1.0 1.0 1.0"/>
      </geometry>
    </collision>
  </link>
</robot>'''
    
    content = template.format(object_path=object_path, object_name=object_name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{object_name}.urdf")
    with open(output_path, 'w') as f:
        f.write(content)


@torch.inference_mode()
def min_dist_blockwise(vertices, obj_verts_all, chunk_O=4096, use_half=True):
    """
    vertices:      [T, V, 3]  torch tensor (CPU or CUDA)
    obj_verts_all: [T, O, 3]  same device/dtype
    returns:       [T, V]     per-vertex min distance to any object point per frame
    """
    dev = vertices.device
    dt  = torch.float16 if (use_half and vertices.dtype == torch.float32 and dev.type == "cuda") else vertices.dtype

    V_all = vertices.to(dtype=dt).contiguous()
    O_all = obj_verts_all.to(dtype=dt).contiguous()

    T, V, _ = V_all.shape
    _, O, _ = O_all.shape

    out = torch.full((T, V), float("inf"), device=dev, dtype=dt)

    # less memory, loop over time
    for t in range(T):
        v = V_all[t:t+1]                  # [1, V, 3]
        best = torch.full((V,), float("inf"), device=dev, dtype=dt)
        for j in range(0, O, chunk_O):
            o = O_all[t:t+1, j:j+chunk_O] # [1, chunk, 3]
            # cdist returns [B,V,chunk]
            d = torch.cdist(v, o, p=2)    # [1, V, chunk]
            best = torch.minimum(best, d.amin(dim=2).squeeze(0))
            del d
        out[t] = best

    # return in float32 for downstream math
    return out.to(torch.float32)


def local_rotation_to_dof_vel(local_rot0, local_rot1, dt):
    # Assume each joint is 3dof
    diff_quat_data = torch_utils.quat_mul(torch_utils.quat_conjugate(local_rot0), local_rot1)
    diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
    dof_vel = diff_axis * diff_angle.unsqueeze(-1) / dt

    return dof_vel[1:, :].flatten()


def compute_motion_dof_vels(motion):
    num_frames = motion.tensor.shape[0]
    dt = 1.0 / motion.fps
    dof_vels = []

    for f in range(num_frames - 1):
        local_rot0 = motion.local_rotation[f]
        local_rot1 = motion.local_rotation[f + 1]
        frame_dof_vel = local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
        dof_vels.append(frame_dof_vel)

    dof_vels.append(dof_vels[-1])
    dof_vels = torch.stack(dof_vels, dim=0).view(num_frames, -1, 3)

    return dof_vels


def _local_rotation_to_dof_smpl(local_rot):
    B, J, _ = local_rot.shape
    dof_pos = torch_utils.quat_to_exp_map(local_rot[:, 1:])
    return dof_pos.reshape(B, -1)
    


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

def decode_pca_pose(pca_pose, gender, model_type, num_pca_comps):
    if model_type == 'smplx':
        smplx_model = smplx12[gender]
        global_orient = pca_pose[:, :3]
        body_pose = pca_pose[:, 3:66]
        left_hand_pca = pca_pose[:, 66:66 + num_pca_comps]
        right_hand_pca = pca_pose[:, 66 + num_pca_comps:66 + num_pca_comps * 2]
        left_hand_pose = left_hand_pca.float() @ smplx_model.left_hand_components.float() + smplx_model.left_hand_mean.float()[None]
        right_hand_pose = right_hand_pca.float() @ smplx_model.right_hand_components.float() + smplx_model.right_hand_mean.float()[None]
        full_pose = torch.cat([global_orient, body_pose, left_hand_pose, right_hand_pose], dim=1)
    return full_pose

def forward_smpl(poses, betas, trans, gender, model_type, num_betas, use_pca=False):
    """
    Load and visualize SMPL data for a single sequence
    """
    frame_times = poses.shape[0]
    
    # Convert trans to tensor if it's numpy, otherwise ensure it's float tensor
    if isinstance(trans, np.ndarray):
        trans_tensor = torch.from_numpy(trans).float()
    elif isinstance(trans, torch.Tensor):
        trans_tensor = trans.float()
    else:
        trans_tensor = torch.tensor(trans).float()
    
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            smplx_output = smpl_model(body_pose=poses[:, 3:66].float(),
                    global_orient=poses[:, :3].float(),
                    left_hand_pose=poses[:, 66:111].float(),
                    right_hand_pose=poses[:, 111:156].float(),
                    betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                    transl=trans_tensor,) 
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
                smplx_output = smpl_model(body_pose=poses[:, 3:66].float(),
                        global_orient=poses[:, :3].float(),
                        left_hand_pose=poses[:, 66:78].float(),
                        right_hand_pose=poses[:, 78:90].float(),
                        jaw_pose=torch.zeros(frame_times, 3).float(),
                        leye_pose=torch.zeros(frame_times, 3).float(),
                        reye_pose=torch.zeros(frame_times, 3).float(),
                        expression=torch.zeros(frame_times, 10).float(),
                        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                        transl=trans_tensor,)
            else:
                smpl_model = smplx10[gender]
                smplx_output = smpl_model(body_pose=poses[:, 3:66].float(),
                        global_orient=poses[:, :3].float(),
                        left_hand_pose=poses[:, 66:111].float(),
                        right_hand_pose=poses[:, 111:156].float(),
                        jaw_pose = torch.zeros([frame_times,3]).float(),
                        reye_pose = torch.zeros([frame_times,3]).float(),
                        leye_pose = torch.zeros([frame_times,3]).float(),
                        expression = torch.zeros([frame_times,10]).float(),
                        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                        transl=trans_tensor,)
        verts = to_cpu(smplx_output.vertices)
        joints= to_cpu(smplx_output.joints)
    elif num_betas == 16: 
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        smplx_output = smpl_model(pose_body=poses[:, 3:66].float(), 
                pose_hand=poses[:, 66:156].float(), 
                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
                root_orient=poses[:, :3].float(), 
                trans=trans_tensor)
        
        verts = to_cpu(smplx_output.v)
        joints= to_cpu(smplx_output.Jtr)

    return verts, joints

def main(args):
    dataset_name_full = args.dataset_name
    # Extract base dataset name (everything before first underscore if exists)
    dataset_name = dataset_name_full.split('_')[0]
    MOTION_PATH = f"../data/{dataset_name_full}/sequences_canonical"
    OBJECT_PATH = f"../data/{dataset_name_full}/objects"
    data_name = sorted(os.listdir(MOTION_PATH))
    os.makedirs(f"intermimic/data/assets/objects/{dataset_name.lower()}", exist_ok=True)
    target_objects_dir = os.path.join("intermimic/data/assets/objects", dataset_name.lower())
    if not os.path.exists(target_objects_dir):
        os.makedirs(target_objects_dir, exist_ok=True)

    if os.path.exists(OBJECT_PATH):
        for obj_name in os.listdir(OBJECT_PATH):
            dst_obj_dir = os.path.join(target_objects_dir)
            generate_urdf(dataset_name.lower(), OBJECT_PATH, obj_name, dst_obj_dir)

    print(data_name)
    double = False

    amass_remove_data = []

    full_motion_dict = {}

    render = False
    gender_betas = []
    skeleton_trees = []

    if dataset_name.upper() == 'BEHAVE':
        verts_number = 6890
        vert_segmentation = smpl_vert_segmentation
        model_type = 'smplh'
    elif dataset_name.upper() == 'GRAB':
        verts_number = 10475
        vert_segmentation = smplx_vert_segmentation
        model_type = 'smplx'
    elif dataset_name.upper() == 'OMOMO':
        verts_number = 10475
        vert_segmentation = smplx_vert_segmentation
        model_type = 'smplx'
    elif dataset_name.upper() == 'NEURALDOME' or dataset_name.upper() == 'IMHD':
        verts_number = 6890
        vert_segmentation = smpl_vert_segmentation
        model_type = 'smplh'
    elif dataset_name.upper() == 'INTERCAP':
        verts_number = 10475
        vert_segmentation = smplx_vert_segmentation
        model_type = 'smplx'
    joints2verts = torch.zeros((52, verts_number), dtype=torch.bool)
    for i, name in enumerate(SMPLH_BONE_ORDER_NAMES):
        verts_list = vert_segmentation[SMPLH_SEGMENT[name]]
        joints2verts[i, verts_list] = True
    joints2verts = joints2verts.to(device)

    
    for k, name in tqdm(enumerate(data_name)):
        sub = name.split('_')[0]
        with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
            obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
        if dataset_name.upper() == 'GRAB':
            # Modified to work with processed GRAB format (from process_grab.py)
            with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
                poses = f['poses']
                vtemp = f['vtemp']
                trans = f['trans']
                gender = str(f['gender'])

            n_comps = 24  # GRAB uses 24 PCA components for hands
            T = len(poses)
            sbj_vtemp = vtemp  # Use vtemp directly

            smpl_model = smplx.create(
                model_path=MODEL_PATH,
                model_type='smplx',
                gender=gender,
                num_pca_comps=n_comps,
                v_template=sbj_vtemp,
                batch_size=T).cuda()

            # Forward pass to get vertices and joints
            smplx_output = smpl_model(
                body_pose=torch.from_numpy(poses[:, 3:66]).float().cuda(),
                global_orient=torch.from_numpy(poses[:, :3]).float().cuda(),
                left_hand_pose=torch.from_numpy(poses[:, 66:90]).float().cuda(),
                right_hand_pose=torch.from_numpy(poses[:, 90:114]).float().cuda(),
                transl=torch.from_numpy(trans).float().cuda(),
            )

            vertices = to_cpu(smplx_output.vertices)
            joints = to_cpu(smplx_output.joints)
            beta = smpl_model.betas[0].detach().cpu().numpy()

            # Expand poses to 156 dimensions (add zeros for full hand pose)
            # poses format: global_orient(3) + body_pose(63) + left_hand_pca(24) + right_hand_pca(24) = 114
            # Need to decode PCA to full hand pose (45 each) for compatibility
            left_hand_pca = torch.from_numpy(poses[:, 66:90]).float()
            right_hand_pca = torch.from_numpy(poses[:, 90:114]).float()
            left_hand_full = left_hand_pca @ smpl_model.left_hand_components[:24].cpu() + smpl_model.left_hand_mean.cpu()
            right_hand_full = right_hand_pca @ smpl_model.right_hand_components[:24].cpu() + smpl_model.right_hand_mean.cpu()

            pose_aa = np.concatenate([
                poses[:, :66],  # global_orient + body_pose
                left_hand_full.numpy(),
                right_hand_full.numpy()
            ], axis=1)

            root_trans = trans.copy()
            smpl_data_entry = {'gender': gender, 'fps': 30.0}
            grab_n_comps = n_comps

        else:
            motion_file = None
            sbj_vtemp = None
            with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
                smpl_data_entry = dict(f)

            start, end = 0, 0

            pose_aa = smpl_data_entry['poses'].copy()[start:].astype(np.float64)
            root_trans = smpl_data_entry['trans'].copy()[start:].astype(np.float64)

            beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy().astype(np.float64)
            if len(beta.shape) == 2:
                beta = beta[0]
        batch_size = pose_aa.shape[0]


        angle = np.pi / 2

        # Create a rotation matrix for the X-axis rotation
        rotation_matrix_x = sRot.from_euler('x', angle, degrees=False)

        # Convert pose_aa to tensor if it's not already
        if isinstance(pose_aa, torch.Tensor):
            pose_aaa = pose_aa
        else:
            pose_aaa = torch.from_numpy(pose_aa)
        print("beta shape: ", beta.shape)
        if dataset_name.upper() == 'BEHAVE':
            vertices, joints = forward_smpl(pose_aaa, beta, root_trans, str(smpl_data_entry['gender']), 'smplh', 10)
            flat_hand_mean = False
        elif dataset_name.upper() == 'NEURALDOME' or dataset_name.upper() == 'IMHD':
            vertices, joints = forward_smpl(pose_aaa, beta, root_trans, str(smpl_data_entry['gender']), 'smplh', 16)
            flat_hand_mean = False
        elif dataset_name.upper() == 'INTERCAP':
            vertices, joints = forward_smpl(pose_aaa, beta, root_trans, str(smpl_data_entry['gender']), 'smplx', 10, True)
            flat_hand_mean = True
            pose_aa = decode_pca_pose(pose_aaa, str(smpl_data_entry['gender']), 'smplx', 12).numpy()
        elif dataset_name.upper() == 'OMOMO':
            flat_hand_mean = False
            vertices, joints = forward_smpl(pose_aaa, beta, root_trans, str(smpl_data_entry['gender']), 'smplx', 16)
        elif dataset_name.upper() == 'GRAB':
            # vertices, joints = forward_smpl(pose_aaa, beta, root_trans, str(smpl_data_entry['gender']), 'smplx', 10)
            flat_hand_mean = False
        
        joint_number = int(pose_aa.shape[1] / 3)
        print("pose_aa shape: ", pose_aa.shape)
        print("joints number: ", joint_number)
        print("verts shape: ", vertices.shape, "joints shape: ", joints.shape)
        pelvis = joints[:, 0].detach().clone()
        mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')


        obj_verts = mesh_obj.vertices
        center = np.mean(obj_verts, 0)

        obj_verts_all = []

        object_points, object_faces = mesh_obj.sample(1024, return_index=True)

        for t in range(pose_aa.shape[0]):
            # print(record['smplfit_params'])
            angle, trans = obj_angles[t], obj_trans[t]
            rot = sRot.from_rotvec(angle).as_matrix()

            mesh_obj_v = np.matmul(object_points, rot.T) + trans
            obj_verts_all.append(mesh_obj_v)

        obj_verts_all = torch.from_numpy(np.array(obj_verts_all)).to(device)
        vertices = vertices.to(device)
        joints = joints.to(device)

        ground_height = min(torch.min(obj_verts_all[:, :, 1]), torch.min(vertices[:, :, 1]))

        is_ground = (torch.min(obj_verts_all[:, :, 1], dim=1)[0] - ground_height) < 0.1
        
        is_static = torch.from_numpy(obj_trans[1:] - obj_trans[:-1]).norm(dim=1) < 0.005
        is_static = torch.cat([is_static[0:1], is_static]).to(device)

        is_falling = (torch.from_numpy((obj_trans[2:] - obj_trans[1:-1]) * 30 - (obj_trans[1:-1] - obj_trans[0:-2]) * 30).to(device) * 30 - torch.tensor([0, 9.8, 0], device=device).unsqueeze(0)).norm(dim=-1) < 6
        # 7, 10, 8, 11
        is_falling = torch.cat([is_falling[0:2], is_falling]).to(device)
        contact_part_label = []
        thres_contact = 0.02
        thres_non_contact = 0.1
        left_foot = joints[:, 10]
        right_foot = joints[:, 11]
        # Make sure tensors are on GPU and contiguous
        verts = vertices.to(device, dtype=torch.float32).contiguous()        # [T,V,3]
        objs  = obj_verts_all.to(device, dtype=torch.float32).contiguous()   # [T,O,3]

        min_dis_v = min_dist_blockwise(verts, objs, chunk_O=4096,use_half=True)  # [T,V]
        min_dis   = min_dis_v.amin(dim=1)  # [T]

        is_contact = min_dis < thres_contact
        is_far     = min_dis > 0.1


        is_ground_not_static = torch.logical_and(is_ground, torch.logical_not(is_static))
        is_contact = torch.logical_or(is_contact, torch.logical_and(torch.logical_not(is_ground), torch.logical_not(is_falling)))
        is_contact = torch.logical_and(is_contact, torch.logical_not(is_far))

        # if torch.logical_and(is_far, torch.logical_and(is_static, torch.logical_not(is_ground))).any():
        #     print("is_far: ", is_far)
        #     continue
        # Reuse the pre-computed distances from GPU
        # dis shape: [T, V, O], min_dis_v shape: [T, V]
        is_contact = is_contact.double()  # Convert to double for later use
        ground_height_gpu = torch.tensor(ground_height, device=device)
        
        for t in range(pose_aa.shape[0]):
            # Reuse pre-computed min_dis_v for this frame
            min_dis_v_t = min_dis_v[t]  # [V]
            
            if is_contact[t] > 0:
                thres_contact = min_dis_v_t.min() + 5e-3
            else:
                thres_contact = 0.01

            verts_contact = (min_dis_v_t < thres_contact).unsqueeze(0)
            verts_non_contact = (min_dis_v_t > thres_non_contact).unsqueeze(0)
            verts_not_on_ground = vertices[t, :, 1] - ground_height_gpu > 0.1

            not_on_ground = (verts_not_on_ground * joints2verts).sum(dim=-1) == joints2verts.sum(dim=-1)

            if t > 0:
                delta_left = torch.norm(left_foot[t, [0, 2]] - left_foot[t-1, [0, 2]])
                delta_right = torch.norm(right_foot[t, [0, 2]] - right_foot[t-1, [0, 2]]) 
                left_static = (delta_left < 0.02)
                right_static = (delta_right < 0.02)
                if left_static == False and right_static == False:
                    if delta_left > delta_right:
                        right_static = True
                    else:
                        left_static = True
                if not left_static:
                    not_on_ground[7] = True
                    not_on_ground[10] = True
                if not right_static:
                    not_on_ground[8] = True
                    not_on_ground[11] = True

            contact = torch.any(verts_contact * joints2verts, dim=-1).float()
            non_contact = (verts_non_contact * joints2verts).sum(dim=-1) == joints2verts.sum(dim=-1)

            if not_on_ground[7] == False or not_on_ground[10] == False or contact[7] > 0 or contact[10] > 0:
                non_contact[7] = False
                non_contact[10] = False

            if not_on_ground[8] == False or not_on_ground[11] == False or contact[8] > 0 or contact[11] > 0:
                non_contact[8] = False
                non_contact[11] = False

            contact[non_contact == True] = -1
            contact_part_label.append(contact.cpu())
        contact_part_label = torch.stack(contact_part_label)
        rotvecs = pose_aa[:, :3]
        rotations = sRot.from_rotvec(rotvecs)

        # Apply the rotation to the batch of rotations
        rotated_rotations = rotation_matrix_x * rotations
        # Convert the rotated rotations back to rotation vectors
        pose_aa[:, :3] = rotated_rotations.as_rotvec()

        obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis.cpu().numpy())
        obj_joints_delta = rotation_matrix_x.apply((obj_trans[..., None, :] - joints.detach().cpu().numpy()).reshape(-1, 3)).reshape(batch_size, -1, 3)
        root_trans = rotation_matrix_x.apply(root_trans)

        rotvecs2 = obj_angles
        rotations2 = sRot.from_rotvec(rotvecs2)

        # Apply the rotation to the batch of rotations
        rotated_rotations2 = rotation_matrix_x * rotations2
        # Convert the rotated rotations back to rotation vectors
        obj_angles = rotated_rotations2.as_rotvec()

        B = pose_aa.shape[0]

        gender = smpl_data_entry.get("gender", "neutral")
        fps = smpl_data_entry.get("fps", 30.0)

        if isinstance(gender, np.ndarray):
            gender = gender.item()

        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        # print(gender)
        if gender == "neutral":
            gender_number = [0]
        elif gender == "male":
            gender_number = [1]
        elif gender == "female":
            gender_number = [2]
        else:
            import ipdb
            ipdb.set_trace()
            raise Exception("Gender Not Supported!!")
        
        # TODO: depending on smplx or smplh, smplx should use mujoco_new
        smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 14, 17, 19, 21,
        37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
        smpl_2_mujoco_new = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 
        25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        14, 17, 19, 21,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
        # if dataset_name.upper() == 'GRAB':
        #     pose_aa_mj = pose_aa.reshape(-1, joint_number, 3)[..., smpl_2_mujoco_new, :].copy()
        #     contact_part_label = contact_part_label[:, smpl_2_mujoco_new].clone()
        # else:
        pose_aa_mj = pose_aa.reshape(-1, joint_number, 3)[..., smpl_2_mujoco, :].copy()
        contact_part_label = contact_part_label[:, smpl_2_mujoco].clone()
        

        num = 1
        
        robot_cfg = {
        "mesh": False,
        "model": model_type,
        "gender": str(smpl_data_entry['gender']),
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "beta": beta,
        "flat_hand_mean": flat_hand_mean,
        }

        smpl_local_robot = LocalRobot(
            robot_cfg,
            data_dir=f"{MODEL_PATH}/{model_type}",
            sbj_vtemp=sbj_vtemp,
        )
        if double:
            num = 2
        for idx in range(num):
            pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, joint_number, 4)


            smpl_local_robot.load_from_skeleton(betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None)
            smpl_local_robot.write_xml("intermimic/data/assets/smplx/{}_{}_{}.xml".format(model_type, dataset_name, sub))
            skeleton_tree = SkeletonTree.from_mjcf("intermimic/data/assets/smplx/{}_{}_{}.xml".format(model_type, dataset_name, sub))

            root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

            new_sk_state = SkeletonState.from_rotation_and_root_translation(
                skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                torch.from_numpy(pose_quat),
                root_trans_offset,
                is_local=True)

            if robot_cfg['upright_start']:
                pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).cpu().numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...

                new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
                pose_quat = new_sk_state.local_rotation.cpu().numpy()

                ############################################################
                # key_name_dump = key_name + f"_{idx}"
                key_name_dump = name

                if idx == 1:
                    left_to_right_index = [0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18]
                    pose_quat_global = pose_quat_global[:, left_to_right_index]
                    pose_quat_global[..., 0] *= -1
                    pose_quat_global[..., 2] *= -1

                    root_trans_offset[..., 1] *= -1
                ############################################################
            
            obj_angles_quat = (rotated_rotations2).as_quat().reshape(B, 4)
            new_motion_out = {}
            new_motion_out['pose_quat_global'] = pose_quat_global
            new_motion_out['pose_quat'] = pose_quat
            new_motion_out['trans_orig'] = root_trans
            new_motion_out['root_trans_offset'] = root_trans_offset
            new_motion_out['beta'] = beta
            new_motion_out['gender'] = gender
            new_motion_out['pose_aa'] = pose_aa
            new_motion_out['fps'] = fps

            # print(new_sk_state)
            trans = new_sk_state.global_translation[:, 0, :].detach().clone()
            pose_aa = torch.from_numpy(pose_aa)
            if dataset_name.upper() == 'BEHAVE':
                verts, joints = forward_smpl(pose_aa, beta, trans, str(smpl_data_entry['gender']), 'smplh', 10)
            elif dataset_name.upper() == 'NEURALDOME' or dataset_name.upper() == 'IMHD':
                verts, joints = forward_smpl(pose_aa, beta, trans, str(smpl_data_entry['gender']), 'smplh', 16)
            elif dataset_name.upper() == 'INTERCAP':
                verts, joints = forward_smpl(pose_aa, beta, trans, str(smpl_data_entry['gender']), 'smplx', 10, True)
            elif dataset_name.upper() == 'OMOMO':
                verts, joints = forward_smpl(pose_aa, beta, trans, str(smpl_data_entry['gender']), 'smplx', 16)
            elif dataset_name.upper() == 'GRAB':
                # GRAB uses SMPLX with PCA components (typically 12, same as INTERCAP)
                # Use PCA=True if n_comps is 12, otherwise use standard model
                print("grab_n_comps: ", grab_n_comps)
                print("beta shape: ", beta.shape)
                use_pca = (grab_n_comps == 12)
                verts, joints = forward_smpl(pose_aa, beta, trans, str(smpl_data_entry['gender']), 'smplx', 10, use_pca)
                # If using PCA, decode the pose similar to INTERCAP
                if use_pca:
                    pose_aa = decode_pca_pose(pose_aa, str(smpl_data_entry['gender']), 'smplx', grab_n_comps).numpy()

            offset = joints[:30, 0] - trans[:30]
            diff_fix = ((verts[:30] - offset[:, None])[:30, ..., -1].min(dim=-1).values).min()
            joints = joints - (joints[:, 0:1] - trans[:, None])
            joints[..., -1] -= diff_fix
            trans[..., -1] -= diff_fix
            
            global_trans = new_sk_state.global_translation.detach().clone()
            global_trans[..., -1] -= diff_fix

            full_motion_dict[key_name_dump] = new_motion_out
            gender_betas.append(torch.cat([torch.tensor([0]), torch.from_numpy(beta)]))
            skeleton_trees.append(skeleton_tree)
            data = torch.zeros((B, 331+52+52*4))

            obj_new_trans = trans + torch.from_numpy(obj_trans_delta).double()
            obj_verts = (object_points)[None, ...]
            obj_verts = torch.from_numpy(np.matmul(obj_verts, np.transpose(sRot.from_quat(obj_angles_quat).as_matrix(), (0, 2, 1)))) + obj_new_trans[:, None, :]
            
            diff = obj_verts[:1, ..., -1].min(dim=-1)[0].min()

            if diff < 0:
                obj_new_trans[..., -1] -= diff
                trans[..., -1] -= diff
                joints[..., -1] -= diff

            data[:, 0:3] = trans
            data[:, 3:7] = torch.from_numpy(pose_quat_global[:, 0, :]).double()

            data[:, 9:9+153] = _local_rotation_to_dof_smpl(torch.from_numpy(pose_quat)).double()
            if model_type == 'smplx':
                print("Joints shape: ", joints.shape)
                data[:, 162: 162+52*3] = joints[..., smpl_2_mujoco_new, :].view(B, -1).double()
            else:
                data[:, 162: 162+52*3] = joints[..., smpl_2_mujoco, :].view(B, -1).double()
            # data[:, 162: 162+52*3] = joints[..., smpl_2_mujoco_new, :].view(B, -1).double()

            data[:, 318:321] = obj_new_trans 

            data[:, 321:325] = torch.from_numpy(obj_angles_quat).double()
            data[:, 330:331] = is_contact[:, None]
            data[:, 331:331+52] = contact_part_label.double()
            data[:, 331+52:331+52+52*4] = torch.from_numpy(pose_quat_global).double().view(-1, 52*4)

            os.makedirs(f"intermimic/InterAct/{dataset_name_full}", exist_ok=True)
            file_path = f"intermimic/InterAct/{dataset_name_full}/{name}.pt"

            torch.save(data, file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=lambda s: s.lower(), default="omomo")
    args = parser.parse_args()
    main(args)