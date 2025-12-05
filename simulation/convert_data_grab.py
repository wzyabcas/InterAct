from ast import Try
import sys
import os
sys.path.append(os.getcwd())
from phc.utils.motion_lib_smpl import MotionLibSMPL
from phc.utils import torch_utils
import torch
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation as sRot
import glob
import pdb
import os.path as osp
import copy
from uhc.khrylib.utils import get_body_qposaddr
from uhc.smpllib.smpl_mujoco import SMPLH_BONE_ORDER_NAMES, SMPLH_SEGMENT, smplx_vert_segmentation
from uhc.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot
import scipy.ndimage.filters as filters
from typing import List, Optional
from tqdm import tqdm
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonMotion, SkeletonState
from render.mesh_viz import visualize_body_obj
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer
import json
import trimesh
import smplx

import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

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
    
    out_dict = copy.copy(in_dict)
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
def load_sbj_verts(sbj_id, seq_data, data_root_folder = './data/grab/'):
    
    mesh_path = os.path.join(data_root_folder,seq_data.body.vtemp)
    if sbj_id in sbj_info:
        sbj_vtemp = sbj_info[sbj_id]
    else:
        sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
        sbj_info[sbj_id] = sbj_vtemp
    return sbj_vtemp
######################################## Visualize GRAB ########################################
# def visualize_grab(name, MOTION_PATH):
#     """
#     vertices: (N, 10475, 3)
#     """
#     motion_file = os.path.join(MOTION_PATH,name,'motion.npz')
#     seq_data = parse_npz(motion_file)
#     n_comps = seq_data['n_comps']
#     gender = seq_data['gender']
#     sbj_id = seq_data['sbj_id']
#     T = seq_data.n_frames
#     sbj_vtemp = load_sbj_verts(sbj_id, seq_data, os.path.dirname(MOTION_PATH))

#     smpl_model = smplx.create( 
#         model_path="/media/volume/Physics/models",
#         model_type='smplx',
#         gender=gender,
#         num_pca_comps=n_comps,
#         v_template = sbj_vtemp,
#         batch_size=T).cuda()
#     sbj_parms = params2torch(seq_data.body.params)

#     smplx_output = smpl_model(**sbj_parms)
#     verts = to_cpu(smplx_output.vertices)
#     faces = smpl_model.faces

#     return verts, faces

behave_path = "../grab/sequences"
original_path = "/home/siruix/Downloads/GRAB-master/grab_data/grab"
OBJECT_PATH = "../grab/objects"
data_name = os.listdir(behave_path)

double = False

mujoco_joint_names = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

amass_remove_data = []

full_motion_dict = {}

render = False
gender_betas = []
skeleton_trees = []
print(len(data_name))
# 7, 10, 8, 11
# used = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
joints2verts = torch.zeros((52, 10475), dtype=torch.bool)
for i, name in enumerate(SMPLH_BONE_ORDER_NAMES):
    if i > 21:
        break
    verts_list = smplx_vert_segmentation[SMPLH_SEGMENT[name]]
    joints2verts[i, verts_list] = True
lbs_weight = False
# print(joints2verts.sum())
for k, name in tqdm(enumerate(data_name)):
    if k < 835:
        continue
    # if "s2_cup" not in name:
    #     continue
    # print(name)
    sub = name.split('_')[0]
    # # if used[int(sub[-2:]) - 1] == 1:
    # #     continue
    # if used[int(sub[-2:]) - 1] == 0:
    #     used[int(sub[-2:]) - 1] = 1
    #     print('use ' + sub)
    # else:
    #     continue
    # with np.load(os.path.join(behave_path, name, 'motion.npz'), allow_pickle=True) as f:
    #     smpl_data_entry = dict(f)
    with np.load(os.path.join(behave_path, name, 'object.npz'), allow_pickle=True) as f:
        obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
    motion_raw_file = os.path.join(original_path, sub, '_'.join(name.split('_')[1:])+'.npz')
    # beta = smpl_data_entry['beta'].copy() if "beta" in smpl_data_entry else smpl_data_entry['betas'].copy().astype(np.float64)
    # if len(beta.shape) == 2:
    #     beta = beta[0]
    # print(beta.shape)
    motion_file = os.path.join(behave_path, name, 'motion.npz')
    seq_data = parse_npz(motion_file)
    seq_raw_data = parse_npz(motion_raw_file)
    # print(seq_raw_data.keys())
    # print(seq_data['table'])
    # {'params': {'transl': array([[-0.01662397, -0.27104488,  0.94277409],
    #    [-0.01661502, -0.27102916,  0.94276189],
    #    [-0.01660489, -0.2710066 ,  0.94275875],
    #    ...,
    #    [-0.01660972, -0.27105305,  0.94277068],
    #    [-0.01659617, -0.27103417,  0.9427757 ],
    #    [-0.01660515, -0.27104627,  0.94277915]]), 'global_orient': array([[-0.00833737, -2.23789978,  2.24666819],
    #    [-0.00848573, -2.23784116,  2.24670846],
    #    [-0.00851332, -2.23783008,  2.24672583],
    #    ...,
    #    [-0.00859622, -2.23763768,  2.24704624],
    #    [-0.00864516, -2.23759843,  2.24707087],
    #    [-0.00857943, -2.23757321,  2.24710168]])}, 'table_mesh': 'tools/object_meshes/contact_meshes/table.ply'}
    n_comps = seq_data['n_comps']
    # print(n_comps)
    gender = seq_data['gender']
    sbj_id = seq_data['sbj_id']
    T = seq_data.n_frames
    sbj_vtemp = load_sbj_verts(sbj_id, seq_data, os.path.dirname(behave_path))

    smpl_model = smplx.create( 
        model_path="/home/siruix/Codes/models",
        model_type='smplx',
        gender=gender,
        num_pca_comps=n_comps,
        v_template = sbj_vtemp,
        batch_size=T).cuda()
    smpl_model_1 = smplx.create( 
        model_path="/home/siruix/Codes/models",
        model_type='smplx',
        gender=gender,
        use_pca=False,
        v_template = sbj_vtemp,
        batch_size=1)
    # print('lbs', smpl_model.lbs_weights.shape)
    if lbs_weight == False:
        joints2verts_index = smpl_model.lbs_weights.max(dim=1)[1]
        for vert in range(joints2verts_index.shape[0]):
            if joints2verts_index[vert] - 3 > 21:
                joints2verts[joints2verts_index[vert] - 3, vert] = True
        lbs_weight = True
        # for i in range(0, 52):
        #     print(i, torch.where(joints2verts[i] > 0))

    sbj_parms = params2torch(seq_data.body.params)
    sbj_parms_raw = params2torch(seq_raw_data.body.params)
    R1 = sRot.from_rotvec(sbj_parms['global_orient'][0].double().numpy())
    R2 = sRot.from_rotvec(sbj_parms_raw['global_orient'][0].double().numpy())
    T2 = seq_raw_data['object']['params']['transl'][0].astype(np.float64)
    # print(T1.shape)
    T1 = obj_trans[0].astype(np.float64)
    R12 = sRot.from_matrix(R1.as_matrix() @ np.linalg.inv(R2.as_matrix()))
    R12_matrix = R12.as_matrix()
    # R12_matrix[0, 0] = 1
    # R12_matrix[0, 1] = 0
    # R12_matrix[0, 2] = 0
    # R12_matrix[1, 0] = 0
    # R12_matrix[1, 1] = 0
    # R12_matrix[1, 2] = 1
    # R12_matrix[2, 0] = 0
    # R12_matrix[2, 1] = 1
    # R12_matrix[2, 2] = 0
    # print(R12_matrix)
    R12 = sRot.from_matrix(R12_matrix)
    T12 = T1 - R12.apply(T2)
    # print(T12)
    table_angles, table_trans = seq_raw_data['table']['params']['global_orient'][::4].astype(np.float64), seq_raw_data['table']['params']['transl'][::4].astype(np.float64)
    table_angles_matrix = sRot.from_rotvec(table_angles)
    # table_angles_matrix = R12 * table_angles_matrix
    # table_angles = table_angles_matrix.as_rotvec()
    table_angles[:,0] = 0
    table_angles[:,1] = 0
    table_angles[:,2] = -np.pi

    # table_angles_matrix = table_angles_matrix.as_matrix()
    # table_angles_matrix = np.transpose(table_angles_matrix, (0, 2, 1))
    # table_angles_matrix = sRot.from_matrix(table_angles_matrix)
    # rotation_matrix_x2 = sRot.from_euler('x', -np.pi/2, degrees=False)
    # table_angles_matrix = rotation_matrix_x2 * table_angles_matrix
    # table_angles = table_angles_matrix.as_rotvec()
    # print(seq_raw_data['object']['params'].keys())
    object_angles = seq_raw_data['object']['params']['global_orient'][::4].astype(np.float64)
    # rotation_matrix_x = Rotation.from_euler('x', np.pi/2, degrees=False)
    angle_matrix = sRot.from_rotvec(object_angles)
    angle_matrix = angle_matrix.as_matrix()
    angle_matrix = np.transpose(angle_matrix, (0, 2, 1))
    angle_matrix = sRot.from_matrix(angle_matrix)
    rotation_matrix_x2 = sRot.from_euler('x', -np.pi/2, degrees=False)
    angle_matrix = rotation_matrix_x2 * angle_matrix
    #################################################################################################################
    obj_angles = angle_matrix.as_rotvec()
    # object_angles_matrix = sRot.from_rotvec(object_angles)
    # object_angles_matrix = R12 * object_angles_matrix
    # obj_angles = object_angles_matrix.as_rotvec()
    # fix_object_angles_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    # # print(fix_object_angles_matrix)
    # # fix_object_angles_matrix[0, 0] = 1
    # # fix_object_angles_matrix[1, 1] = 1
    # # fix_object_angles_matrix[2, 2] = 1
    # fix_object_angles_matrix = sRot.from_matrix(fix_object_angles_matrix)
    # object_angles_matrix = fix_object_angles_matrix * object_angles_matrix
    # obj_angles = object_angles_matrix.as_rotvec()
    # print("table_angles", table_angles)
    # obj_angles[:, 2] = -obj_angles[:, ]
    table_trans = R12.apply(table_trans) + T12[np.newaxis, :]
    # print(sbj_parms.keys())
    B = sbj_parms['transl'].shape[0]

    # print(smpl_data_entry['gender'])


    start, end = 0, 0
    smplx_output = smpl_model(**sbj_parms)
    # verts = to_cpu(smplx_output.vertices)
    faces = smpl_model.faces
    # if str(smpl_data_entry['gender']) != 'neutral':
    #     print('deduct')
    #     beta = beta[:10]
    # print(root_trans[0], obj_trans[0], data[0, 0:3], data[0, 318: 321])
    # Define the rotation angle in radians (90 degrees)
    angle = np.pi / 2

    # Create a rotation matrix for the X-axis rotation
    rotation_matrix_x = sRot.from_euler('x', angle, degrees=False)
 
    joints = smplx_output.joints
    # print('joints', joints.shape)
    pelvis = joints[:, 0].detach().clone().double()
    vertices = smplx_output.vertices.cpu().detach().clone()

    # info_file = os.path.join(behave_path, name, 'info.json')
    # # print(joints[0, :22] - jtr_[0, :22])
    # info = json.load(open(info_file))
    # obj_name = info['cat']
    # print(seq_data.object.object_mesh)
    obj_mesh = os.path.join("/home/siruix/Downloads/GRAB-master/grab_data", seq_data.object.object_mesh)
    obj_mesh = Mesh(filename=obj_mesh)
    obj_verts = np.array(obj_mesh.vertices)
    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts = mesh_obj.vertices
    mesh_obj.vertices = mesh_obj.vertices - mesh_obj.vertices.mean(axis=0)
    obj_mesh.export(os.path.join("/home/siruix/Codes/PhysHOI/physhoi/data/assets/mjcf/objects/", f"{obj_name}/{obj_name}.obj"))
    # center = np.mean(obj_verts, 0)
    # center_all = []
    obj_verts_all = []

    object_points, object_faces = obj_mesh.sample(1024, return_index=True)
    # object_points = object_points - center
    for t in range(B):
        # print(record['smplfit_params'])
        angle, trans = obj_angles[t], obj_trans[t]
        rot = sRot.from_rotvec(angle).as_matrix()
        # transform canonical mesh to fitting
        # center_v = np.matmul(center, rot.T)
        # center_all.append(center_v)
        # transform canonical mesh to fitting
        mesh_obj_v = np.matmul(object_points, rot.T) + trans
        obj_verts_all.append(mesh_obj_v)
    # print(center)
    # center_all = np.array(center_all)
    obj_verts_all = torch.from_numpy(np.array(obj_verts_all))
    ground_height = min(torch.min(obj_verts_all[:, :, 1]), torch.min(vertices[:, :, 1])).cpu()
    # is_ground = (torch.min(obj_verts_all[:, :, 1], dim=1)[0] - ground_height) < 0.1
    # # print((ground_height - torch.max(obj_verts_all[:, :, 1], dim=1)[0])[0])
    
    # is_static = torch.from_numpy(obj_trans[1:] - obj_trans[:-1]).norm(dim=1) < 0.005
    # is_static = torch.cat([is_static[0:1], is_static])

    # is_falling = (torch.from_numpy((obj_trans[2:] - obj_trans[1:-1]) * 30 - (obj_trans[1:-1] - obj_trans[0:-2]) * 30) * 30 - torch.tensor([0, 9.8, 0]).unsqueeze(0)).norm(dim=-1) < 6
    # # 7, 10, 8, 11
    # is_falling = torch.cat([is_falling[0:2], is_falling])
    # print(is_falling[440:460], torch.from_numpy((obj_trans[2:] - obj_trans[1:-1]) * 30 - (obj_trans[1:-1] - obj_trans[0:-2]) * 30)[438:458]*30, (torch.from_numpy((obj_trans[2:] - obj_trans[1:-1]) * 30 - (obj_trans[1:-1] - obj_trans[0:-2]) * 30) * 30 - torch.tensor([0, 9.8, 0]).unsqueeze(0)).norm(dim=-1)[438:458])
    is_contact_all = []
    is_far_all = []
    contact_part_label = []
    thres_contact = 0.001
    thres_non_contact = 0.01
    left_foot = joints[:, 10]
    right_foot = joints[:, 11]
    # for t in range(B):
    #     dis = (vertices[t].unsqueeze(0) - obj_verts_all[t].unsqueeze(1)).norm(dim=-1)
    #     min_dis_v = dis.min(dim=0)[0]
    #     min_dis = min_dis_v.min(dim=-1)[0]
    #     is_contact = min_dis < thres_contact
    #     is_far = min_dis > 0.1
    #     is_contact_all.append(is_contact)
    #     is_far_all.append(is_far)

    is_contact = torch.tensor(seq_data['contact']['object']).sum(dim=-1)[::4] > 0
    # is_far = torch.tensor(is_far_all)
    # # print(is_ground[0], is_static[0], is_contact[0], is_far[0])
    # is_ground_not_static = torch.logical_and(is_ground, torch.logical_not(is_static))
    # # is_contact = torch.logical_or(is_contact, is_ground)
    # is_contact = torch.logical_or(is_contact, torch.logical_and(torch.logical_not(is_ground), torch.logical_not(is_falling)))
    # is_contact = torch.logical_and(is_contact, torch.logical_not(is_far)).double()


    for t in range(B):
        dis = (vertices[t].unsqueeze(0) - obj_verts_all[t].unsqueeze(1)).norm(dim=-1)
        min_dis_v = dis.min(dim=0)[0]
        # if is_contact[t] > 0:
        #     thres_contact = min_dis_v.min(dim=-1)[0] + 5e-3
        # else:
        #     thres_contact = 0.01
        # print(min_dis_v.shape)
        verts_contact = torch.tensor(seq_data['contact']['body'])[4 * t].unsqueeze(0) > 0
        verts_non_contact = (min_dis_v > thres_non_contact).unsqueeze(0)
        verts_not_on_ground = vertices[t, :, 1] - ground_height > 0.1

        not_on_ground = (verts_not_on_ground * joints2verts).sum(dim=-1) == joints2verts.sum(dim=-1)
        # print(not_on_ground[7], not_on_ground[10], not_on_ground[8], not_on_ground[11])
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
        # print(not_on_ground[7], not_on_ground[10], not_on_ground[8], not_on_ground[11])
        # print(verts_non_contact.sum())
        # contact_part_label.append(torch.any(verts_contact * joints2verts, dim=-1))
        contact = torch.any(verts_contact * joints2verts, dim=-1).float()
        non_contact = (verts_non_contact * joints2verts).sum(dim=-1) == joints2verts.sum(dim=-1)

        if not_on_ground[7] == False or not_on_ground[10] == False or contact[7] > 0 or contact[10] > 0:
            non_contact[7] = False
            non_contact[10] = False

        if not_on_ground[8] == False or not_on_ground[11] == False or contact[8] > 0 or contact[11] > 0:
            non_contact[8] = False
            non_contact[11] = False

        contact[non_contact == True] = -1
        contact_part_label.append(contact)
        # print(contact_part_label[-1])

    # poses = torch.tensor(poses).float()
    contact_part_label = torch.stack(contact_part_label)
    rotvecs = sbj_parms['global_orient'].double().detach().clone().cpu().numpy()
    rotations = sRot.from_rotvec(rotvecs)

    # Apply the rotation to the batch of rotations
    rotated_rotations = rotation_matrix_x * rotations
    # Convert the rotated rotations back to rotation vectors
    sbj_parms['global_orient'] = torch.tensor(rotated_rotations.as_rotvec()).cuda()

    obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis.numpy())
    table_trans_delta = rotation_matrix_x.apply(table_trans - pelvis.numpy())
    obj_joints_delta = rotation_matrix_x.apply((obj_trans[..., None, :] - joints.detach().numpy()).reshape(-1, 3)).reshape(B, -1, 3)
    root_trans = rotation_matrix_x.apply(sbj_parms['transl'].double().detach().clone().cpu().numpy()) #.cuda()
    # print(pelvis[0], (obj_trans - center - pelvis.numpy())[0], obj_trans_delta[0])
    # poses = torch.tensor(poses).float()
    rotvecs2 = obj_angles.astype(np.float64)
    rotations2 = sRot.from_rotvec(rotvecs2)

    # Apply the rotation to the batch of rotations
    rotated_rotations2 = rotation_matrix_x * rotations2
    # # Convert the rotated rotations back to rotation vectors
    # obj_angles = rotated_rotations2.as_rotvec()

    rotvecs3 = table_angles.astype(np.float64)
    rotations3 = sRot.from_rotvec(rotvecs3)

    # Apply the rotation to the batch of rotations
    rotated_rotations3 = rotation_matrix_x * rotations3
    # Convert the rotated rotations back to rotation vectors
    table_angles = rotated_rotations3.as_rotvec()
    # B = pose_aa.shape[0]

    fps = 30.0

    # vertices_curr, joints_curr = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,], trans[:frame_check])
    # offset = joints_curr[:, 0] - trans[:frame_check] # account for SMPL root offset. since the root trans we pass in has been processed, we have to "add it back".
    
    # if fix_height_mode == FixHeightMode.ankle_fix:
    #     assignment_indexes = mesh_parser.lbs_weights.argmax(axis=1)
    #     pick = (((assignment_indexes != mesh_parser.joint_names.index("L_Toe")).int() + (assignment_indexes != mesh_parser.joint_names.index("R_Toe")).int() 
    #         + (assignment_indexes != mesh_parser.joint_names.index("R_Hand")).int() + + (assignment_indexes != mesh_parser.joint_names.index("L_Hand")).int()) == 4).nonzero().squeeze()
    #     diff_fix = ((vertices_curr[:, pick] - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
    # elif fix_height_mode == FixHeightMode.full_fix:
        
    #     diff_fix = ((vertices_curr - offset[:, None])[:frame_check, ..., -1].min(dim=-1).values - height_tolorance).min()  # Only acount the first 30 frames, which usually is a calibration phase.
    
    
    
    # trans[..., -1] -= diff_fix
    if isinstance(gender, np.ndarray):
        gender = gender.item()

    if isinstance(gender, bytes):
        gender = gender.decode("utf-8")
    print(gender)
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

    smpl_2_mujoco = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 14, 17, 19, 21,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    smpl_2_mujoco_joints = [20, 
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 21,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]
    # print(smpl_2_mujoco)
    # print(pose_aa.shape)
    # pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
    left_hand_pose = torch.einsum(
        "bi,ij->bj", [sbj_parms['left_hand_pose'], smpl_model.left_hand_components]) # + smpl_model.left_hand_mean
    right_hand_pose = torch.einsum(
        "bi,ij->bj", [sbj_parms['right_hand_pose'], smpl_model.right_hand_components]) # + smpl_model.right_hand_mean
    pose_aa = torch.cat([sbj_parms['global_orient'], sbj_parms['body_pose'], left_hand_pose, right_hand_pose], dim=-1).detach().clone().cpu().numpy().astype(np.float64)
    # pose_aa[:, :3] = sbj_parms['global_orient'].detach().clone().cpu().numpy()
    pose_aa_mj = pose_aa.reshape(-1, 52, 3)[..., smpl_2_mujoco, :].copy()
    contact_part_label = contact_part_label[:, smpl_2_mujoco].clone()
    # print(pose_aa_mj.shape)
    num = 1
    robot_cfg = {
    "mesh": False,
    "model": "smplx",
    "gender": gender,
    "upright_start": True,
    "body_params": {},
    "joint_params": {},
    "geom_params": {},
    "actuator_params": {},
    }
    print(robot_cfg)
    smpl_local_robot = LocalRobot(
        robot_cfg,
        data_dir="/home/siruix/Codes/models/smplx",
    )
    if double:
        num = 2
    for idx in range(num):
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(B, 52, 4)
        # gender_number, gender = [0], "male"
        # print("using neutral model")

        smpl_local_robot.load_from_skeleton(smpl_model=smpl_model_1, gender=gender_number, objs_info=None, path=os.path.join(behave_path, name))
        smpl_local_robot.write_xml(os.path.join("/home/siruix/Codes/PhysHOI/physhoi/data/assets/smplx", '{}.xml'.format(name.split('_')[0])))
        skeleton_tree = SkeletonTree.from_mjcf(os.path.join("/home/siruix/Codes/PhysHOI/physhoi/data/assets/smplx", '{}.xml'.format(name.split('_')[0])))
        
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]
        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True)

        if robot_cfg['upright_start']:
            pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(B, -1, 4)  # should fix pose_quat as well here...
            new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
            pose_quat = new_sk_state.local_rotation.numpy().astype(np.float64)
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
        table_angles_quat = (rotated_rotations3).as_quat().reshape(B, 4)
        # new_motion_out = {}
        # new_motion_out['pose_quat_global'] = pose_quat_global
        # new_motion_out['pose_quat'] = pose_quat
        # new_motion_out['trans_orig'] = root_trans
        # new_motion_out['root_trans_offset'] = root_trans_offset
        # new_motion_out['beta'] = beta
        # new_motion_out['gender'] = gender
        # new_motion_out['pose_aa'] = pose_aa
        # new_motion_out['fps'] = fps

        trans = new_sk_state.global_translation[:, 0, :].detach().clone().cuda()
        pose_aa = torch.from_numpy(pose_aa).cuda()
        smplx_output = smpl_model(body_pose=pose_aa[:, 3:66].float(),
                    global_orient=pose_aa[:, :3].float(),
                    left_hand_pose=sbj_parms['left_hand_pose'].float(),
                    right_hand_pose=sbj_parms['right_hand_pose'].float(),
                    v_template = sbj_vtemp,
                    transl=trans[:].float(),
                    return_full_pose=True)

        verts = smplx_output.vertices
        joints = smplx_output.joints
        A = smplx_output.full_pose
        left_wrist = A[:, 20, :3, :3]
        right_wrist = A[:, 21, :3, :3]
        # print(pose_quat_global.shape, left_wrist.shape)
        # print(torch.from_numpy((sRot.from_quat(pose_quat_global[:, 0, :].reshape([-1, 4])) * sRot.from_matrix(left_wrist.detach().numpy())).as_quat()).view(B, -1, 4).shape)

        left_wrist_rot = torch_utils.quat_to_exp_map(torch.from_numpy(pose_quat_global[:, 17, :])).double()
        right_wrist_rot = torch_utils.quat_to_exp_map(torch.from_numpy(pose_quat_global[:, 36, :])).double()
        
        # pelvis = joints[:, 0].detach().clone()
        offset = joints[:30, 0] - trans[:30]
        diff_fix = ((verts[:30] - offset[:, None])[:30, ..., -1].min(dim=-1).values).min()
        joints = joints - (joints[:, 0:1] - trans[:, None])
        joints[..., -1] -= diff_fix
        trans[..., -1] -= diff_fix
        asesdf = (joints[:, 28] - joints[:, 20]).norm(dim=-1)
        skeleton_trees.append(skeleton_tree)
        data = torch.zeros((B, 331+52+52*4+7))
        trans = trans.cpu()
        obj_new_trans = trans + torch.from_numpy(obj_trans_delta).double()
        table_new_trans = trans + torch.from_numpy(table_trans_delta).double()
        # trans = trans[0:1]
        # obj_verts = (object_points)[None, ...]
        # obj_verts = torch.from_numpy(np.matmul(obj_verts, np.transpose(sRot.from_quat(obj_angles_quat).as_matrix(), (0, 2, 1)))) + obj_new_trans[:, None, :]
        
        # diff = obj_verts[:1, ..., -1].min(dim=-1)[0].min()
        # if is_ground[:1] or diff < 0:
        #     obj_new_trans[..., -1] -= diff
        #     trans[..., -1] -= diff
        #     joints[..., -1] -= diff
        # print(obj_new_trans[0])
        # obj_new_trans = obj_new_trans.mean(dim=1)

        # obj_transl_rec = torch.autograd.Variable(obj_new_trans.detach().clone().cuda(), requires_grad=True)
        # optimizer = torch.optim.Adam([obj_transl_rec],
        #                             lr=0.005)
        
        # joints_cuda = joints.detach().cuda()
        # joints_cuda.requires_grad=False
        # obj_joints_delta_cuda = torch.from_numpy(obj_joints_delta).cuda()
        # obj_joints_delta_cuda.requires_grad=False
        # for ii in range(300):
        #     optimizer.zero_grad()
        #     loss = (((obj_transl_rec[:, None] - joints_cuda).norm(dim=-1) - obj_joints_delta_cuda.norm(dim=-1))**2).sum()
        #     print(loss)
        #     loss.backward(retain_graph=False)
        #     optimizer.step()
        # print(delta.min(), delta.max(), delta.mean())
        dof_smpl_all = _local_rotation_to_dof_smpl(torch.from_numpy(pose_quat).double())
        data[:, 0:3] = joints[..., 20, :].double()
        data[:, 3:6] = left_wrist_rot
        data[:, 6:51] = dof_smpl_all[:, 17*3:32*3]

        data[:, 51:54] = joints[..., 21, :].double()
        data[:, 54:57] = right_wrist_rot
        data[:, 57:102] = dof_smpl_all[:, 36*3:51*3]

        # data[:, 3:7] = torch.from_numpy(pose_quat_global[:, 0, :]).double()# torch_utils.quat_to_exp_map(torch.from_numpy(pose_quat_global[:, 0, :]))
        # data[:, 9:9+153] = _local_rotation_to_dof_smpl(torch.from_numpy(pose_quat).double())
        # # print(data[0, 9:9+153].view(51, 3), pose_aa[0, 3:].view(51, 3))
        # # pose_hand = data[:, 9 + 21*3:9+153].view(B, 30, 3)
        # # pose_hand = pose_hand[:, :, [1, 0, 2]]
        # # data[:, 9 + 21*3:9+153] = pose_hand.view(B, 90)
        data[:, 102: 102+32*3] = joints[..., smpl_2_mujoco_joints, :].view(B, -1).double()
        data[:, 198:201] = obj_new_trans #trans + torch.from_numpy(obj_trans_delta).double()
        # data[:, 320] -= 0.05
        # data[:, 318] -= 0.05
        data[:, 201:205] = torch.from_numpy(obj_angles_quat).double()
        data[:, 205:206] = is_contact[:, None]
        contact_part_label_hand = torch.cat([contact_part_label[:, 17:33], contact_part_label[:, 36:52]], dim=1)
        data[:, 206:206+32] = contact_part_label_hand.double()
        # data[:, 331+52:331+52+52*4] = torch.from_numpy(pose_quat_global).double().view(-1, 52*4)# torch_utils.quat_to_exp_map(torch.from_numpy(pose_quat_global[:, 0, :]))
        data[:, 238:241] = table_new_trans #trans + torch.from_numpy(obj_trans_delta).double()
        # data[:, 320] -= 0.05
        # data[:, 318] -= 0.05
        data[:, 241:245] = torch.from_numpy(table_angles_quat).double()
        print(pose_quat_global.shape)
        data[:, 245:245+32*4] = torch.from_numpy(np.concatenate([pose_quat_global[..., 17:33, :], pose_quat_global[..., 36:52, :]], axis=-2)).double().view(-1, 32*4)# torch_utils.quat_to_exp_map(torch.from_numpy(pose_quat_global[:, 0, :]))

        # data[:, 318:321] = data[:, 318:321] - diff_fix # + skeleton_tree.local_translation[0]
        # Specify the file path where you want to save the dictionary
        file_path = os.path.join(behave_path, name, 'interaction_hand.pt')
        # print(obj_trans - root_trans, data[:, 318:321] - data[:, 0:3])
        # Save the dictionary to a .pt file
        # joblib.dump(full_motion_dict, "data/behave/pkls/{}.pkl".format(key_name_dump))
        
        # smplx_output = smpl_model(body_pose=pose_aa[:, 3:66].float(),
        #             global_orient=pose_aa[:, :3].float(),
        #             L_hand_pose=pose_aa[:, 66:111].float(),
        #             R_hand_pose=pose_aa[:, 111:156].float(),
        #             betas=torch.from_numpy(beta[None,]).repeat(batch_size, 1).float(),
        #             transl=trans[:].float(),)

        # verts = smplx_output.vertices
        # joints = smplx_output.joints
        # if render:
        #     info_file = os.path.join(behave_path, name, 'info.json')
        #     # print(joints[0, :22] - jtr_[0, :22])
        #     info = json.load(open(info_file))
        #     gender = info['gender']
        # mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
        # # mesh_obj.load_from_obj(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"))

        # obj_verts = mesh_obj.vertices
        # obj_faces = mesh_obj.faces
        # center = np.mean(obj_verts, 0)
        # obj_verts = obj_verts - center
        
        # obj_verts_all = []
        # for t in range(pose_aa.shape[0]):
        #     # print(record['smplfit_params'])
        #     angle, trans = obj_angles_quat[t], data[t, 318:321].detach().numpy()
        #     rot = sRot.from_quat(angle).as_matrix()
        #     # transform canonical mesh to fitting
        #     mesh_obj_v = np.matmul(obj_verts, rot.T) + trans
        #     obj_verts_all.append(mesh_obj_v)
        
        # obj_verts_all = np.array(obj_verts_all)
        # print(obj_verts_all[:, :, -1].min())
        # print(root_trans[0], obj_trans[0], data[0, 0:3], data[0, 318: 321])
        # data[:, 320] -= obj_verts_all[:, :, -1].min()
        # print(root_trans[0], obj_trans[0], data[0, 0:3], data[0, 318: 321])
        torch.save(data, file_path)
        # rotation_matrix_x2 = sRot.from_euler('x', -np.pi, degrees=False)
        # visualize_body_obj(verts.detach().numpy(), smpl_model.faces, np.array(obj_verts_all), obj_faces, 
        #         save_path="data/behave/gifs/{}_all_smplx.gif".format(name))
        
    # joblib.dump(full_motion_dict, "data/behave/pkls/behave.pkl")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# motionlib = MotionLibSMPL("data/behave/pkls/behave.pkl", device=device, multi_thread=False)

# motionlib.load_motions(skeleton_trees, gender_betas, limb_weights=np.zeros([1]))
# state = motionlib.get_motions(torch.tensor([0]))
# print(motionlib.get_motion_length())
# print((state['root_pos'] - data[:, 0:3].to(device)).abs().sum())

        # return {
        #     "root_pos": rg_pos[..., 0, :].clone(),
        #     "root_rot": rb_rot[..., 0, :].clone(),
        #     "dof_pos": dof_pos.clone(),
        #     "root_vel": body_vel[..., 0, :].clone(),
        #     "root_ang_vel": body_ang_vel[..., 0, :].clone(),
        #     "dof_vel": dof_vel.view(dof_vel.shape[0], -1),
        #     "rg_pos": rg_pos,
        #     "rb_rot": rb_rot,
        #     "body_vel": body_vel,
        #     "body_ang_vel": body_ang_vel,
        #     "motion_bodies": self._motion_bodies[motion_ids],
        #     "motion_limb_weights": self._motion_limb_weights[motion_ids],
        # }