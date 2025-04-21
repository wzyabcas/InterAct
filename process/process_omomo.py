import os
import os.path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import trimesh
import joblib
import torch
from human_body_prior.body_model.body_model import BodyModel
import pytorch3d.transforms as transforms 


MOTION_PATH = './data/omomo/sequences'
OBJECT_PATH = './data/omomo/objects'
MOTION_PATH_RAW = './data/omomo/raw/train_diffusion_manip_seq_joints24.p'
MOTION_PATH_RAW_TEST = './data/omomo/raw/test_diffusion_manip_seq_joints24.p'
OBJECT_PATH_RAW = './data/omomo/raw/captured_objects'

SMPLX_PATH = './models/smplx'
SMPLH_PATH = './models/smplh'

data_dict = joblib.load(MOTION_PATH_RAW)
data_dict_test = joblib.load(MOTION_PATH_RAW_TEST)

data_dict.update(data_dict_test)

surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")
dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 

smpl_model_male = BodyModel(bm_fname=surface_model_male_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smpl_model_female = BodyModel(bm_fname=surface_model_female_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smpl_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)

smpl = {'male': smpl_model_male, 'female': smpl_model_female, 'neutral': smpl_model_neutral}

def process(human, obj):
    poses, betas, trans, gender = human['poses'], human['betas'], human['trans'], str(human['gender'])
    obj_rot, obj_trans, obj_name = obj['rot'], obj['trans'], str(obj['name'])
    frame_times = poses.shape[0]
    smpl_model = smpl[gender]
    smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
                            pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
                            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
                            root_orient=torch.from_numpy(poses[:, :3]).float(), 
                            trans=torch.from_numpy(trans).float())
    pelvis = smplx_output.Jtr.detach().numpy()[:, 0, :]
    rotvecs = poses[:, :3]
    rotations = Rotation.from_rotvec(rotvecs)
    rotation_matrix_x = Rotation.from_euler('x', -np.pi/2, degrees=False)
    # Apply the rotation to the batch of rotations
    rotated_rotations = rotation_matrix_x * rotations
    # Convert the rotated rotations back to rotation vectors
    poses[:, :3] = rotated_rotations.as_rotvec()

    trans = rotation_matrix_x.apply(trans)

    rotations2 = Rotation.from_matrix(obj_rot)

    # Apply the rotation to the batch of rotations
    rotated_rotations2 = rotation_matrix_x * rotations2
    # Convert the rotated rotations back to rotation vectors
    obj_angles = rotated_rotations2.as_rotvec()
    obj_trans_delta = rotation_matrix_x.apply(obj_trans - pelvis)
    smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
                            pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
                            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
                            root_orient=torch.from_numpy(poses[:, :3]).float(), 
                            trans=torch.from_numpy(trans).float())
    
    verts = smplx_output.v.detach().numpy()
    pelvis = smplx_output.Jtr.detach().numpy()[:, 0, :]
    
    obj_trans = pelvis + obj_trans_delta
    
    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts = mesh_obj.vertices

    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = mesh_obj.vertices[None, ...]
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    diff = min(verts[:, :, 1].min(), obj_verts[:, :, 1].min())
    obj_trans[..., 1] -= diff
    trans[..., 1] -= diff
    

    obj = {
        'angles': np.array(obj_angles),
        'trans': np.array(obj_trans),
        'name': obj_name,
    }
    human = {
        'poses': np.array(poses),
        'betas': np.array(betas),
        'trans': np.array(trans),
        'gender': gender,
    }
    return human, obj



def get_smpl_parents(use_joints24=False):
    bm_path = os.path.join(SMPLH_PATH, 'male/model.npz')
    npz_data = np.load(bm_path)
    ori_kintree_table = npz_data['kintree_table'] # 2 X 52 

    if use_joints24:
        parents = ori_kintree_table[0, :23] # 23 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.

        parents_list = parents.tolist()
        parents_list.append(ori_kintree_table[0][37])
        parents = np.asarray(parents_list) # 24 
    else:
        parents = ori_kintree_table[0, :22] # 22 
        parents[0] = -1 # Assign -1 for the root joint's parent idx.
    
    return parents


def length(x, axis=-1, keepdims=True):
    """
    Computes vector norm along a tensor axis(axes)

    :param x: tensor
    :param axis: axis(axes) along which to compute the norm
    :param keepdims: indicates if the dimension(s) on axis should be kept
    :return: The length or vector of lengths.
    """
    lgth = np.sqrt(np.sum(x * x, axis=axis, keepdims=keepdims))
    return lgth


def normalize(x, axis=-1, eps=1e-8):
    """
    Normalizes a tensor over some axis (axes)

    :param x: data tensor
    :param axis: axis(axes) along which to compute the norm
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized tensor
    """
    res = x / (length(x, axis=axis) + eps)
    return res

def quat_normalize(x, eps=1e-8):
    """
    Normalizes a quaternion tensor

    :param x: data tensor
    :param eps: epsilon to prevent numerical instabilities
    :return: The normalized quaternions tensor
    """
    res = normalize(x, eps=eps)
    return res


def quat_ik(grot, gpos, parents):
    """
    Performs Inverse Kinematics (IK) on global quaternions and global positions to retrieve local representations

    :param grot: tensor of global quaternions with shape (..., Nb of joints, 4)
    :param gpos: tensor of global positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of local quaternion, local positions
    """
    res = [
        np.concatenate(
            [
                grot[..., :1, :],
                quat_mul(quat_inv(grot[..., parents[1:], :]), grot[..., 1:, :]),
            ],
            axis=-2,
        ),
        np.concatenate(
            [
                gpos[..., :1, :],
                quat_mul_vec(
                    quat_inv(grot[..., parents[1:], :]),
                    gpos[..., 1:, :] - gpos[..., parents[1:], :],
                ),
            ],
            axis=-2,
        ),
    ]

    return res


def quat_mul(x, y):
    """
    Performs quaternion multiplication on arrays of quaternions

    :param x: tensor of quaternions of shape (..., Nb of joints, 4)
    :param y: tensor of quaternions of shape (..., Nb of joints, 4)
    :return: The resulting quaternions
    """
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]

    res = np.concatenate(
        [
            y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
            y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
            y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
            y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
        ],
        axis=-1,
    )

    return res

def quat_between(x, y):
    """
    Quaternion rotations between two 3D-vector arrays

    :param x: tensor of 3D vectors
    :param y: tensor of 3D vetcors
    :return: tensor of quaternions
    """
    res = np.concatenate(   
        [
            np.sqrt(np.sum(x * x, axis=-1) * np.sum(y * y, axis=-1))[..., np.newaxis]
            + np.sum(x * y, axis=-1)[..., np.newaxis],
            np.cross(x, y),
        ],
        axis=-1,
    )
    return res

def quat_inv(q):
    """
    Inverts a tensor of quaternions

    :param q: quaternion tensor
    :return: tensor of inverted quaternions
    """
    res = np.asarray([1, -1, -1, -1], dtype=np.float32) * q
    return res

def quat_mul_vec(q, x):
    """
    Performs multiplication of an array of 3D vectors by an array of quaternions (rotation).

    :param q: tensor of quaternions of shape (..., Nb of joints, 4)
    :param x: tensor of vectors of shape (..., Nb of joints, 3)
    :return: the resulting array of rotated vectors
    """
    t = 2.0 * np.cross(q[..., 1:], x)
    res = x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)

    return res

def quat_fk(lrot, lpos, parents):
    """
    Performs Forward Kinematics (FK) on local quaternions and local positions to retrieve global representations

    :param lrot: tensor of local quaternions with shape (..., Nb of joints, 4)
    :param lpos: tensor of local positions with shape (..., Nb of joints, 3)
    :param parents: list of parents indices
    :return: tuple of tensors of global quaternion, global positions
    """
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(
            quat_mul_vec(gr[parents[i]], lpos[..., i : i + 1, :]) + gp[parents[i]]
        )
        gr.append(quat_mul(gr[parents[i]], lrot[..., i : i + 1, :]))

    res = np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)
    return res

def rotate_at_frame_w_obj(X, Q, obj_x, obj_q, trans2joint_list, parents, n_past=1, floor_z=False):
    """
    Re-orients the animation data according to the last frame of past context.

    :param X: tensor of local positions of shape (Batchsize, Timesteps, Joints, 3)
    :param Q: tensor of local quaternions (Batchsize, Timesteps, Joints, 4)
    :obj_x: N X T X 3
    :obj_q: N X T X 4
    :trans2joint_list: N X 3 
    :param parents: list of parents' indices
    :param n_past: number of frames in the past context
    :return: The rotated positions X and quaternions Q
    """
    # Get global quats and global poses (FK)
    global_q, global_x = quat_fk(Q, X, parents)

    key_glob_Q = global_q[:, n_past - 1 : n_past, 0:1, :]  # (B, 1, 1, 4)
    if floor_z: 
        # The floor is on z = xxx. Project the forward direction to xy plane. 
        forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  
    else: 
        # The floor is on y = xxx. Project the forward direction to xz plane. 
        forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  
        # forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
        #     key_glob_Q, np.array([0, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :]
        # ) # In rest pose, z direction is forward direction. This also works. 

    forward = normalize(forward)
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))
    new_glob_Q = quat_mul(quat_inv(yrot), global_q)
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x)

    # Process object rotation and translation 
    # new_obj_x = quat_mul_vec(quat_inv(yrot[:, 0, :, :]), obj_x)
    new_obj_q = quat_mul(quat_inv(yrot[:, 0, :, :]), obj_q)

    # Apply corresponding rotation to the object translation 
    obj_trans = obj_x + trans2joint_list[:, np.newaxis, :] # N X T X 3  
    obj_trans = quat_mul_vec(quat_inv(yrot[:, 0, :, :]), obj_trans) # N X T X 3
    obj_trans = obj_trans - trans2joint_list[:, np.newaxis, :] # N X T X 3 
    new_obj_x = obj_trans.copy()  

    # back to local quat-pos
    Q, X = quat_ik(new_glob_Q, new_glob_X, parents)

    return X, Q, new_obj_x, new_obj_q

# process the objects
if not os.path.exists(OBJECT_PATH):
    os.rename(OBJECT_PATH_RAW, OBJECT_PATH)

    for object in os.listdir(OBJECT_PATH):
        for index in data_dict:
            seq_name = data_dict[index]['seq_name']
            obj_name = seq_name.split("_")[1]
            if obj_name == object.split("_")[0]:
                print(obj_name)
                obj_scale = data_dict[index]['obj_scale']
                mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}_cleaned_simplified.obj"), force='mesh')
                mesh_obj.vertices *= obj_scale[0]
                os.makedirs(os.path.join(OBJECT_PATH, f"{obj_name}"), exist_ok=True)
                mesh_obj.export(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"))
                break

# process the sequences
for index in data_dict:
    
    seq_name = data_dict[index]['seq_name']

    object_name = seq_name.split("_")[1]

    trans2joint = data_dict[index]['trans2joint']
    rest_human_offsets = data_dict[index]['rest_offsets']
    
    betas = data_dict[index]['betas'][0] # 1 X 16 
    gender = data_dict[index]['gender']

    trans =  data_dict[index]['trans'] # T X 3 
    frame_times = len(trans)
    global_orient = data_dict[index]['root_orient'] # T X 3 
    body_pose = data_dict[index]['pose_body'].reshape(-1, 21, 3) # T X 63

    obj_trans = data_dict[index]['obj_trans'][:, :, 0] # T X 3
    obj_rot = data_dict[index]['obj_rot'] # T X 3 X 3 
    obj_angles = Rotation.from_matrix(obj_rot).as_rotvec()
    obj_scale = data_dict[index]['obj_scale'] # T X 1
    obj_com_pos = data_dict[index]['obj_com_pos'] # T X 3 


    padding_zeros_hand = np.zeros((frame_times, 90))



    joint_aa_rep = torch.cat((torch.from_numpy(global_orient).float()[:, None, :], \
                    torch.from_numpy(body_pose).float()), dim=1) # T X J X 3 
    X = torch.from_numpy(rest_human_offsets).float()[None].repeat(joint_aa_rep.shape[0], 1, 1).detach().cpu().numpy() # T X J X 3
    X[:, 0, :] = trans
    local_rot_mat = transforms.axis_angle_to_matrix(joint_aa_rep) # T X J X 3 X 3  
    Q = transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy() # T X J X 4

    obj_x = obj_trans.copy() # T X 3 
    obj_rot_mat = torch.from_numpy(obj_rot).float()# T X 3 X 3 
    obj_q = transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy() # T X 4 
    parents = get_smpl_parents()
    _, _, new_obj_x, new_obj_q = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
    obj_x[np.newaxis], obj_q[np.newaxis], \
    trans2joint[np.newaxis], parents, n_past=1, floor_z=True)
    # 1 X T X J X 3, 1 X T X J X 4, 1 X T X 3, 1 X T X 4 

    X, Q, new_obj_com_pos, _ = rotate_at_frame_w_obj(X[np.newaxis], Q[np.newaxis], \
    obj_com_pos[np.newaxis], obj_q[np.newaxis], \
    trans2joint[np.newaxis], parents, n_past=1, floor_z=True)

    new_seq_root_trans = X[0, :, 0, :] # T X 3 
    new_local_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(Q[0]).float()) # T X J X 3 X 3 
    new_local_aa_rep = transforms.matrix_to_axis_angle(new_local_rot_mat) # T X J X 3 
    new_seq_root_orient = new_local_aa_rep[:, 0, :] # T X 3
    new_seq_pose_body = new_local_aa_rep[:, 1:, :] # T X 21 X 3 
    new_obj_rot_mat = transforms.quaternion_to_matrix(torch.from_numpy(new_obj_q[0]).float())
    new_obj_trans = new_obj_x[0]
    cano_obj_mat = torch.matmul(new_obj_rot_mat[0], obj_rot_mat[0].transpose(0, 1))

    poses = np.concatenate((new_seq_root_orient, new_seq_pose_body.reshape(-1,63), padding_zeros_hand),axis=1)

    obj = {
            'rot': np.array(new_obj_rot_mat),
            'trans': np.array(new_obj_trans),
            'name': object_name,
        }


    human = {
            'poses': np.array(poses),
            'betas': np.array(betas),
            'trans': np.array(new_seq_root_trans),
            'gender': gender,
        }
    
        
    human, obj = process(human, obj)


    os.makedirs(os.path.join(MOTION_PATH, seq_name), exist_ok=True)
    np.savez(os.path.join(MOTION_PATH, seq_name,'object.npz'), **obj)
    np.savez(os.path.join(MOTION_PATH, seq_name, 'human.npz'), **human)
    print('save ' + os.path.join(MOTION_PATH, seq_name))
    
    
    
    