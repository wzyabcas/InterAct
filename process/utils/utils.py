from __future__ import annotations

import os
import os.path
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
import trimesh

def base_object_name(instance_name):
    # HUMOTO uses Blender instance suffixes such as mug.001.
    return instance_name.split(".", 1)[0]



def load_objects(objs_with_pose_path, objs_with_mesh_path, expected_frames=None) \
    -> tuple[list[str], list[str], list[trimesh.Trimesh], list[np.ndarray], list[np.ndarray]]:
    """Load every object instance from one HUMOTO sequence.

    ``objs_with_pose_path`` is the sequence's exact ``obj_pose.npz`` path,
    rather than the directory containing all 735 sequences.  Keeping this
    function sequence-local prevents human data from one sequence being paired
    with object data from another one.
    """
    objs_with_pose_path = Path(objs_with_pose_path)
    objs_with_mesh_path = Path(objs_with_mesh_path)
    if not objs_with_pose_path.is_file():
        raise FileNotFoundError(f"Missing HUMOTO object poses: {objs_with_pose_path}")

    list_instance_name, list_mesh_name, list_mesh, list_rot, list_trans = [], [], [], [], []
    with np.load(objs_with_pose_path, allow_pickle=True) as pose_data:
        if not pose_data.files:
            raise ValueError(f"No objects found in {objs_with_pose_path}")
        for instance_name in sorted(pose_data.files):
            pose = np.asarray(pose_data[instance_name], dtype=np.float32)
            if pose.ndim != 2 or pose.shape[1] != 7:
                raise ValueError(f"{instance_name}: expected (T, 7), got {pose.shape}")
            if expected_frames is not None and pose.shape[0] != expected_frames:
                raise ValueError(
                    f"{instance_name}: expected ({expected_frames}, 7), got {pose.shape}"
                )
            if not np.isfinite(pose).all():
                raise ValueError(f"{instance_name}: object pose contains NaN or Inf")

            quaternion_wxyz = pose[:, :4]
            quaternion_norm = np.linalg.norm(quaternion_wxyz, axis=1)
            if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
                raise ValueError(f"{instance_name}: non-unit object quaternion")
            quaternion_xyzw = quaternion_wxyz[:, [1, 2, 3, 0]]
            rotation = Rotation.from_quat(quaternion_xyzw).as_matrix().astype(np.float32)

            mesh_name = base_object_name(instance_name)
            mesh_path = objs_with_mesh_path / mesh_name / f"{mesh_name}.obj"
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f"{instance_name}: missing object mesh {mesh_path}"
                )
            mesh = trimesh.load(mesh_path, force="mesh", process=False)
            if not isinstance(mesh, trimesh.Trimesh):
                raise TypeError(f"Expected a mesh at {mesh_path}, got {type(mesh)}")

            list_instance_name.append(instance_name)
            list_mesh_name.append(mesh_name)
            list_mesh.append(mesh)
            list_rot.append(rotation)
            list_trans.append(pose[:, 4:7])

    return list_instance_name, list_mesh_name, list_mesh, list_rot, list_trans

def load_human(motion_path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Load and validate one HUMOTO SMPL-H parameter sequence."""
    human_path = Path(motion_path)
    if not human_path.is_file():
        raise FileNotFoundError(f"Missing HUMOTO SMPL-H file: {human_path}")

    with np.load(human_path, allow_pickle=True) as data:
        expected = {"poses", "betas", "trans", "gender"}
        if set(data.files) != expected:
            raise ValueError(f"Unexpected human fields in {human_path}: {data.files}")
        poses = np.asarray(data["poses"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
        trans = np.asarray(data["trans"], dtype=np.float32)
        gender_raw = data["gender"]
        gender = str(gender_raw.item() if gender_raw.shape == () else gender_raw.reshape(-1)[0])

    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ValueError(f"Expected poses (T, 156), got {poses.shape}")
    if trans.shape != (poses.shape[0], 3):
        raise ValueError(f"Expected trans ({poses.shape[0]}, 3), got {trans.shape}")
    if betas.shape != (10,):
        raise ValueError(f"Expected 10 SMPL-H betas, got {betas.shape}")
    if gender not in {"male", "female", "neutral"}:
        raise ValueError(f"Unsupported gender {gender!r}")
    if not np.isfinite(poses).all() or not np.isfinite(betas).all() or not np.isfinite(trans).all():
        raise ValueError(f"Human parameters contain NaN or Inf: {human_path}")

    return poses, betas, trans, gender


def process(human, obj, smpl, OBJECT_PATH):
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

def get_smpl_parents(use_joints24=False, SMPLH_PATH=None):
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

    key_glob_Q = global_q[:, n_past - 1 : n_past, 0:1, :]  # (B, 1, 1, 4), global rot for the root joint.
    if floor_z: 
        # The floor is on z = xxx. Project the forward direction to xy plane. 
        forward = np.array([1, 1, 0])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint. 
        # In all, forward doesn't mean the forward direction of the human, it's just a stable direction.
        # For example, now in this branch, it represent his left direction, because the rightmost is (1,0,0). 
    else: 
        # The floor is on y = xxx. Project the forward direction to xz plane. 
        forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
            key_glob_Q, np.array([1, 0, 0])[np.newaxis, np.newaxis, np.newaxis, :]
        ) # In rest pose, x direction is the body left direction, root joint point to left hip joint.  
        # forward = np.array([1, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :] * quat_mul_vec(
        #     key_glob_Q, np.array([0, 0, 1])[np.newaxis, np.newaxis, np.newaxis, :]
        # ) # In rest pose, z direction is forward direction. This also works. 

    forward = normalize(forward)
    # How much rotation is needed to rotate the forward direction to (1,0,0).
    # quat_inv(yrot) = correction rotation.
    yrot = quat_normalize(quat_between(np.array([1, 0, 0]), forward))   
    new_glob_Q = quat_mul(quat_inv(yrot), global_q) # rot * rot, so using quat_mul
    new_glob_X = quat_mul_vec(quat_inv(yrot), global_x) # rot * 3-dim vec, so using quat_mul_vec
    # now, both obj and human are facing the standard direction (1,0,0) in the world coordinate system.

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
