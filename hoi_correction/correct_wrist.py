# Comprehensive Pipeline: Joint Pose Fix + Palm Fix + Optimization
import sys

import math
from scipy.spatial.distance import cdist

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import smplx
import pytorch3d.loss
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix,rotation_6d_to_matrix,matrix_to_rotation_6d
from torch.autograd import Variable
import torch.optim as optim
import copy
import argparse
import csv

from scipy.spatial.transform import Rotation
import trimesh
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()
from scipy.spatial.transform import Rotation as R
from human_body_prior.body_model.body_model import BodyModel

from utils import vertex_normals
# from render.mesh_viz import visualize_body_obj
from loss import point2point_signed
from prior import *
from fix_pose import fix_joint_poses_simple, pose_delta_axis_angle
from interpolate import smooth_flips
from optimize_wrist import optimize_poses

# Joint indices
LEFT_COLLAR = 13
RIGHT_COLLAR = 14
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21
# Joint mapping to pose parameters (for SMPLX/SMPLH)
# Pose parameters start from index 3 (after global orientation 0:3)
# Each joint has 3 pose parameters
MODEL_PATH = './models'

# Load SMPL models (same as in other files) - for Steps 1 and 2
smplh10 = {}
smplx10 = {}
smplx12 = {}
smplh16 = {}
smplx16 = {}

def load_models():
    global smplh10, smplx10, smplx12, smplh16, smplx16
    
    # SMPLH 10
    smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh', gender="male", use_pca=False, ext='pkl')
    smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh', gender="female", use_pca=False, ext='pkl')
    smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh', gender="neutral", use_pca=False, ext='pkl')
    smplh10 = {'male': smplh_model_male.to(device), 'female': smplh_model_female.to(device), 'neutral': smplh_model_neutral.to(device)}
    
    # SMPLX 10
    smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx', gender='male', use_pca=False, ext='pkl')
    smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx', gender="female", use_pca=False, ext='pkl')
    smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx', gender="neutral", use_pca=False, ext='pkl')
    smplx10 = {'male': smplx_model_male.to(device), 'female': smplx_model_female.to(device), 'neutral': smplx_model_neutral.to(device)}
    
    # SMPLX 12
    smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx', gender="male", num_pca_comps=12, use_pca=True, flat_hand_mean=True, ext='pkl')
    smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx', gender="female", num_pca_comps=12, use_pca=True, flat_hand_mean=True, ext='pkl')
    smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx', gender="neutral", num_pca_comps=12, use_pca=True, flat_hand_mean=True, ext='pkl')
    smplx12 = {'male': smplx12_model_male.to(device), 'female': smplx12_model_female.to(device), 'neutral': smplx12_model_neutral.to(device)}
    
    # SMPLH 16
    SMPLH_PATH = MODEL_PATH+'/smplh'
    surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
    surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
    surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
    dmpl_fname = None
    num_dmpls = None 
    num_expressions = None
    num_betas = 16
    
    smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}
    
    # SMPLX 16
    SMPLX_PATH = MODEL_PATH+'/smplx'
    surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
    surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
    surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")
    
    smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname, num_betas=num_betas, num_expressions=num_expressions, num_dmpls=num_dmpls, dmpl_fname=dmpl_fname).to(device)
    smplx16 = {'male': smplx16_model_male.to(device), 'female': smplx16_model_female.to(device), 'neutral': smplx16_model_neutral.to(device)}

# Load models for Steps 1 and 2
load_models()

rhand_idx = np.load('./assets/smplx_hand_index/rhand_smplx_ids.npy')
lhand_idx = np.load('./assets/smplx_hand_index/lhand_smplx_ids.npy')


class FixTracker:
    """Track which frames and joints have been fixed by different methods"""
    def __init__(self, num_frames):
        self.num_frames = num_frames
        # Track which joints were fixed in which frames
        # Shape: (num_frames, num_joints) where num_joints = 8 (collar, shoulder, elbow, wrist)
        self.joint_fixed = np.zeros((num_frames, 8), dtype=bool)  # [left_collar, right_collar, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist]
        self.palm_fixed = np.zeros((num_frames, 8), dtype=bool)
        self.optimized = np.zeros((num_frames, 8), dtype=bool)
    
    def mark_joint_fixed(self, frame_indices, joint_indices):
        """Mark specific joints in specific frames as fixed by joint pose fixing"""
        for frame_idx in frame_indices:
            for joint_idx in joint_indices:
                if 0 <= joint_idx < 8:
                    self.joint_fixed[frame_idx, joint_idx] = True
    
    def mark_palm_fixed(self, frame_indices, joint_indices):
        """Mark specific joints in specific frames as fixed by palm orientation fixing"""
        for frame_idx in frame_indices:
            for joint_idx in joint_indices:
                if 0 <= joint_idx < 8:
                    self.palm_fixed[frame_idx, joint_idx] = True
    
    def mark_optimized(self, frame_indices, joint_indices):
        """Mark specific joints in specific frames as optimized"""
        for frame_idx in frame_indices:
            for joint_idx in joint_indices:
                if 0 <= joint_idx < 8:
                    self.optimized[frame_idx, joint_idx] = True
    
    def get_fixed_frames_and_joints(self):
        """Get all frames and joints that have been fixed by any method"""
        return self.joint_fixed | self.palm_fixed
    
    def get_joint_fixed_frames_and_joints(self):
        """Get frames and joints fixed by joint pose fixing"""
        return self.joint_fixed
    
    def get_palm_fixed_frames_and_joints(self):
        """Get frames and joints fixed by palm orientation fixing"""
        return self.palm_fixed
    
    def get_optimized_frames_and_joints(self):
        """Get frames and joints that have been optimized"""
        return self.optimized
    
    def mark_optimized_from_mask(self, mask):
        """Mark joints as optimized based on a boolean mask
        
        Args:
            mask: (T, 8) boolean array indicating which joint-frame combinations were optimized
        """
        self.optimized = mask.copy()
    
    def print_summary(self):
        """Print summary of fixes"""
        joint_names = ['left_collar', 'right_collar', 'left_shoulder', 'right_shoulder', 
                      'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
        
        print(f"Fix Summary:")
        print(f"  Joint pose fixed:")
        for i, name in enumerate(joint_names):
            fixed_count = self.joint_fixed[:, i].sum()
            print(f"    {name}: {fixed_count}/{self.num_frames} frames")
        
        print(f"  Palm orientation fixed:")
        for i, name in enumerate(joint_names):
            fixed_count = self.palm_fixed[:, i].sum()
            print(f"    {name}: {fixed_count}/{self.num_frames} frames")
        
        print(f"  Optimized:")
        for i, name in enumerate(joint_names):
            optimized_count = self.optimized[:, i].sum()
            print(f"    {name}: {optimized_count}/{self.num_frames} frames")
        
        total_fixed = (self.joint_fixed | self.palm_fixed).sum()
        print(f"  Total fixed joints: {total_fixed}")


def compute_palm_contact_and_orientation(
    human_joints: torch.Tensor,       # (T, J, 3)
    object_verts: torch.Tensor,       # (T, N, 3)
    hand: str = 'right',              # 'left' or 'right'
    contact_thresh: float = 0.09,     # Contact distance threshold (in meters)
    orient_angle_thresh: float = 70.0,# Maximum orientation angle threshold (degrees), 90° represents hemisphere
    orient_dist_thresh: float = 0.09  # Orientation distance threshold (in meters), used for filtering contact points

):
    """
    Returns:
      contact_mask:  (T,) bool Tensor, whether there is contact (any vertex distance < contact_thresh)
      orient_mask:   (T,) bool Tensor, only when in contact and there exists a vertex
                     within normal ±orient_angle_thresh range and distance < contact_thresh
      normals:       (T, 3)   Tensor, palm normals for each frame (normalized)
    """
    # 0) Preprocessing: ensure same device
    if human_joints.device != object_verts.device:
        human_joints = human_joints.to(object_verts.device)
    T, J, _ = human_joints.shape

    # 1) Select joint indices & normal flipping
    hand = hand.lower()
    if hand.startswith('r'):
        # Right hand indices
        IDX_WRIST     = 21
        IDX_INDEX     = 40
        IDX_PINKY     = 46
        flip_normal   = False
    else:
        # Left hand indices
        IDX_WRIST     = 20
        IDX_INDEX     = 25
        IDX_PINKY     = 31
        flip_normal   = True

    # 2) Extract joint positions
    wrist = human_joints[:, IDX_WRIST    , :]  # (T,3)
    idx   = human_joints[:, IDX_INDEX    , :]  # (T,3)
    pinky = human_joints[:, IDX_PINKY    , :]  # (T,3)

    # 3) Calculate normals & normalize
    v1 = idx   - wrist   # (T,3)
    v2 = pinky - wrist   # (T,3)
    normals = torch.cross(v1, v2, dim=1)  # (T,3)
    if flip_normal:
        normals = -normals
    normals = normals / (normals.norm(dim=1, keepdim=True) + 1e-8)

    # 4) Calculate palm centroid
    centroid = (wrist + idx + pinky) / 3.0  # (T,3)

    # 5) Calculate all vertex relative vectors & distances
    #    object_verts: (T, N, 3)
    rel = object_verts - centroid.unsqueeze(1)   # (T, N, 3)
    dists = rel.norm(dim=2)                     # (T, N)

    # 6) contact_mask: any vertex distance < contact_thresh
    contact_mask = (dists < contact_thresh).any(dim=1)  # (T,)

    # 7) orient_mask: exists vertex that satisfies both distance < contact_thresh
    #    and angle ≤ orient_angle_thresh
    #    cos_thresh = cos(orient_angle_thresh)
    cos_thresh = torch.cos(torch.deg2rad(torch.tensor(orient_angle_thresh, device=normals.device)))

    # 7.1) First normalize rel vectors
    rel_dir = rel / (dists.unsqueeze(-1) + 1e-8)       # (T, N, 3)
    # 7.2) Calculate cosine: dot product of normals.unsqueeze(1) and rel_dir
    cosines = (normals.unsqueeze(1) * rel_dir).sum(dim=2)  # (T, N)
    # 7.3) Filter: cosines >= cos_thresh and dists < contact_thresh
    mask = (cosines >= cos_thresh) & (dists < orient_dist_thresh)  # (T, N)
    orient_mask = mask.any(dim=1)  # (T,)

    return contact_mask, orient_mask, normals


def detect_hand_twist_from_canonical_batch(poses, joints_canonical):
    """Detect wrist twist angles for all frames using canonical bone axis"""
    T = poses.shape[0]
    twist_left_list = []
    twist_right_list = []
    elbow_left_list = []
    elbow_right_list = []
    for frame_idx in range(T):
        pose_i = poses[frame_idx].reshape(52, 3)
        twist_left, twist_right = detect_hand_twist_from_canonical(pose_i, joints_canonical)
        elbow_left, elbow_right = detect_elbow_twist_from_canonical(pose_i, joints_canonical)
        twist_left_list.append(twist_left)
        twist_right_list.append(twist_right)
        elbow_left_list.append(elbow_left)
        elbow_right_list.append(elbow_right)
    
    return twist_left_list, twist_right_list, elbow_left_list, elbow_right_list

def detect_hand_twist_from_canonical(pose_i, joints_canonical):
    """Detect wrist twist angles using canonical bone axis"""
    def compute_twist_angle(pose_wrist, bone_axis):
        rotvec = pose_wrist
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return 0.0
        axis = rotvec / angle
        twist_cos = np.dot(axis, bone_axis)
        twist_angle = angle * twist_cos
        return np.rad2deg(twist_angle)

    bone_axis_left = joints_canonical[LEFT_WRIST] - joints_canonical[LEFT_ELBOW]
    bone_axis_left /= np.linalg.norm(bone_axis_left)

    bone_axis_right = joints_canonical[RIGHT_WRIST] - joints_canonical[RIGHT_ELBOW]
    bone_axis_right /= np.linalg.norm(bone_axis_right)

    twist_left = compute_twist_angle(pose_i[LEFT_WRIST], bone_axis_left)
    twist_right = compute_twist_angle(pose_i[RIGHT_WRIST], bone_axis_right)

    return twist_left, twist_right

def detect_elbow_twist_from_canonical(pose_i, joints_canonical):
    """Detect wrist twist angles using canonical bone axis"""
    def compute_twist_angle(pose_wrist, bone_axis):
        rotvec = pose_wrist
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return 0.0
        axis = rotvec / angle
        twist_cos = np.dot(axis, bone_axis)
        twist_angle = angle * twist_cos
        return np.rad2deg(twist_angle)

    bone_axis_left = joints_canonical[LEFT_ELBOW] - joints_canonical[LEFT_SHOULDER]
    bone_axis_left /= np.linalg.norm(bone_axis_left)

    bone_axis_right = joints_canonical[RIGHT_ELBOW] - joints_canonical[RIGHT_SHOULDER]
    bone_axis_right /= np.linalg.norm(bone_axis_right)

    twist_left = compute_twist_angle(pose_i[LEFT_ELBOW], bone_axis_left)
    twist_right = compute_twist_angle(pose_i[RIGHT_ELBOW], bone_axis_right)

    return twist_left, twist_right

def rotate_pose_around_axis(pose_vec, axis, angle_deg):
    """Rotate pose around given axis"""
    R_current = R.from_rotvec(pose_vec).as_matrix()
    R_correction = R.from_rotvec(np.deg2rad(angle_deg) * axis).as_matrix()
    R_fixed = R_current @ R_correction
    return R.from_matrix(R_fixed).as_rotvec()

def compute_palm_object_angle(
    human_joints: torch.Tensor,       # (T, J, 3)
    object_verts: torch.Tensor,       # (T, N, 3)
    obj_normals: torch.Tensor,        # (T, N, 3)
    hand: str = 'left',               # 'left' or 'right'
    K: int = 500                      # number of closest verts to use
):
    """
    Compute anti-parallel angle between palm normal and average of K closest object normals.
    """
    # pick a common device (use human_joints as reference)
    device = human_joints.device
    human_joints = human_joints.to(device)
    object_verts = object_verts.to(device)
    obj_normals  = obj_normals.to(device)

    T, J, _ = human_joints.shape
    N = object_verts.shape[1]
    K = min(K, N)

    # --- select palm joints ---
    if hand.lower().startswith('r'):
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 21, 42, 48
        flip_normal = False
    else:
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 20, 27, 36
        flip_normal = True

    wrist = human_joints[:, IDX_WRIST, :]
    index = human_joints[:, IDX_INDEX, :]
    pinky = human_joints[:, IDX_PINKY, :]

    # --- palm normal ---
    v1 = index - wrist
    v2 = pinky - wrist
    palm_normals = torch.cross(v1, v2, dim=1)
    if flip_normal:
        palm_normals = -palm_normals
    palm_normals = F.normalize(palm_normals, dim=1, eps=1e-8)  # (T,3)

    # --- centroid ---
    centroid = (wrist + index + pinky) / 3.0  # (T,3)

    # --- find K nearest object vertices per frame ---
    rel = object_verts - centroid.unsqueeze(1)   # (T,N,3)
    dists = rel.norm(dim=2)                      # (T,N)
    _, topi = torch.topk(dists, k=K, dim=1, largest=False)  # (T,K)

    # gather normals
    obj_normals_u = F.normalize(obj_normals, dim=2, eps=1e-8)           # (T,N,3)
    sel_normals = torch.gather(
        obj_normals_u, dim=1, index=topi.unsqueeze(-1).expand(T, K, 3)
    )                                                                   # (T,K,3)

    # --- average normals ---
    avg_norm = sel_normals.mean(dim=1)                                  # (T,3)
    avg_norm = F.normalize(avg_norm, dim=1, eps=1e-8)

    # --- angle (anti-parallel) ---
    cosine = -(palm_normals * avg_norm).sum(dim=1)                      # (T,)
    cosine = torch.clamp(cosine, -1.0, 1.0)
    angles = torch.rad2deg(torch.acos(cosine))

    return angles.detach().cpu().numpy()

def fix_left_palm(
    twist_list,
    contact_mask,
    orient_mask,
    poses,
    joint_idx,
    axis,
    human_joints=None,
    object_verts=None,
    object_normals=None,
    contact_thresh=0.09,
    twist_bounds=( -110.0, 80.0 ),   # (min, max) acceptable wrist twist in degrees
):
    """
    Simplified wrist correction for LEFT hand.

    Logic per frame t:
      - If contact_mask[t] and not orient_mask[t]:
            angle = compute_palm_object_angle(...)[t]
            rotate_pose_around_axis(poses[t, joint_idx*3:joint_idx*3+3], axis, angle)
      - Else if not contact_mask[t] and twist is out-of-bounds:
            angle = -twist_list[t]   # neutralize twist toward 0 (change sign if desired)
            rotate_pose_around_axis(..., angle)

    Returns:
      poses (modified in-place), fixed_frames (list of frame indices that were adjusted)
    """

    T = len(twist_list)
    # build out-of-bounds mask from twist bounds
    lo, hi = twist_bounds
    out_of_bounds = [(tw > hi or tw < lo) for tw in twist_list]
    # convert masks to CPU numpy/bool for indexing consistency
    contact_mask_cpu = contact_mask.bool().cpu().numpy()
    orient_mask_cpu  = orient_mask.bool().cpu().numpy()
    contact_frames = contact_mask_cpu.sum().item()
    if contact_frames > 0:
        contact_but_wrong_orient = (contact_mask_cpu & (~orient_mask_cpu)).sum().item()
        proportion_wrong_orient_given_contact = contact_but_wrong_orient / contact_frames
    else:
        proportion_wrong_orient_given_contact = 0.0
    
    # Calculate proportion of out-of-bounds frames among non-contact frames only
    non_contact_frames = (~contact_mask_cpu).sum().item()
    if non_contact_frames > 0:
        non_contact_out_of_bounds = ((~contact_mask_cpu) & (np.array(out_of_bounds))).sum().item()
        proportion_out_of_bounds_given_no_contact = non_contact_out_of_bounds / non_contact_frames
    else:
        proportion_out_of_bounds_given_no_contact = 0.0
    
    # expects degrees; one angle per frame
    specific_angles = compute_palm_object_angle(
        human_joints, object_verts, object_normals, hand='left'
    )

    fixed_frames = []
    
    for t in range(T):
        if out_of_bounds[t]:
            angle = 0.0
            if contact_mask_cpu[t]:
                angle = float(specific_angles[t])
                if twist_list[t] > 0:
                    angle = -angle
            else:
                angle = float(-twist_list[t])
            poses[t, joint_idx*3 : joint_idx*3 + 3] = rotate_pose_around_axis(
                poses[t, joint_idx*3 : joint_idx*3 + 3], axis, angle
            )
            fixed_frames.append(t)

    return poses, fixed_frames

def fix_right_palm(
    twist_list,
    contact_mask,
    orient_mask,
    poses,
    joint_idx,
    axis,
    human_joints=None,
    object_verts=None,
    object_normals=None,
    contact_thresh=0.09,
    twist_bounds=(-80.0, 110.0),   # acceptable right-wrist twist range (deg)
):
    """
    Simplified wrist correction for RIGHT hand.

    Per frame t:
      - If contact_mask[t] and not orient_mask[t]:
            angle = compute_palm_object_angle(..., hand='right')[t]
            rotate_pose_around_axis(poses[t, joint_idx*3:joint_idx*3+3], axis, angle)
      - Else if not contact_mask[t] and twist is out-of-bounds:
            rotate by the twist angle itself (as requested)

    Returns:
      poses (modified in-place), fixed_frames (list of corrected frame indices)
    """
    T = len(twist_list)

    # out-of-bounds mask from twist bounds
    lo, hi = twist_bounds
    out_of_bounds = [(tw < lo or tw > hi) for tw in twist_list]

    # masks to CPU/numpy for indexing
    contact_mask_cpu = contact_mask.bool().cpu().numpy()
    orient_mask_cpu  = orient_mask.bool().cpu().numpy()
    twist_array = np.array(twist_list, dtype=np.float32)
    
    # Calculate proportion of frames where hand is in contact but not in correct orientation
    # Move tensors to CPU for numpy operations
    contact_mask_cpu = contact_mask.bool().cpu()
    orient_mask_cpu = orient_mask.bool().cpu()
    
    contact_frames = contact_mask_cpu.sum().item()
    if contact_frames > 0:
        contact_but_wrong_orient = (contact_mask_cpu & (~orient_mask_cpu)).sum().item()
        proportion_wrong_orient_given_contact = contact_but_wrong_orient / contact_frames
    else:
        proportion_wrong_orient_given_contact = 0.0
    
    specific_angles = compute_palm_object_angle(
        human_joints, object_verts,object_normals, hand='right'
    )
    # Calculate proportion of out-of-bounds frames among non-contact frames only
    non_contact_frames = (~contact_mask_cpu).sum().item()
    if non_contact_frames > 0:
        non_contact_out_of_bounds = ((~contact_mask_cpu) & (np.array(out_of_bounds))).sum().item()
        proportion_out_of_bounds_given_no_contact = non_contact_out_of_bounds / non_contact_frames
    else:
        proportion_out_of_bounds_given_no_contact = 0.0
    
    fixed_frames = []
    
    for t in range(T):
        # Case 1: in contact but wrong orientation -> use palm-object angle
        if out_of_bounds[t]:
            angle = 0.0
            if contact_mask_cpu[t]:
                angle = float(specific_angles[t])
                if twist_list[t] > 0:
                    angle = -angle
            else:
                angle = float(-twist_list[t])
            poses[t, joint_idx*3 : joint_idx*3 + 3] = rotate_pose_around_axis(
                poses[t, joint_idx*3 : joint_idx*3 + 3], axis, angle
            )
            fixed_frames.append(t)
            
    return poses, fixed_frames



def precompute_hand_object_distances(verts, verts_obj_transformed, obj_normals, rhand_idx, lhand_idx):
    """
    Pre-compute hand-object distances for the original sequence to create distance masks.
    This is computed once before optimization to identify frames that are close to the object.
    
    Args:
        verts: Human mesh vertices (T, V, 3) for the original sequence
        verts_obj_transformed: Object vertices (T, N, 3)
        rhand_idx: Right hand vertex indices
        lhand_idx: Left hand vertex indices
    
    Returns:
        dict: Contains distance masks and contact information for both hands
    """
    if rhand_idx is None or lhand_idx is None:
        print("Warning: Hand indices not available, skipping distance pre-computation")
        return None
    
    # Ensure all tensors are on the same device
    device = verts.device
    verts_obj_transformed = verts_obj_transformed.to(device)
    
    T = verts.shape[0]# Compute distances for right hand using point2point_signed (same as optimize.py)
    right_hand_verts = verts[:, rhand_idx, :]  # (T, R, 3) where R = 778
    # Use point2point_signed to get signed distances efficiently
    _, right_signed_distances, _, _, _, _ = point2point_signed(right_hand_verts, verts_obj_transformed, y_normals = obj_normals, return_vector=True)
    right_hand_min_dist = torch.min(right_signed_distances, dim=1)[0]  # (T,) - minimum distance for each frame
    # right_o2h_min_dist = torch.min(right_o2h_signed, dim=1)[0]  # (T,) - minimum distance for each frame
    # Compute distances for left hand using point2point_signed (same as optimize.py)
    left_hand_verts = verts[:, lhand_idx, :]  # (T, L, 3) where L = 778
    # Use point2point_signed to get signed distances efficiently
    _, left_signed_distances, _, _, _, _ = point2point_signed(left_hand_verts, verts_obj_transformed, y_normals = obj_normals, return_vector=True)
    left_hand_min_dist = torch.min(left_signed_distances, dim=1)[0]  # (T,) - minimum distance for each frame
 # Create distance masks (frames that are close to object)
    correction_thresh = -0.1
    penetration_thresh = 0
    contact_thresh = 0.02  # 2cm threshold for contact
    close_thresh = 0.06    # 20cm threshold for "close" frames
    
    right_pen_mask = (right_hand_min_dist <= penetration_thresh) & (right_hand_min_dist >= correction_thresh)
    left_pen_mask = (left_hand_min_dist <= penetration_thresh) & (left_hand_min_dist >= correction_thresh)
    
    right_contact_mask = (right_hand_min_dist <= contact_thresh) & (right_hand_min_dist >= correction_thresh)
    left_contact_mask = (left_hand_min_dist <= contact_thresh) & (left_hand_min_dist >= correction_thresh)
    
    right_close_mask = (right_hand_min_dist <= close_thresh) & (right_hand_min_dist >= correction_thresh)
    left_close_mask = (left_hand_min_dist <= close_thresh) & (left_hand_min_dist >= correction_thresh)
    
    # Create optimization masks (frames that should be optimized for penetration)
    right_optimize_mask = (right_hand_min_dist <= close_thresh) & (right_hand_min_dist >= correction_thresh)
    left_optimize_mask = (left_hand_min_dist <= close_thresh) & (left_hand_min_dist >= correction_thresh)
    
    
    return {
        'right_pen_mask': right_pen_mask,
        'left_pen_mask': left_pen_mask,
        'right_contact_mask': right_contact_mask,
        'left_contact_mask': left_contact_mask,
        'right_close_mask': right_close_mask,
        'left_close_mask': left_close_mask,
        'right_optimize_mask': right_optimize_mask,
        'left_optimize_mask': left_optimize_mask,
        'right_hand_min_dist': right_hand_min_dist,
        'left_hand_min_dist': left_hand_min_dist,
    }


def quick_pose_comparison(original_poses, optimized_poses, joint_names=None):
    """
    Quick comparison of poses before and after optimization.
    
    Args:
        original_poses: (T, 156) - Original poses
        optimized_poses: (T, 156) - Optimized poses
        joint_names: List of joint names to check (default: all 8 joints)
    
    Returns:
        dict: Summary of changes
    """
    if joint_names is None:
        joint_names = ['left_collar', 'right_collar', 'left_shoulder', 'right_shoulder', 
                      'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
    
    # Joint pose indices
    joint_indices = {
        'left_collar': (39, 42),
        'right_collar': (42, 45),
        'left_shoulder': (48, 51),
        'right_shoulder': (51, 54),
        'left_elbow': (54, 57),
        'right_elbow': (57, 60),
        'left_wrist': (60, 63),
        'right_wrist': (63, 66)
    }
    
    changes = {}
    total_max_change = 0.0
    
    for joint_name in joint_names:
        if joint_name in joint_indices:
            start_idx, end_idx = joint_indices[joint_name]
            
            # Extract poses for this joint
            orig_joint = original_poses[:, start_idx:end_idx]
            opt_joint = optimized_poses[:, start_idx:end_idx]
            
            # Calculate differences
            diff = np.abs(opt_joint - orig_joint)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            changes[joint_name] = {
                'max_change': max_diff,
                'mean_change': mean_diff,
                'max_change_deg': np.degrees(max_diff),
                'mean_change_deg': np.degrees(mean_diff)
            }
            
            total_max_change = max(total_max_change, max_diff)
    
    # Print summary
    print(f"\nQuick Pose Comparison:")
    print(f"{'Joint':<15} {'Max Change':<12} {'Mean Change':<12}")
    print("-" * 45)
    
    for joint_name in joint_names:
        if joint_name in changes:
            change = changes[joint_name]
            print(f"{joint_name:<15} {change['max_change_deg']:<12.2f}° {change['mean_change_deg']:<12.2f}°")
    
    print(f"\nOverall max change: {np.degrees(total_max_change):.2f}°")
    
    return changes

def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False):
    """Load SMPL data"""
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    
    frame_times = poses.shape[0]
    
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            smplx_output = smpl_model(
                body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                global_orient=torch.from_numpy(poses[:, :3]).float(),
                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                transl=torch.from_numpy(trans).float()
            )
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
            else:
                smpl_model = smplx10[gender]
            smplx_output = smpl_model(
                body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                global_orient=torch.from_numpy(poses[:, :3]).float(),
                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                transl=torch.from_numpy(trans).float()
            )
        verts = to_cpu(smplx_output.vertices)
        joints = to_cpu(smplx_output.joints)
        faces = smpl_model.faces.astype(np.int32)
        faces = torch.from_numpy(faces).unsqueeze(0).repeat(frame_times, 1, 1)
    elif num_betas == 16:
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            # Use SMPLX16 model (same as two_stage_wrist_optimize.py)
            smpl_model = smplx16[gender]
        smplx_output = smpl_model(
            pose_body=torch.from_numpy(poses[:, 3:66]).float().to(device),
            pose_hand=torch.from_numpy(poses[:, 66:156]).float().to(device),
            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device),
            root_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            trans=torch.from_numpy(trans).float().to(device)
        )
        verts = to_cpu(smplx_output.v)
        joints = to_cpu(smplx_output.Jtr)
        faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)

    return verts, joints, faces, poses, betas, trans, gender

def get_mean_pose_joints(name, gender, model_type, num_betas, use_pca=False):
    """Get canonical joint positions - Only supports SMPLX16 for OMOMO"""
    frame_times = 1
    pose_zeros = torch.zeros(frame_times, 156).float().to(device)
    trans_zeros = torch.zeros(frame_times, 3).float().to(device)        
    betas_zeros = torch.zeros(frame_times, num_betas).float().to(device)

    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
            else:
                smpl_model = smplx10[gender]
        output = smpl_model(
            body_pose=pose_zeros[:, 3:66],
            global_orient=pose_zeros[:, :3],
            left_hand_pose=pose_zeros[:, 66:111],
            right_hand_pose=pose_zeros[:, 111:156],
            transl=trans_zeros,
            betas=betas_zeros
        )
        joints = output.joints[0].detach().cpu().numpy()
    else:
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        output = smpl_model(
            pose_body=pose_zeros[:, 3:66],
            pose_hand=pose_zeros[:, 66:156],
            root_orient=pose_zeros[:, :3],
            trans=trans_zeros,
            betas=betas_zeros
        )
        joints = output.Jtr[0].detach().cpu().numpy()

    return joints

def main(dataset_path, sequence_name, threshold):
    """Main pipeline function - Supports all datasets for Steps 1&2, SMPLX16 for optimization"""
    # Derived paths
    human_path = os.path.join(dataset_path, 'sequences_canonical')
    object_path = os.path.join(dataset_path, 'objects')
    dataset_path_name = dataset_path.split('/')[-1]
    data_name = []
    if sequence_name is None:
        # reads from scan_results.csv
        with open('scan_results.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                data_name.append(row[0])
    else:
        data_name = data_name + [sequence_name]
    
    for sequence_name in tqdm(data_name):
        if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
            verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplh', 10)
            canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 10)
        elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
            verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplh', 16)
            canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplh', 16)
        elif dataset_path_name.upper() == 'CHAIRS':
            verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 10)
            canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10)
        elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
            verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 10, True)
            canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 10, True)
        elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
            verts, joints, faces, poses, betas, trans, gender = visualize_smpl(sequence_name, human_path, 'smplx', 16)
            canonical_joints = get_mean_pose_joints(sequence_name, gender, 'smplx', 16)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_path_name}")

        # Load object data
        with np.load(os.path.join(human_path, sequence_name, 'object.npz'), allow_pickle=True) as f:
            obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
        angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()

        OBJ_MESH = trimesh.load(os.path.join(object_path, obj_name, obj_name+'.obj'))

        ov = np.array(OBJ_MESH.vertices).astype(np.float32)
        object_faces = OBJ_MESH.faces.astype(np.int32)
        device = torch.device('cuda:0')
        ov = torch.from_numpy(ov).float().to(device)
        rot = torch.tensor(angle_matrix).float().to(device)
        obj_trans = torch.tensor(obj_trans).float().to(device)
        object_verts = torch.einsum('ni,tij->tnj', ov, rot.permute(0,2,1)) + obj_trans.unsqueeze(1)
        render_path = f'./save_fix/{dataset_path_name}'
        os.makedirs(render_path, exist_ok=True)
        T = poses.shape[0]

        if isinstance(verts, torch.Tensor):
            verts = verts.to(device)
        else:
            verts = torch.from_numpy(verts).float().to(device)

        # if args.visualize:
        #     visualize_body_obj(
        #         verts.float().detach().cpu().numpy(),
        #         faces[0].detach().cpu().numpy().astype(np.int32),
        #         object_verts.detach().cpu().numpy(),
        #         object_faces,
        #         save_path=os.path.join(render_path, f'{sequence_name}_original.mp4'),
        #         show_frame=True,
        #         multi_angle=True,
        #     )
        joints = joints.to(device)
        
        obj_normals=vertex_normals(object_verts,torch.tensor(object_faces.astype(np.float32)).unsqueeze(0).repeat(object_verts.shape[0],1,1).to(device))
        distance_info = precompute_hand_object_distances(verts, object_verts, obj_normals, rhand_idx, lhand_idx)
        
        # Print contact information for both hands
        if distance_info is not None:
            left_pen_frames = torch.where(distance_info['left_pen_mask'])[0].tolist()
            right_pen_frames = torch.where(distance_info['right_pen_mask'])[0].tolist()
            
            # Convert to unique frame indices (since each frame has 778 hand vertices)
            left_pen_unique_frames = sorted(list(set(left_pen_frames)))
            right_pen_unique_frames = sorted(list(set(right_pen_frames)))
            
        
            left_contact_frames = torch.where(distance_info['left_contact_mask'])[0].tolist()
            right_contact_frames = torch.where(distance_info['right_contact_mask'])[0].tolist()
            left_contact_unique_frames = sorted(list(set(left_contact_frames)))
            right_contact_unique_frames = sorted(list(set(right_contact_frames)))
        else:
            print("Distance info not available")


        # Initialize fix tracker
        fix_tracker = FixTracker(T)

        joints_to_fix = []
        # find the joints with any diff_angle > 0.4
            
        # Track which frames were fixed for each joint using exact information from fix_joint_poses
        joint_fixed_frames = {joint_idx: [] for joint_idx in [LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]}
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses).float()  # now a torch.Tensor
        # view poses from (N, 156) to (N, 52, 3)
        poses = poses.reshape(-1, 52, 3)
        diff_angle, diff_axis = pose_delta_axis_angle(poses)
        
        for j in [13, 16, 18, 20, 14, 17, 19, 21]:
            for i in range(diff_angle.shape[0]):
                if abs(diff_angle[i, j]) > threshold:
                    joints_to_fix.append(j)
                    break

        for joint in joints_to_fix:
            poses, fixed_boundaries = fix_joint_poses_simple(poses, joint, angle_thresh=threshold)

            joint_fixed_frames[joint].extend(range(fixed_boundaries[0],poses.shape[0]))
        
        joint_to_tracker_idx = {
            LEFT_COLLAR: 0, RIGHT_COLLAR: 1,
            LEFT_SHOULDER: 2, RIGHT_SHOULDER: 3,
            LEFT_ELBOW: 4, RIGHT_ELBOW: 5,
            LEFT_WRIST: 6, RIGHT_WRIST: 7
        }
        
        # Mark the frames that were actually fixed during the joint pose fixing process
        for joint_idx, fixed_frames in joint_fixed_frames.items():
            if fixed_frames:  # Only process joints that had frames fixed
                tracker_joint_idx = joint_to_tracker_idx[joint_idx]
                # print(f"Joint {joint_idx}: {len(fixed_frames)} frames fixed")
                fix_tracker.mark_joint_fixed(fixed_frames, [tracker_joint_idx])
        
        total_joint_fixes = fix_tracker.get_joint_fixed_frames_and_joints().sum()
        # print(f"Joint pose fixing applied to {total_joint_fixes} joint-frame combinations")
        
        poses = poses.reshape(-1, 156)
        poses = poses.cpu().numpy()
        if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
            verts, joints, faces = regen_smpl(args.sequence_name, poses, betas, trans, gender, 'smplh', 10)
        elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
            verts, joints, faces = regen_smpl(args.sequence_name, poses, betas, trans, gender, 'smplh', 16)
        elif dataset_path_name.upper() == 'CHAIRS':
            verts, joints, faces = regen_smpl(args.sequence_name, poses, betas, trans, gender, 'smplx', 10)
        elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
            verts, joints, faces = regen_smpl(args.sequence_name, poses, betas, trans, gender, 'smplx', 10, True)
        elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
            verts, joints, faces = regen_smpl(args.sequence_name, poses, betas, trans, gender, 'smplx', 16)

        contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
            joints, object_verts, hand='left'
        )
        
        contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
            joints, object_verts, hand='right'
        )
        twist_left_list, twist_right_list, elbow_left_list, elbow_right_list = detect_hand_twist_from_canonical_batch(poses, canonical_joints)

        # Fix left hand
        axis_left = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]
        axis_left /= np.linalg.norm(axis_left)
        poses, left_fixed_frames = fix_left_palm(
            twist_left_list, distance_info['left_close_mask'], orient_mask_l, poses, LEFT_WRIST, axis_left, joints, object_verts, obj_normals
        )
        if left_fixed_frames:
            fix_tracker.mark_palm_fixed(left_fixed_frames, [6])  # left_wrist = index 6
        # # Fix right hand
        axis_right = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]
        axis_right /= np.linalg.norm(axis_right)
        poses, right_fixed_frames = fix_right_palm(
            twist_right_list, distance_info['right_close_mask'], orient_mask_r, poses, RIGHT_WRIST, axis_right, joints, object_verts, obj_normals
        )
        if right_fixed_frames:
            fix_tracker.mark_palm_fixed(right_fixed_frames, [7])  # right_wrist = index 7
        twist_left_list, twist_right_list, elbow_left_list, elbow_right_list = detect_hand_twist_from_canonical_batch(poses, canonical_joints)
 

        # Step 2: Palm orientation fixing
        
        # Regenerate joints from updated poses after joint fixing
        # print("Regenerating joints from updated poses...")
        if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
            _, joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 10)
        elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
            _, joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 16)
        elif dataset_path_name.upper() == 'CHAIRS':
            _, joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10)
        elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
            _, joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10, True)
        elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
            _, joints, _ = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 16)
        
        # Compute twist angles for palm fixing
        twist_left_list, twist_right_list, elbow_left_list, elbow_right_list = detect_hand_twist_from_canonical_batch(poses, canonical_joints)
        contact_mask_l, orient_mask_l, _ = compute_palm_contact_and_orientation(
            joints, object_verts, hand='left'
        )
        contact_mask_r, orient_mask_r, _ = compute_palm_contact_and_orientation(
            joints, object_verts, hand='right'
        )

        joint_fixed_frames = {joint_idx: [] for joint_idx in [LEFT_COLLAR, RIGHT_COLLAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]}
        
        # Get information about which joints were fixed from the fix tracker
        all_fixed_joints = fix_tracker.get_fixed_frames_and_joints()
        
        # Create mask for the smoothing function
        # joint_optimization_mask: (8,) - which joints were optimized
        joint_optimization_mask = np.any(all_fixed_joints, axis=0)  # (8,) - True if joint was fixed in any frame
        poses = smooth_flips(poses, joint_optimization_mask, window_size=10)
        
        # Step 3: Optimization with selective loss (uses SMPLX16 model)
        all_fixed_joints = fix_tracker.get_fixed_frames_and_joints()
        total_fixed_combinations = all_fixed_joints.sum()
        original_poses = poses.copy()
        if total_fixed_combinations > 0:
            poses = optimize_poses(
                poses, betas, trans, gender, object_verts, obj_normals, fix_tracker, 
                distance_info, rhand_idx = rhand_idx, lhand_idx = lhand_idx,
                num_epochs=600, lr = 0.001, canonical_joints=canonical_joints, smpl_model=smplx16[gender]
            )
            # Quick comparison of poses before and after optimization
            # if args.visualize:
            #     print("\n" + "="*60)
            #     print("QUICK POSE COMPARISON (Before vs After Optimization)")
            #     print("="*60)
            #     quick_pose_comparison(original_poses, poses)
            
            # Mark only the specific joint-frame combinations that were actually fixed as optimized
            fix_tracker.mark_optimized_from_mask(all_fixed_joints)
            # fixed_sequence_name.append(sequence_name)
        else:
            print("No joint-frame combinations to optimize")
        
        # Save temporary result after Step 3

        # Print summary
        # fix_tracker.print_summary()

        # Save final results
        fixed_human_path = os.path.join(human_path, sequence_name, 'human.npz')
        np.savez(fixed_human_path, 
                poses=poses, 
                betas=betas, 
                trans=trans, 
                gender=gender)
        print(f"Final pipeline results saved to: {fixed_human_path}")

        # Generate visualization
        
        # Regenerate SMPL with fixed poses
        # if args.visualize:
        #     if dataset_path_name.upper() == 'BEHAVE' or dataset_path_name.upper() == 'BEHAVE_CORRECT':
        #         verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 10)
        #     elif dataset_path_name.upper() == 'NEURALDOME' or dataset_path_name.upper() == 'IMHD':
        #         verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplh', 16)
        #     elif dataset_path_name.upper() == 'CHAIRS':
        #         verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10)
        #     elif dataset_path_name.upper() == 'INTERCAP' or dataset_path_name.upper() == 'INTERCAP_CORRECT':
        #         verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 10, True)
        #     elif dataset_path_name.upper() == 'OMOMO' or dataset_path_name.upper() == 'OMOMO_CORRECT':
        #         verts, joints, faces = regen_smpl(sequence_name, poses, betas, trans, gender, 'smplx', 16)

        #     visualize_body_obj(
        #         verts.float().detach().cpu().numpy(),
        #         faces[0].detach().cpu().numpy().astype(np.int32),
        #         object_verts.detach().cpu().numpy(),
        #         object_faces,
        #         save_path=os.path.join(render_path, f'{sequence_name}_optimized.mp4'),
        #         show_frame=True,
        #         multi_angle=True,
        #     )

def regen_smpl(name, poses, betas, trans, gender, model_type, num_betas, use_pca=False):
    """Regenerate SMPL with fixed poses"""
    frame_times = poses.shape[0]
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
            else:
                smpl_model = smplx10[gender]
        
        smplx_output = smpl_model(
            body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device),
            global_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            left_hand_pose=torch.from_numpy(poses[:, 66:111]).float().to(device),
            right_hand_pose=torch.from_numpy(poses[:, 111:156]).float().to(device),
            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device),
            transl=torch.from_numpy(trans).float().to(device)
        )
        verts = to_cpu(smplx_output.vertices)
        joints = to_cpu(smplx_output.joints)
        faces = smpl_model.faces.astype(np.int32)
        faces = torch.from_numpy(faces).unsqueeze(0).repeat(frame_times, 1, 1)
    elif num_betas == 16:
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        
        smplx_output = smpl_model(
            pose_body=torch.from_numpy(poses[:, 3:66]).float().to(device),
            pose_hand=torch.from_numpy(poses[:, 66:156]).float().to(device),
            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device),
            root_orient=torch.from_numpy(poses[:, :3]).float().to(device),
            trans=torch.from_numpy(trans).float().to(device)
        )
        verts = to_cpu(smplx_output.v)
        joints = to_cpu(smplx_output.Jtr)
        faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)

    return verts, joints, faces



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive pipeline: Joint fix + Palm fix + Optimization")
    parser.add_argument("--dataset", required=True, help="Path to the dataset root.")
    parser.add_argument("--sequence_name", required=False, default=None, help="Name of the sequence.")
    parser.add_argument("--threshold", type=float, default=0.2, help="Angle threshold in radians for flip detection.")
    # parser.add_argument("--visualize", type = bool, default=False, help="Whether to visualize the results.")
    args = parser.parse_args()
    dataset_path = os.path.join('./data', args.dataset)
    main(dataset_path, args.sequence_name, args.threshold)
