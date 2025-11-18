#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import csv
import smplx
from scipy.spatial.transform import Rotation as R
from human_body_prior.body_model.body_model import BodyModel
from tqdm import tqdm
MODEL_PATH = 'models'

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


def get_mean_pose_joints(name, MOTION_PATH, model_type, num_betas, use_pca=False):
    """
    Return: joints_np: (55, 3) numpy array in canonical (zero-pose) space
    """

    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        _, _, _, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])

    frame_times = 1  # only one frame needed
    pose_zeros = torch.zeros(frame_times, 156).float()  # works for SMPLX or SMPLH

    trans_zeros = torch.zeros(frame_times, 3).float()
    betas_zeros = torch.zeros(frame_times, num_betas).float()

    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            output = smpl_model(body_pose=pose_zeros[:, 3:66],
                                global_orient=pose_zeros[:, :3],
                                left_hand_pose=pose_zeros[:, 66:111],
                                right_hand_pose=pose_zeros[:, 111:156],
                                transl=trans_zeros,
                                betas=betas_zeros)
        elif model_type == 'smplx':
            if use_pca:
                smpl_model = smplx12[gender]
                output = smpl_model(body_pose=pose_zeros[:, 3:66],
                                    global_orient=pose_zeros[:, :3],
                                    left_hand_pose=pose_zeros[:, 66:78],
                                    right_hand_pose=pose_zeros[:, 78:90],
                                    jaw_pose=torch.zeros(frame_times, 3),
                                    leye_pose=torch.zeros(frame_times, 3),
                                    reye_pose=torch.zeros(frame_times, 3),
                                    expression=torch.zeros(frame_times, 10),
                                    transl=trans_zeros,
                                    betas=betas_zeros)
            else:
                smpl_model = smplx10[gender]
                output = smpl_model(body_pose=pose_zeros[:, 3:66],
                                    global_orient=pose_zeros[:, :3],
                                    left_hand_pose=pose_zeros[:, 66:111],
                                    right_hand_pose=pose_zeros[:, 111:156],
                                    jaw_pose=torch.zeros(frame_times, 3),
                                    leye_pose=torch.zeros(frame_times, 3),
                                    reye_pose=torch.zeros(frame_times, 3),
                                    expression=torch.zeros(frame_times, 10),
                                    transl=trans_zeros,
                                    betas=betas_zeros)
        joints = output.joints[0].detach().cpu().numpy()  # (55, 3)
    else:
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        output = smpl_model(pose_body=pose_zeros[:, 3:66],
                            pose_hand=pose_zeros[:, 66:156],
                            root_orient=pose_zeros[:, :3],
                            trans=trans_zeros,
                            betas=betas_zeros)
        joints = output.Jtr[0].detach().cpu().numpy()

    return joints

# ---------------- Quaternion helpers (kept identical in spirit to yours) ----------------
def axis_angle_to_quat(aa):
    # aa: (..., 3) axis-angle, angle = ||aa||
    aa_t = torch.as_tensor(aa, dtype=torch.float32)
    angle = torch.linalg.norm(aa_t, dim=-1, keepdims=True).clamp_min(1e-8)
    axis  = aa_t / angle
    half  = 0.5 * angle
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)
    return torch.cat([axis * sin_h, cos_h], dim=-1)  # (x,y,z,w)

def quat_conjugate(q):
    q = torch.as_tensor(q, dtype=torch.float32)
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def quat_mul(a, b):
    a = torch.as_tensor(a, dtype=torch.float32)
    b = torch.as_tensor(b, dtype=torch.float32)
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return torch.stack([x,y,z,w], dim=-1)

def quat_normalize(q, eps=1e-8):
    q = torch.as_tensor(q, dtype=torch.float32)
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_to_angle_axis(q, eps=1e-8):
    q = quat_normalize(q)
    w = q[..., 3].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)                      # [0, pi]
    sin_half = torch.sqrt((1.0 - w*w).clamp_min(0))  # = ||xyz||
    axis = torch.zeros_like(q[..., :3])
    mask = sin_half > 1e-5
    axis[mask] = q[..., :3][mask] / sin_half[mask].unsqueeze(-1)
    # default axis when angle ~ 0
    axis[~mask] = torch.tensor([0.,0.,1.], device=q.device, dtype=q.dtype)
    return angle, axis

def pose_delta_axis_angle(poses):
    """
    poses: (T, N, 3) axis-angle
    returns:
      diff_angle: (T-1, N) radians in [0, pi]
      diff_axis:  (T-1, N, 3) unit axes
    """
    T, N, _ = poses.shape
    q = axis_angle_to_quat(poses)         # (T, N, 4)
    q_prev = q[:-1]                       # (T-1, N, 4)
    q_curr = q[1:]                        # (T-1, N, 4)
    q_rel  = quat_mul(quat_conjugate(q_prev), q_curr)
    q_rel  = quat_normalize(q_rel)
    diff_angle, diff_axis = quat_to_angle_axis(q_rel)
    return diff_angle, diff_axis

# ---------------- Scanner ----------------
def scan_dataset(dataset_path, threshold=0.4, joints=None, pose_file="human.npz",
                 output_csv="scan_results.csv"):
    """
    Scans all sequences under <dataset_path>/sequences_canonical.
    For each sequence, loads <pose_file> (default: human.npz), computes per-frame deltas,
    and records sequences where any selected-joint delta > threshold.

    Args:
      dataset_path: path to data/[dataset_name]
      threshold: radians (default 0.2)
      joints: list of int joint indices to check; if None, use [13,16,18,20,14,17,19,21]
      pose_file: name of the pose npz file (default 'human.npz')
      output_csv: path to write results (CSV)
    """
    if joints is None:
        joints = [13, 16, 18, 20, 14, 17, 19, 21]  # 8 joints for wrist, elbow, shoulder, collar

    seq_root = os.path.join(dataset_path, "sequences_canonical")
    if not os.path.isdir(seq_root):
        raise FileNotFoundError(f"sequences_canonical not found under {dataset_path}")
    dataset_name = dataset_path.split('/')[-1]
    print(f"Dataset name: {dataset_name}")
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    results = []

    seq_names = sorted([d for d in os.listdir(seq_root)
                        if os.path.isdir(os.path.join(seq_root, d))])

    print(f"Found {len(seq_names)} sequences under {seq_root}")
    n_flagged = 0

    # use tqdm to print the progress
    for seq in tqdm(seq_names):
        npz_path = os.path.join(seq_root, seq, pose_file)
        if dataset_name.upper() == 'GRAB':
            canonical_joints = get_mean_pose_joints(seq, seq_root, 'smplh', 10)
        elif dataset_name.upper() == 'BEHAVE':
            canonical_joints = get_mean_pose_joints(seq, seq_root, 'smplh', 10)
        elif dataset_name.upper() == 'NEURALDOME' or dataset_name.upper() == 'IMHD':
            canonical_joints = get_mean_pose_joints(seq, seq_root, 'smplh', 16)
        elif dataset_name.upper() == 'CHAIRS':
            canonical_joints = get_mean_pose_joints(seq, seq_root, 'smplx', 10)
        elif dataset_name.upper() == 'INTERCAP':
            canonical_joints = get_mean_pose_joints(seq, seq_root, 'smplx', 10, True)
        elif dataset_name.upper() == 'OMOMO':
            canonical_joints = get_mean_pose_joints(seq, seq_root, 'smplx', 16)

        if not os.path.isfile(npz_path):
            print(f"[WARN] Missing {pose_file} for sequence '{seq}', skipping.")
            continue

        try:
            with np.load(npz_path, allow_pickle=True) as f:
                poses = f["poses"]  # (T, 156) or (T, 52, 3)
        except Exception as e:
            print(f"[WARN] Failed to load poses for '{seq}': {e}")
            continue

        if poses.ndim == 2 and poses.shape[1] == 156:
            T = poses.shape[0]
            poses_torch = torch.from_numpy(poses).float().reshape(T, 52, 3)
        elif poses.ndim == 3 and poses.shape[1:] == (52, 3):
            poses_torch = torch.from_numpy(poses).float()
            T = poses_torch.shape[0]
        else:
            print(f"[WARN] Unexpected pose shape {poses.shape} for '{seq}', skipping.")
            continue

        if T < 2:
            print(f"[INFO] Sequence '{seq}' has <2 frames, skipping.")
            continue


        with torch.no_grad():
            diff_angle, _ = pose_delta_axis_angle(poses_torch)   # (T-1, N)
            twist_left_list, twist_right_list = detect_hand_twist_from_canonical(poses_torch, canonical_joints)
            # Apply wrist twist rule across all frames
            twist_violations = []
            if len(twist_left_list) > 0:
                prev_left = twist_left_list[0]
                prev_right = twist_right_list[0]
                for t_idx_twist in range(len(twist_left_list)):
                    tl = twist_left_list[t_idx_twist]
                    tr = twist_right_list[t_idx_twist]
                    cond_limits = (tl > 90) or (tl < -110) or (tr > 110) or (tr < -90)
                    cond_jump = False if t_idx_twist == 0 else (
                        abs(tl - prev_left) > 40 or abs(tr - prev_right) > 40
                    )
                    if cond_limits or cond_jump:
                        twist_violations.append(t_idx_twist)
                    prev_left = tl
                    prev_right = tr
            

            # restrict to selected joints
            da_sel = diff_angle[:, joints]                       # (T-1, Jsel)
            max_val, flat_idx = torch.max(da_sel.reshape(-1), dim=0)
            max_val = float(max_val.item())
            if twist_violations:
                results.append({
                    "sequence": seq,
                    "twist_num_violations": int(len(twist_violations)),
                })
                n_flagged += 1
            elif max_val > threshold:
                # decode frame and joint
                t_idx = int(flat_idx // len(joints))
                j_idx_local = int(flat_idx % len(joints))
                j_global = int(joints[j_idx_local])
                # count exceedances
                num_exceed = int((da_sel > threshold).sum().item())
                results.append({
                    "sequence": seq,
                    "max_delta_rad": max_val,
                    "max_delta_deg": float(np.degrees(max_val)),
                    "frame_start": t_idx,
                    "frame_end": t_idx + 1,
                    "joint_index": j_global,
                    "num_exceed": num_exceed,
                })
                n_flagged += 1

    # write CSV
    # only write the sequence column
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sequence"
        ])
        for row in results:
            row_dict = {"sequence": row["sequence"]}
            w.writerow(row_dict)
    print("\n=== Scan complete ===")
    print(f"Threshold: {threshold} rad ({np.degrees(threshold):.2f}°)")
    print(f"Sequences scanned: {len(seq_names)}")
    print(f"Sequences flagged: {n_flagged}")
    print(f"Wrote results to: {output_csv}")

def parse_joints(arg: str):
    """
    Parses a comma/space-separated list of ints.
    Returns list[int].
    """
    s = arg.strip().lower()
    parts = [p for p in s.replace(",", " ").split() if p]
    return [int(p) for p in parts]

def detect_hand_twist_from_canonical(poses, joints_canonical):
    """
    Detect wrist twist angles using canonical bone axis (rest pose).

    Args:
        poses: (T, 52, 3) array of axis-angle poses per joint, or (52, 3) for a single frame
        joints_canonical: (55, 3) array of joint positions in rest pose

    Returns:
        twist_left_deg_list, twist_right_deg_list: lists of twist angles (degrees) per frame
    """
    def compute_twist_angle(pose_wrist, bone_axis):
        """Compute rotation around the given bone axis (twist) in degrees."""
        rotvec = pose_wrist
        angle = np.linalg.norm(rotvec)
        if angle < 1e-6:
            return 0.0
        axis = rotvec / angle
        twist_cos = np.dot(axis, bone_axis)
        twist_angle = angle * twist_cos  # project rotation onto bone axis
        return np.rad2deg(twist_angle)

    # Joint indices
    LEFT_ELBOW, LEFT_WRIST = 18, 20
    RIGHT_ELBOW, RIGHT_WRIST = 19, 21

    # Canonical bone axes from rest pose
    bone_axis_left = joints_canonical[LEFT_WRIST] - joints_canonical[LEFT_ELBOW]
    bone_axis_left /= np.linalg.norm(bone_axis_left)

    bone_axis_right = joints_canonical[RIGHT_WRIST] - joints_canonical[RIGHT_ELBOW]
    bone_axis_right /= np.linalg.norm(bone_axis_right)

    # Normalize input to (T, 52, 3)
    poses_arr = np.asarray(poses)
    if poses_arr.ndim == 2 and poses_arr.shape == (52, 3):
        poses_arr = poses_arr[None, ...]  # (1, 52, 3)
    elif poses_arr.ndim != 3 or poses_arr.shape[1:] != (52, 3):
        raise ValueError(f"Expected poses shape (T,52,3) or (52,3), got {poses_arr.shape}")

    twist_left_list = []
    twist_right_list = []
    for frame_pose in poses_arr:
        twist_left = compute_twist_angle(frame_pose[LEFT_WRIST], bone_axis_left)
        twist_right = compute_twist_angle(frame_pose[RIGHT_WRIST], bone_axis_right)
        twist_left_list.append(float(twist_left))
        twist_right_list.append(float(twist_right))

    return twist_left_list, twist_right_list

def main():
    ap = argparse.ArgumentParser(description="Scan sequences for large pose deltas.")
    ap.add_argument("--dataset", required=True,
                    help="Path to data/[dataset_name]")
    ap.add_argument("--threshold", type=float, default=0.4,
                    help="Delta threshold in radians (default 0.4)")
    ap.add_argument("--joints", type=str, default="13,16,18,20,14,17,19,21",
                    help="Joint indices to check")
    ap.add_argument("--pose_file", type=str, default="human.npz",
                    help="Pose file name inside each sequence folder (default 'human.npz')")
    ap.add_argument("--output_csv", type=str, default="scan_results.csv",
                    help="Where to write the results CSV")
    args = ap.parse_args()

    joints_sel = parse_joints(args.joints)
    dataset_path = os.path.join('./data', args.dataset)
    scan_dataset(
        dataset_path=dataset_path,
        threshold=args.threshold,
        joints=joints_sel,
        pose_file=args.pose_file,
        output_csv=args.output_csv
    )

if __name__ == "__main__":
    main()
