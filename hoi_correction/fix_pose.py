import torch
import numpy as np

LEFT_COLLAR = 13
RIGHT_COLLAR = 14
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21


JOINT_TO_POSE_MAPPING = {
    LEFT_COLLAR: 39,    # pose indices 39:42 (joint 13)
    RIGHT_COLLAR: 42,   # pose indices 42:45 (joint 14)
    LEFT_SHOULDER: 48,  # pose indices 48:51 (joint 16)
    RIGHT_SHOULDER: 51, # pose indices 51:54 (joint 17)
    LEFT_ELBOW: 54,     # pose indices 54:57 (joint 18)
    RIGHT_ELBOW: 57,    # pose indices 57:60 (joint 19)
    LEFT_WRIST: 60,     # pose indices 60:63 (joint 20)
    RIGHT_WRIST: 63     # pose indices 63:66 (joint 21)
}

def axis_angle_to_quat(aa):
    # aa: (..., 3) axis-angle, angle = ||aa||
    angle = torch.linalg.norm(aa, dim=-1, keepdims=True).clamp_min(1e-8)
    axis  = aa / angle
    half  = 0.5 * angle
    sin_h = torch.sin(half)
    cos_h = torch.cos(half)
    # quaternion layout: (x, y, z, w)
    return torch.cat([axis * sin_h, cos_h], dim=-1)

def quat_conjugate(q):
    return torch.cat([-q[..., :3], q[..., 3:]], dim=-1)

def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = aw*bx + ax*bw + ay*bz - az*by
    y = aw*by - ax*bz + ay*bw + az*bx
    z = aw*bz + ax*by - ay*bx + az*bw
    w = aw*bw - ax*bx - ay*by - az*bz
    return torch.stack([x,y,z,w], dim=-1)

def quat_normalize(q, eps=1e-8):
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_to_angle_axis(q, eps=1e-8):
    q = quat_normalize(q)
    w = q[..., 3].clamp(-1.0, 1.0)
    angle = 2.0 * torch.acos(w)                      # [0, pi]
    sin_half = torch.sqrt((1.0 - w*w).clamp_min(0))  # = ||xyz||
    axis = torch.zeros_like(q[..., :3])
    mask = sin_half > 1e-5
    axis[mask] = q[..., :3][mask] / sin_half[mask].unsqueeze(-1)
    axis[~mask] = torch.tensor([0.,0.,1.], device=q.device, dtype=q.dtype)  # default
    return angle, axis

def quat_from_axis_angle(axis, angle, eps=1e-8):
    if not isinstance(axis, torch.Tensor): axis = torch.tensor(axis, dtype=torch.float32)
    if not isinstance(angle, torch.Tensor): angle = torch.tensor(angle, dtype=torch.float32)
    # normalize axis defensively
    axis = axis / (axis.norm(dim=-1, keepdim=True) + eps)
    half = 0.5 * angle
    s = torch.sin(half)[..., None]
    c = torch.cos(half)[..., None]
    return torch.cat([axis * s, c], dim=-1)
    
def pose_delta_axis_angle(poses):
    # poses: (T, N, 3)
    T, N, _ = poses.shape
    q = axis_angle_to_quat(poses)                # (T, N, 4)
    q_prev = q[:-1]                               # (T-1, N, 4)
    q_curr = q[1:]
    q_rel  = quat_mul(quat_conjugate(q_prev), q_curr)
    q_rel  = quat_normalize(q_rel)
    diff_angle, diff_axis = quat_to_angle_axis(q_rel)  # (T-1, N), (T-1, N, 3)
    return diff_angle, diff_axis

def reverse_rotate(pose_t1_aa, diff_angle, diff_axis):
    """
    Rotate pose at t+1 back to pose at t. Returns axis-angle vector (...,3).
    """
    q_t1   = quat_normalize(axis_angle_to_quat(pose_t1_aa))
    q_corr = quat_from_axis_angle(diff_axis, -diff_angle)   # inverse of delta
    q_t    = quat_normalize(quat_mul(q_t1, q_corr))
    ang, ax = quat_to_angle_axis(q_t)                       # tensors
    return ax * ang.unsqueeze(-1)                           # axis-angle vector

def forward_rotate(pose_t_aa, diff_angle, diff_axis):
    """
    Rotate pose at t forward to pose at t+1. Returns axis-angle vector (...,3).
    """
    q_t    = quat_normalize(axis_angle_to_quat(pose_t_aa))
    q_corr = quat_from_axis_angle(diff_axis,  diff_angle)   # apply delta
    q_t1   = quat_normalize(quat_mul(q_t, q_corr))
    ang, ax = quat_to_angle_axis(q_t1)
    return ax * ang.unsqueeze(-1)

def fix_joint_poses_simple(poses, joint_idx, angle_thresh=0.2, max_passes=20):
    """
    poses: (T, 156) or (T, 52, 3) axis-angle.  Assumes segment 0 is good.
    For any boundary (t -> t+1) with diff_angle > angle_thresh for this joint,
    take the segment after that boundary and reverse-rotate all its frames by that boundary's delta.
    After fixing one segment, recompute diffs and continue, up to max_passes.
    Returns: poses_fixed (same shape), fixed_boundaries (list of t indices used)
    """
    # reshape to (T, 52, 3)
    if poses.ndim == 2 and poses.shape[1] == 156:
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).clone()
        else:
            poses_reshaped = poses.reshape(poses.shape[0], 52, 3).copy()
    elif poses.ndim == 3 and poses.shape[1:] == (52, 3):
        if isinstance(poses, torch.Tensor):
            poses_reshaped = poses.clone()
        else:
            poses_reshaped = poses.copy()
    else:
        raise ValueError("poses must be (T,156) or (T,52,3) axis-angle")

    torch_device = None
    np_input = not isinstance(poses_reshaped, torch.Tensor)
    if np_input:
        poses_t = torch.tensor(poses_reshaped, dtype=torch.float32)
    else:
        poses_t = poses_reshaped.float()
        torch_device = poses_t.device

    T = poses_t.shape[0]
    fixed_boundaries = []

    for _ in range(max_passes):
        diff_angle, diff_axis = pose_delta_axis_angle(poses_t)   # (T-1, N), (T-1, N, 3)
        # flips for this joint
        da = diff_angle[:, joint_idx]       # (T-1,)
        ax = diff_axis[:, joint_idx, :]     # (T-1,3)

        # find boundaries with big jumps
        flip_idxs = torch.nonzero(da > angle_thresh, as_tuple=False).flatten().tolist()
        if not flip_idxs:
            break

        # Always take the earliest boundary first, fix the segment after it.
        b = flip_idxs[0]                     # boundary between b (good) and b+1.. (bad)
        angle_b = da[b]
        axis_b  = ax[b]

        # reverse-rotate ALL frames from b+1 to the next flip (or to end)
        next_b = next((k for k in flip_idxs[1:] if k > b), None)
        seg_start = b + 1
        seg_end = (next_b if next_b is not None else (T-1))  # inclusive end boundary for rotations
        # we rotate frames seg_start..(T-1) for the target joint; per your spec, whole segment to the end of that segment
        target_slice = slice(seg_start, seg_end + 1)

        # apply reverse rotation to that joint across the segment
        poses_t[target_slice, joint_idx, :] = reverse_rotate(
            poses_t[target_slice, joint_idx, :], angle_b, axis_b
        )
        fixed_boundaries.append(int(b))
        # loop continues: recompute diffs and handle next earliest boundary (after update)

    poses_fixed = poses_t
    if np_input:
        poses_fixed = poses_fixed.cpu().numpy()
        # reshape back to original
        if poses.ndim == 2:
            poses_fixed = poses_fixed.reshape(T, 156)
    else:
        if poses.ndim == 2:
            poses_fixed = poses_fixed.reshape(T, 156)

    return poses_fixed, fixed_boundaries
