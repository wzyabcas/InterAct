import torch
import torch.nn.functional as F
from utils import vertex_normals
from render.mesh_viz import visualize_body_obj
from bone_lists import bone_list_behave, bone_list_omomo
from loss import point2point_signed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from human_body_prior.body_model.body_model import BodyModel

LEFT_COLLAR = 13
RIGHT_COLLAR = 14
LEFT_SHOULDER = 16
RIGHT_SHOULDER = 17
LEFT_ELBOW = 18
RIGHT_ELBOW = 19
LEFT_WRIST = 20
RIGHT_WRIST = 21

def hand_joints_smoothing_loss(
    joints,                        # (T, J, 3) torch.Tensor
    fixed_joints_mask=None,        # (T, 8) bool (collar/shoulder/elbow/wrist L/R), optional
    joint_optimization_mask=None,  # (8,) bool which of the above 8 joints were optimized, optional
    hand_joint_ids=None,           # list[int] in 0..J-1; default wrists+fingers
    neighbor_radius: int = 1,      # include +/- this many frames around fixed ones
    use_root_relative: bool = True,
    root_joint_index: int = 0,
    per_joint_weights=None,        # optional (K,) weights for selected hand joints
    weight_vel: float = 0.5,
    weight_accel: float = 0.5,
):
    """
    Temporal smoothing on HAND JOINT 3D positions (no FK inside).
    Only frames near fixed frames get gradients (fixed frames +/- neighbor_radius).
    """
    assert torch.is_tensor(joints) and joints.ndim == 3 and joints.size(-1) == 3, "joints must be (T,J,3)"
    dev = joints.device
    dtype = joints.dtype
    T, J, _ = joints.shape

    # ---- default hand joint set (wrists + fingertip chains used earlier) ----
    if hand_joint_ids is None:
        # wrists (20,21) + left fingertips bases (25,28,31,34,37) + right (40,43,46,49,52)
        hand_joint_ids = [20, 21, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
    Jsel = torch.as_tensor(hand_joint_ids, device=dev, dtype=torch.long)

    # ---- frame mask (which frames get smoothed) ----
    if fixed_joints_mask is None or joint_optimization_mask is None:
        frame_mask = torch.ones(T, dtype=torch.bool, device=dev)
    else:
        if not torch.is_tensor(fixed_joints_mask):
            fixed_joints_mask = torch.as_tensor(fixed_joints_mask, dtype=torch.bool, device=dev)
        if not torch.is_tensor(joint_optimization_mask):
            joint_optimization_mask = torch.as_tensor(joint_optimization_mask, dtype=torch.bool, device=dev)

        # frames where any optimized joint was fixed
        active = fixed_joints_mask[:, joint_optimization_mask]   # (T, N_active)
        frame_mask = active.any(dim=1)                           # (T,)

        # dilate by neighbor_radius (include neighbors)
        if neighbor_radius > 0 and frame_mask.any():
            idx = torch.where(frame_mask)[0]
            fm = frame_mask.clone()
            for d in range(1, neighbor_radius + 1):
                left  = (idx - d).clamp_min(0)
                right = (idx + d).clamp_max(T - 1)
                fm[left]  = True
                fm[right] = True
            frame_mask = fm

    # ---- select hand joints, optionally make root-relative ----
    P = joints[:, Jsel, :]  # (T, K, 3)
    if use_root_relative:
        root = joints[:, root_joint_index, :].unsqueeze(1)  # (T,1,3)
        P = P - root

    # per-joint weights (K,)
    if per_joint_weights is not None:
        wj = torch.as_tensor(per_joint_weights, device=dev, dtype=dtype).view(1, -1, 1)  # (1,K,1)
    else:
        wj = None

    # gate gradients to masked frames only
    mask3 = frame_mask.view(T, 1, 1)
    P_sel = P * mask3 + P.detach() * (~mask3)

    loss = torch.zeros((), device=dev, dtype=dtype)

    # velocity term
    if T >= 2 and weight_vel != 0.0:
        V = P_sel[1:] - P_sel[:-1]   # (T-1, K, 3)
        if wj is not None:
            V = V * wj
        loss_vel = (V * V).mean()
        loss = loss + weight_vel * loss_vel

    # acceleration term
    if T >= 3 and weight_accel != 0.0:
        A = P_sel[2:] - 2 * P_sel[1:-1] + P_sel[:-2]  # (T-2, K, 3)
        if wj is not None:
            A = A * wj
        loss_acc = (A * A).mean()
        loss = loss + weight_accel * loss_acc

    return loss


def compute_palm_loss(
    joints,                      # (T, J, 3)
    verts_obj_transformed,       # (T, N, 3)
    obj_normals,                 # (T, N, 3)
    contact_mask,                # (T,) bool
    fixed_joints_mask,           # (T, 8) bool
    is_left_hand=True,
    joint_optimization_mask=None,
    K: int = 700,
    align_mode: str = "antiparallel",      # "abs" | "parallel" | "antiparallel",
    epoch: int = 0
):
    """
    Palm-facing loss using OBJECT NORMALS around the palm centroid.
    Prints per-frame facing loss for debugging.
    """

    device = joints.device
    T = joints.shape[0]
    N = verts_obj_transformed.shape[1]
    K = min(K, N)

    # --- pick hand joint indices ---
    if is_left_hand:
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 20, 27, 33
        wrist_col = 6
        flip_normal = True
    else:
        IDX_WRIST, IDX_INDEX, IDX_PINKY = 21, 42, 48
        wrist_col = 7
        flip_normal = False

    wrist = joints[:, IDX_WRIST, :]
    index = joints[:, IDX_INDEX, :]
    pinky = joints[:, IDX_PINKY, :]

    # Palm normal
    v1 = index - wrist
    v2 = pinky - wrist
    palm_normals = torch.cross(v1, v2, dim=1)
    if flip_normal:
        palm_normals = -palm_normals
    palm_normals = F.normalize(palm_normals, dim=1, eps=1e-8)

    # Palm centroid
    palm_centroid = (wrist + index + pinky) / 3.0  # (T,3)

    # Distances to object vertices
    dists = torch.norm(verts_obj_transformed - palm_centroid.unsqueeze(1), dim=2)  # (T,N)
    _, topi = torch.topk(dists, k=K, dim=1, largest=False)  # (T,K)

    # Gather K object normals
    obj_normals = F.normalize(obj_normals, dim=2, eps=1e-8)
    obj_n_K = torch.gather(
        obj_normals, dim=1, index=topi.unsqueeze(-1).expand(T, K, 3)
    )  # (T,K,3)

    # Cosine scores
    dots = (palm_normals.unsqueeze(1) * obj_n_K).sum(dim=2)  # (T,K)

    if align_mode == "abs":
        score = dots.abs()
    elif align_mode == "parallel":
        score = dots
    elif align_mode == "antiparallel":
        score = -dots
    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    facing_loss_per_frame = 1.0 - score.mean(dim=1)  # (T,)

    # Mask frames
    if contact_mask.sum() == 0:
        return torch.zeros((), device=device, dtype=joints.dtype)

    combined_mask = contact_mask & fixed_joints_mask[:, wrist_col]

    if combined_mask.any():
        # Print debug info
        if epoch == 0 or epoch == 499:
            frame_ids = combined_mask.nonzero(as_tuple=True)[0]
            # for f in frame_ids:
        return facing_loss_per_frame[combined_mask].mean()
    else:
        return torch.zeros((), device=device, dtype=joints.dtype)

def compute_finger_loss(joints, verts_obj_transformed, contact_mask, frame_mask, is_left_hand=True):
    """
    Compute finger distance loss: penalize when pinky and index finger distances to object are different during contact.
    Only applies to frames specified by frame_mask (e.g., frames close to object).
    
    Args:
        joints: Joint positions (T, J, 3)
        verts_obj_transformed: Object vertices (T, M, 3)
        contact_mask: Boolean mask (T,) indicating when hand is in contact
        frame_mask: Boolean mask (T,) indicating which frames to compute loss for
        is_left_hand: Whether computing for left or right hand
    
    Returns:
        distance_loss: Loss penalizing finger distance differences
    """
    # Define finger end joint indices
    if is_left_hand:
        IDX_INDEX_END = 25  # Left index finger end joint
        IDX_MIDDLE_END = 28  # Left middle finger end joint
        IDX_PINKY_END = 31  # Left pinky finger end joint
    else:
        IDX_INDEX_END = 40  # Right index finger end joint
        IDX_MIDDLE_END = 43  # Right middle finger end joint
        IDX_PINKY_END = 46  # Right pinky finger end joint
    
    # Extract finger end joint positions
    index_end = joints[:, IDX_INDEX_END, :]  # (T, 3)
    middle_end = joints[:, IDX_MIDDLE_END, :]
    pinky_end = joints[:, IDX_PINKY_END, :]  # (T, 3)
    
    # Compute distances from finger end joints to nearest object point
    index_rel_to_obj = verts_obj_transformed - index_end.unsqueeze(1)  # (T, N, 3)
    middle_rel_to_obj = verts_obj_transformed - middle_end.unsqueeze(1)  # (T, N, 3)
    pinky_rel_to_obj = verts_obj_transformed - pinky_end.unsqueeze(1)  # (T, N, 3)
    
    index_dists = index_rel_to_obj.norm(dim=2)  # (T, N)
    middle_dists = middle_rel_to_obj.norm(dim=2)  # (T, N)
    pinky_dists = pinky_rel_to_obj.norm(dim=2)  # (T, N)
    
    index_min_dist, _ = index_dists.min(dim=1)  # (T,)
    middle_min_dist, _ = middle_dists.min(dim=1)  # (T,)
    pinky_min_dist, _ = pinky_dists.min(dim=1)  # (T,)
    
    # Compute distance differences
    finger_dist_diff = torch.abs(index_min_dist - pinky_min_dist) + torch.abs(middle_min_dist - pinky_min_dist) + torch.abs(index_min_dist - middle_min_dist) # (T,)
    
    # Only apply loss when hand is in contact AND frame is in the specified mask
    combined_mask = contact_mask & frame_mask  # (T,) - both contact AND in frame mask
    
    if combined_mask.sum() == 0:
        return torch.tensor(0.0, device=joints.device)
    
    # Loss: penalize when finger distances are different during contact
    distance_loss = torch.mean(torch.where(
        combined_mask, 
        finger_dist_diff,  # Direct penalty: larger difference = higher loss
        torch.zeros_like(finger_dist_diff)
    ))
    
    # Additional penalty: penalize total distance to object during contact
    total_distance_penalty = torch.mean(torch.where(
        combined_mask,
        index_min_dist + pinky_min_dist,  # Penalize total distance to object
        torch.zeros_like(index_min_dist)
    ))
    
    return distance_loss + total_distance_penalty

def compute_hand_penetration_loss(
    verts_full: torch.Tensor,         # (T, V, 3) all human vertices
    verts_obj_transformed: torch.Tensor,
    obj_normals: torch.Tensor,
    fixed_joints_mask: torch.Tensor,  # (T, 8)
    lhand_idx: torch.Tensor,
    rhand_idx: torch.Tensor,
    thresh: float = 0.0,
    detach_opposite: bool = True,
    epoch: int = 0,
):
    """
    Compute penetration loss for both hands, ensuring no gradient flow from the opposite hand.

    Returns:
      pen_left, pen_right
    """

    device = verts_full.device

    # Optionally detach the *other* hand vertices
    if detach_opposite:
        verts_left  = verts_full.clone()
        verts_right = verts_full.clone()

        # Detach right hand when computing left
        verts_left[:, rhand_idx, :] = verts_left[:, rhand_idx, :].detach()
        # Detach left hand when computing right
        verts_right[:, lhand_idx, :] = verts_right[:, lhand_idx, :].detach()
    else:
        verts_left = verts_full
        verts_right = verts_full

    # --- Left hand ---
    pen_left = _compute_single_hand_penetration(
        verts_left[:, lhand_idx, :],
        verts_obj_transformed, obj_normals, fixed_joints_mask[:, 6],
        thresh=thresh, epoch=-1
    )

    # --- Right hand ---
    pen_right = _compute_single_hand_penetration(
        verts_right[:, rhand_idx, :],
        verts_obj_transformed, obj_normals, fixed_joints_mask[:, 7],
        thresh=thresh, epoch=epoch
    )

    return pen_left, pen_right


def _compute_single_hand_penetration(
    hand_verts, verts_obj_transformed, obj_normals,
    hand_fixed_mask, thresh=0.0, eps=1e-8, epoch=0
):
    """
    Per-hand penetration loss computed only from vertices with depth ≤ 0.1.
    Frames contribute loss only from shallow penetrations; vertices deeper than 0.1 are excluded.
    """
    device = hand_verts.device
    dtype  = hand_verts.dtype
    T = hand_verts.shape[0]

    total_penetration_loss = torch.zeros((), device=device, dtype=dtype)

    # signed distances: sbj2obj[t,v] is distance for vertex v in frame t
    o2h_signed, sbj2obj, *_ = point2point_signed(
        hand_verts, verts_obj_transformed.to(device),
        y_normals=obj_normals, return_vector=True
    )

    # penetration mask and depths
    depths   = (thresh - sbj2obj).clamp_min(0.0)        # positive depth = penetration amount
    pen_mask = (sbj2obj < thresh)                       # (T, Vh)
    shallow_mask = depths <= 0.1                        # exclude depths > 0.1
    effective_mask = pen_mask & shallow_mask            # only shallow penetrations

    counts = effective_mask.sum(dim=-1)                 # (T,) number of shallow penetrating verts per frame
    # deepest shallow penetration per frame (0 if none)
    max_depths = (depths * effective_mask).max(dim=-1).values

    # Frame mask: only keep frames with shallow penetration, fixed joint, and reasonably deep (≥0.01)
    frame_mask = (counts > 0) & hand_fixed_mask & (0.01 <= max_depths)

    if frame_mask.any():
        depth_sum = (depths * effective_mask).sum(dim=-1)  # (T,) sum of shallow depths per frame
        per_frame = depth_sum                           # loss contribution per frame

        if epoch % 50 == 0:
            frame_ids = frame_mask.nonzero(as_tuple=True)[0]
            for f in frame_ids:
                num_pene = counts[f].item()
                if num_pene > 0:
                    max_depth = depths[f].max().item()
                    frame_loss = per_frame[f].item()

        total_penetration_loss = per_frame[frame_mask].sum()
        return total_penetration_loss / T
    else:
        return torch.tensor(0.0, device=device, dtype=dtype)
    # Return average over T frames (so sequence length doesn’t bias)
    


def optimize_poses(poses, betas, trans, gender, verts_obj_transformed, obj_normals,
                                     fix_tracker, distance_info, rhand_idx, lhand_idx,smpl_model,
                                     num_epochs=500, lr=0.01, canonical_joints=None):
    """Two-stage optimization: Stage 1 optimizes only wrist poses, Stage 2 optimizes all joints.
    
    Stage 1: Only optimize wrist poses (left and right) to fix palm orientation
    Stage 2: Optimize all joints (collar, shoulder, elbow, wrist) with full loss functions
    
    This approach ensures that palm orientation loss only affects wrist poses in Stage 1,
    preventing unwanted gradients to other joints.
    """
    
    # Get which joints were fixed in which frames
    fixed_joints_mask = fix_tracker.get_fixed_frames_and_joints()  # Shape: (T, 8)
    # Convert to tensor and move to device
    fixed_joints_mask = torch.from_numpy(fixed_joints_mask).bool().to(device)
    
    # Extract all joint poses into one tensor
    all_joint_poses = np.concatenate([
        poses[:, 39:42],   # Left collar (joint 13)
        poses[:, 42:45],   # Right collar (joint 14) 
        poses[:, 48:51],   # Left shoulder (joint 16)
        poses[:, 51:54],   # Right shoulder (joint 17)
        poses[:, 54:57],   # Left elbow (joint 18)
        poses[:, 57:60],   # Right elbow (joint 19)
        poses[:, 60:63],   # Left wrist (joint 20)
        poses[:, 63:66],   # Right wrist (joint 21)
    ], axis=1)  # Shape: (T, 24) - 8 joints × 3 parameters each
    
    # Convert to tensor for optimization
    all_joint_poses_tensor = torch.from_numpy(all_joint_poses).float().to(device).requires_grad_(True)
    
    # Create reference poses (original poses)
    reference_all_joint_poses = torch.from_numpy(all_joint_poses).float().to(device)
    
    # Create a mask tensor that indicates which joints should be optimized
    # We want to optimize ALL frames for joints that were fixed in previous steps
    joint_optimization_mask = torch.zeros(8, dtype=torch.bool)  # Which joints to optimize
    for joint_idx in range(8):  # 8 joints
        # Mark joints that were fixed in any frame
        if fixed_joints_mask[:, joint_idx].any():  # If any frame was fixed for this joint
            joint_optimization_mask[joint_idx] = True  # Mark this joint for optimization
    selected_joints = torch.nonzero(joint_optimization_mask, as_tuple=False).flatten().tolist()
    
    # Print out for each joint which frames are fixed according to fixed_joints_mask
    joint_names = [
        "left_collar", "right_collar", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    for joint_idx in range(8):
        fixed_frames = torch.where(fixed_joints_mask[:, joint_idx])[0].cpu().numpy().tolist()
    # Create optimization mask for all poses: ALL frames for fixed joints
    optimization_mask = torch.zeros_like(all_joint_poses_tensor, dtype=torch.bool)
    for joint_idx in range(8):  # 8 joints
        pose_start = joint_idx * 3
        pose_end = pose_start + 3
        # Mark ALL frames for joints that were fixed
        if joint_optimization_mask[joint_idx]:
            optimization_mask[:, pose_start:pose_end] = True  # Mark all frames for this joint
    
    JOINT_156_SLICES = [(39,42),(42,45),(48,51),(51,54),(54,57),(57,60),(60,63),(63,66)]  # length 8

    # Early stopping setup
    best_loss = float('inf')
    best_poses = None
    patience_counter = 0
    patience = 200
    left_finger_mask = distance_info['left_close_mask'] & fixed_joints_mask[:, 6]
    right_finger_mask = distance_info['right_close_mask'] & fixed_joints_mask[:, 7]

    left_not_close_mask = ~distance_info['left_close_mask']  # (T,)
    right_not_close_mask = ~distance_info['right_close_mask']  # (T,)
    left_forearm_axis = canonical_joints[LEFT_WRIST] - canonical_joints[LEFT_ELBOW]  # (3,)
    right_forearm_axis = canonical_joints[RIGHT_WRIST] - canonical_joints[RIGHT_ELBOW]  # (3,)
    
    # Convert to PyTorch tensors and move to device
    left_forearm_axis = torch.from_numpy(left_forearm_axis).float().to(device)
    right_forearm_axis = torch.from_numpy(right_forearm_axis).float().to(device)
    
    # Normalize bone axes
    left_forearm_axis = left_forearm_axis / torch.norm(left_forearm_axis)
    right_forearm_axis = right_forearm_axis / torch.norm(right_forearm_axis)

    # ==================== STAGE 1: Optimize only wrist poses ====================
    sel_blocks = [slice(j*3, j*3+3) for j in selected_joints]
    sel_init = torch.cat([all_joint_poses_tensor[:, blk] for blk in sel_blocks], dim=1).detach().clone().to(device)
    sel_params = sel_init.clone().requires_grad_(True)  # (T, 3*K)

    sel_ref = torch.cat([reference_all_joint_poses[:, blk] for blk in sel_blocks], dim=1)

    wrist_optimizer = torch.optim.Adam([sel_params], lr=0.001)
    wrist_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(wrist_optimizer, mode='min', factor=0.7, patience=150, verbose=True)

    # Stage 1 optimization loop
    stage1_epochs = num_epochs  # Use half of total epochs for stage 1
    best_wrist_loss = float('inf')
    best_wrist_poses = None
    T = poses.shape[0]

    for epoch in range(stage1_epochs):
        wrist_optimizer.zero_grad()
        
        # Create poses tensor with only wrist poses updated
        poses_tensor = torch.from_numpy(poses).float().to(device)
        offset = 0
        for j in selected_joints:
            start156, end156 = JOINT_156_SLICES[j]
            poses_tensor[:, start156:end156] = sel_params[:, offset:offset+3]
            offset += 3

        model = smpl_model
        
        # ---- Forward pass for other losses (posedirs ON) ----
        output = model(
            pose_body=poses_tensor[:, 3:66],
            pose_hand=poses_tensor[:, 66:156],
            betas=torch.from_numpy(betas[None, :]).repeat(poses_tensor.shape[0], 1).float().to(device),
            root_orient=poses_tensor[:, :3],
            trans=torch.from_numpy(trans).float().to(device)
        )
            
        joints = output.Jtr
 
        if epoch < 200:
            pen_left, pen_right = compute_hand_penetration_loss(
                output.v, verts_obj_transformed, obj_normals, fixed_joints_mask,
                lhand_idx, rhand_idx, detach_opposite=True, epoch=epoch
            )
            wrist_ref_loss = torch.mean((sel_params - sel_ref) ** 2)
            left_palm_loss = compute_palm_loss(
                joints, verts_obj_transformed, obj_normals, distance_info['left_close_mask'], fixed_joints_mask, is_left_hand=True, joint_optimization_mask=joint_optimization_mask, epoch=epoch
            )
            right_palm_loss = compute_palm_loss(
                joints, verts_obj_transformed, obj_normals, distance_info['right_close_mask'], fixed_joints_mask, is_left_hand=False, joint_optimization_mask=joint_optimization_mask, epoch=epoch
            )
            palm_loss = left_palm_loss + right_palm_loss

            finger_loss = compute_finger_loss(
                joints, verts_obj_transformed, distance_info['left_close_mask'], left_finger_mask, is_left_hand=True
            ) + compute_finger_loss(
                joints, verts_obj_transformed, distance_info['right_close_mask'], right_finger_mask, is_left_hand=False
            )
        elif epoch < 400:
            pen_left, pen_right = compute_hand_penetration_loss(
                output.v, verts_obj_transformed, obj_normals, fixed_joints_mask,
                lhand_idx, rhand_idx, detach_opposite=True, epoch=epoch
            )
            wrist_ref_loss = torch.mean((sel_params - sel_ref) ** 2)
            finger_loss = torch.tensor(0.0)
            palm_loss = torch.tensor(0.0)
        else:
            pen_left, pen_right = torch.tensor(0.0), torch.tensor(0.0)
            wrist_ref_loss = torch.tensor(0.0)
            finger_loss = torch.tensor(0.0)
            palm_loss = torch.tensor(0.0)

        hand_smooth_loss = hand_joints_smoothing_loss(
            joints=joints,                      # (T,J,3) from your model forward
            fixed_joints_mask=fixed_joints_mask,     # (T,8) bool
            joint_optimization_mask=joint_optimization_mask,  # (8,) bool
            hand_joint_ids=[20,21,25,28,31,34,37,40,43,46,49,52],
            neighbor_radius=1,
            use_root_relative=True,
            root_joint_index=0,
            per_joint_weights=None,                  # or e.g. wrist-heavy weights
            weight_vel=0.25,
            weight_accel=0.5,
        )
        # Total loss for Stage 1
        total_wrist_loss = 10 * hand_smooth_loss + 0.1 * pen_left + 0.1 * pen_right+ 10 * wrist_ref_loss + 0.5 * finger_loss

        # Check for improvement
        if total_wrist_loss < best_wrist_loss - 1e-6:
            best_wrist_loss = total_wrist_loss.detach()
            best_sel_params = sel_params.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Stage 1 early stopping at epoch {epoch}")
            break

        # if epoch % 20 == 0:

        total_wrist_loss.backward()
        torch.nn.utils.clip_grad_norm_([sel_params], max_norm=1.0)
        wrist_optimizer.step()
        wrist_scheduler.step(total_wrist_loss)
        
    
    # Update all_joint_poses_tensor with optimized wrist poses
    if best_sel_params is None:
        best_sel_params = sel_params.detach()

    with torch.no_grad():
        off = 0
        for j in selected_joints:
            all_joint_poses_tensor[:, j*3:(j+1)*3] = best_sel_params[:, off:off+3]
            off += 3

    # Refresh reference for Stage 2
    reference_all_joint_poses = all_joint_poses_tensor.clone().detach()
    
    # Reconstruct final poses using best poses if available
    final_poses = poses.copy()
    poses_to_use = best_poses if best_poses is not None else all_joint_poses_tensor
    
    # Create final combined poses: optimized for fixed frames, original for non-fixed frames
    final_combined_poses = poses_to_use.clone()
    final_combined_poses[~optimization_mask] = reference_all_joint_poses[~optimization_mask]
    
    final_poses[:, 39:42] = final_combined_poses[:, :3].detach().cpu().numpy()      # Left collar
    final_poses[:, 42:45] = final_combined_poses[:, 3:6].detach().cpu().numpy()     # Right collar
    final_poses[:, 48:51] = final_combined_poses[:, 6:9].detach().cpu().numpy()     # Left shoulder
    final_poses[:, 51:54] = final_combined_poses[:, 9:12].detach().cpu().numpy()    # Right shoulder
    final_poses[:, 54:57] = final_combined_poses[:, 12:15].detach().cpu().numpy()  # Left elbow
    final_poses[:, 57:60] = final_combined_poses[:, 15:18].detach().cpu().numpy()  # Right elbow
    final_poses[:, 60:63] = final_combined_poses[:, 18:21].detach().cpu().numpy()  # Left wrist
    final_poses[:, 63:66] = final_combined_poses[:, 21:24].detach().cpu().numpy()  # Right wrist
    
    return final_poses