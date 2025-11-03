import numpy as np
from scipy.spatial.transform import Rotation

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

def calculate_axis_angle_difference(pose1, pose2):
    """
    Calculate the difference between two rotations using rotation matrices.
    This is often more stable than axis-angle comparisons.
    
    Args:
        pose1 (np.array): First pose vector (3D rotation vector)
        pose2 (np.array): Second pose vector (3D rotation vector)
    
    Returns:
        tuple: (angle_diff_degrees, relative_rotation_axis)
    """
    from scipy.spatial.transform import Rotation
    
    # Convert axis-angle to rotation objects
    rot1 = Rotation.from_rotvec(pose1)
    rot2 = Rotation.from_rotvec(pose2)
    
    # Calculate the relative rotation using matrix multiplication
    relative_rot = rot1.inv().as_matrix() @ rot2.as_matrix()
    relative_rot = Rotation.from_matrix(relative_rot)
    
    # Get the rotation angle in degrees
    angle_diff = np.rad2deg(relative_rot.magnitude())
    
    # Get the rotation axis
    relative_axis = relative_rot.as_rotvec()
    if np.linalg.norm(relative_axis) > 1e-6:
        relative_axis = relative_axis / np.linalg.norm(relative_axis)
    else:
        relative_axis = np.array([1.0, 0.0, 0.0])  # fallback
    
    return angle_diff, relative_axis

def detect_flips(poses, joint_idx, threshold=20):
    """
    Detect flips (large axis-angle changes) for a joint.
    
    Args:
        poses: (T, 156) pose parameters
        joint_idx: joint index to analyze
        threshold: angle threshold in degrees
    
    Returns:
        tuple: (flip_indices, angle_diffs, relative_axes)
    """
    T = poses.shape[0]
    pose_start = JOINT_TO_POSE_MAPPING[joint_idx]
    flip_indices = []
    angle_diffs = []
    relative_axes = []
    
    for t in range(T-1):
        pose1 = poses[t, pose_start:pose_start+3]
        pose2 = poses[t+1, pose_start:pose_start+3]
        
        # Calculate angle difference and relative rotation axis
        angle_diff, relative_axis = calculate_axis_angle_difference(pose1, pose2)
        angle_diffs.append(angle_diff)
        relative_axes.append(relative_axis)
        # Check if this is a flip (large angle change)
        if angle_diff > threshold:
            flip_indices.append(t)
    
    return flip_indices, angle_diffs, relative_axes

def smooth_flips(poses, joint_optimization_mask, window_size=10, threshold=5):
    """
    Smooth remaining large flips for ALL joints that were fixed in previous steps.
    
    Args:
        poses: (T, 156) pose parameters
        joint_optimization_mask: (8,) mask indicating which joints were optimized
        window_size: number of adjacent frames to use for smoothing (default: 10)
    
    Returns:
        poses: smoothed pose parameters
    """
    T = poses.shape[0]
    poses_smoothed = poses.copy()
    
    # Joint mapping: joint index -> pose parameter indices
    joint_to_pose_mapping = {
        0: (39, 42),   # left_collar: 39:42
        1: (42, 45),   # right_collar: 42:45
        2: (48, 51),   # left_shoulder: 48:51
        3: (51, 54),   # right_shoulder: 51:54
        4: (54, 57),   # left_elbow: 54:57
        5: (57, 60),   # right_elbow: 57:60
        6: (60, 63),   # left_wrist: 60:63
        7: (63, 66)    # right_wrist: 63:66
    }
    
    total_flips_smoothed = 0
    
    # Process each joint that was optimized
    for joint_idx in range(8):
        if not joint_optimization_mask[joint_idx]:
            continue
            
        # Get pose parameter indices for this joint
        pose_start, pose_end = joint_to_pose_mapping[joint_idx]
        # Use detect_flips function with 10-degree threshold for this joint
        # The joint index in detect_flips corresponds to the actual joint index in the pose parameters
        joint_flip_indices, angle_diffs, _ = detect_flips(poses, joint_idx + 13 + (joint_idx > 1), threshold)
        # print(f"joint {joint_idx + 13 + (joint_idx > 1)} has {len(joint_flip_indices)} flips at frames {joint_flip_indices}")
        if len(joint_flip_indices) == 0:
            continue
        # sort the joint_flip_indices by angle_diffs
        joint_flip_indices = [x for _, x in sorted(zip(angle_diffs, joint_flip_indices))]
        # Smooth flips for this joint
        for flip_frame in joint_flip_indices:
            if flip_frame < T:
                # Calculate window boundaries
                start_frame = max(0, flip_frame - window_size // 2 + 1)
                end_frame = min(T, flip_frame + window_size // 2 + 1)
                # print(f"fixing joint {joint_idx + 13 + (joint_idx > 1)} of flip at frame {flip_frame} from {start_frame} to {end_frame}")
                # Extract joint poses for the window
                window_poses = poses[start_frame:end_frame, pose_start:pose_end].copy()
                num_frames = end_frame - start_frame
                flip_idx_in_window = flip_frame - start_frame
                
                # Get the specific poses right before and after the flip as reference poses
                reference_before_flip = window_poses[flip_idx_in_window]  # Frame right before the flip
                reference_after_flip = window_poses[flip_idx_in_window + 1]       # Frame right after the flip

                for i in range(num_frames // 2):
                    if i <= flip_idx_in_window and flip_idx_in_window != 0:
                        # Before the flip: use reference_after_flip as reference (poses after flip)
                        distance_from_flip = flip_idx_in_window - i
                        # Weight decreases as we move away from the flip (max 0.4 at flip boundary)
                        reference_weight = max(0.1, 0.46 * (1.0 - i / flip_idx_in_window))
                        # reference_weight = 1.0
                        if i == 0:
                            poses_smoothed[flip_frame - i, pose_start:pose_end] = (
                                (1 - reference_weight) * window_poses[flip_idx_in_window - i] + reference_weight * window_poses[flip_idx_in_window + 1 + i]
                            )
                            poses_smoothed[flip_frame + i + 1, pose_start:pose_end] = (
                                (1 - reference_weight) * window_poses[flip_idx_in_window + 1 + i] + reference_weight * window_poses[flip_idx_in_window - i]
                            )
                        else:
                            if flip_frame - i >= 0:
                                poses_smoothed[flip_frame - i, pose_start:pose_end] = (
                                    (1 - reference_weight) * window_poses[flip_idx_in_window - i] + reference_weight * poses_smoothed[flip_frame - i + 1, pose_start:pose_end]
                                )
                            if flip_frame + i + 1 < T:
                                poses_smoothed[flip_frame + i + 1, pose_start:pose_end] = (
                                    (1 - reference_weight) * window_poses[flip_idx_in_window + 1 + i] + reference_weight * poses_smoothed[flip_frame + i + 1, pose_start:pose_end]
                                )
                total_flips_smoothed += 1
    
    return poses_smoothed