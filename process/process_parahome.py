import os
import sys
import shutil
import json
import pickle
import argparse
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from pathlib import Path
from datetime import datetime
import smplx
from multiprocessing import Pool, cpu_count, set_start_method
from functools import partial

# Set multiprocessing start method to 'spawn' for CUDA compatibility
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Gravity Transformation Utilities ====================

def _normalize(v, eps=1e-8):
    """Normalize a vector."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def _rot_between(a, b):
    """Compute rotation matrix to rotate vector a to vector b."""
    a = _normalize(a).reshape(3,)
    b = _normalize(b).reshape(3,)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = _normalize(np.cross(a, [1,0,0])) if abs(a[0]) < 0.9 else _normalize(np.cross(a, [0,1,0]))
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3, dtype=np.float64)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c)/(s*s))


def _leftmul_rotvec(rotvecs, R_left):
    """Left-multiply rotation matrix to rotation vectors."""
    Rt = Rotation.from_rotvec(rotvecs)      # (T,)
    Rl = Rotation.from_matrix(R_left)
    return (Rl * Rt).as_rotvec()


def closest_axis_unit(v):
    """Return the unit axis in {±X, ±Y, ±Z} that has max |dot| with v (and preserves sign)."""
    v = _normalize(v).reshape(3,)
    axes = np.array([[ 1,0,0],[0, 1,0],[0,0, 1],
                     [-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    dots = axes @ v
    return axes[np.argmax(np.abs(dots))]


def build_smplx_model_parahome(gender, num_betas=20, model_path='models'):
    """Build SMPLX model for ParaHome."""
    smplx_model = smplx.create(
        model_path=model_path,
        model_type="smplx",
        flat_hand_mean=True,
        use_pca=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        gender=gender,
        ext='pkl'
    ).to(device)
    return smplx_model


def get_smplx_joints(poses, betas, trans, gender, num_betas=20):
    """Get SMPLX joints from poses, betas, trans."""
    T = poses.shape[0]
    smpl_model = build_smplx_model_parahome(gender, num_betas)
    
    # Extract pose components
    global_orient = poses[:, :3]
    body_pose = poses[:, 3:66]
    hand_pose = poses[:, 66:156]
    lhand_pose = hand_pose[:, :45]
    rhand_pose = hand_pose[:, 45:90]
    
    # Reshape
    body_pose_reshaped = body_pose.reshape((T, 21, 3))
    lhand_pose_reshaped = lhand_pose.reshape((T, 15, 3))
    rhand_pose_reshaped = rhand_pose.reshape((T, 15, 3))
    
    # Convert to tensors
    with torch.no_grad():
        smplx_output = smpl_model(
            betas=torch.from_numpy(betas[None, :]).repeat(T, 1).float().to(device),
            return_verts=True,
            body_pose=torch.from_numpy(body_pose_reshaped).float().to(device),
            left_hand_pose=torch.from_numpy(lhand_pose_reshaped).float().to(device),
            right_hand_pose=torch.from_numpy(rhand_pose_reshaped).float().to(device),
            global_orient=torch.from_numpy(global_orient).float().to(device),
            transl=torch.from_numpy(trans).float().to(device),
            jaw_pose=torch.zeros([T, 3]).float().to(device),
            leye_pose=torch.zeros([T, 3]).float().to(device),
            reye_pose=torch.zeros([T, 3]).float().to(device),
            expression=torch.zeros([T, 10]).float().to(device)
        )
    
    joints = smplx_output.joints.detach().cpu().numpy()
    verts = smplx_output.vertices.detach().cpu().numpy()
    
    return joints, verts


# ==================== Conversion Functions ====================

def extract_rotation_translation_from_matrix(transformation_matrix):
    """
    Extract rotation vector and translation from a 4x4 transformation matrix.
    
    Args:
        transformation_matrix: 4x4 numpy array
        
    Returns:
        rotation_vector: rotation as a 3D vector (axis-angle representation)
        translation: translation as a 3D vector
    """
    rotation_matrix = transformation_matrix[:3, :3]
    translation = transformation_matrix[:3, 3]
    
    # Convert rotation matrix to rotation vector (axis-angle)
    rotation = Rotation.from_matrix(rotation_matrix)
    rotation_vector = rotation.as_rotvec()
    
    return rotation_vector, translation


def apply_gravity_transformation(poses, betas, trans, gender, obj_angles_dict, obj_trans_dict):
    """
    Apply gravity transformation to align ParaHome data to interact coordinate system (+Y up).
    
    Args:
        poses: (T, 156) pose parameters
        betas: (num_betas,) shape parameters
        trans: (T, 3) translations
        gender: Gender string
        obj_angles_dict: Dict of object_part_name -> (T, 3) rotation vectors
        obj_trans_dict: Dict of object_part_name -> (T, 3) translations
    
    Returns:
        Transformed poses, trans, obj_angles_dict, obj_trans_dict
    """
    
    # Get original joints to estimate gravity direction
    joints0, verts0 = get_smplx_joints(poses, betas, trans, gender, num_betas=len(betas))
    
    # Pelvis world positions
    r = joints0[:, 0, :]  # (T, 3) pelvis positions
    
    # Estimate gravity from first frame (pelvis -> ankle-mid)
    j0 = joints0[0]
    # Pelvis idx=0, left ankle idx=7, right ankle idx=8
    up_src_cont = -_normalize(0.5 * (j0[7] + j0[8]) - j0[0])
    # Snap to nearest axis
    up_src = closest_axis_unit(up_src_cont)
    
    # Target: +Y up
    up_tgt = np.array([0, 1, 0], dtype=np.float64)
    
    # Compute rotation to align source to target
    R_can = _rot_between(up_src, up_tgt)
    
    # Choose pivot point: first frame pelvis
    p0 = r[0].astype(np.float64)
    
    # Split poses
    orang = poses[:, :3]  # global_orient
    rest = poses[:, 3:]   # body_pose + hand_pose
    
    # New orientations (left-multiply)
    orang_new = _leftmul_rotvec(orang, R_can)
    
    # Compute target pelvis positions
    r_target = (R_can @ (r - p0[None, :]).T).T + p0[None, :]
    
    # Rebuild SMPL with new orientations but zero translation to get local pelvis
    T = poses.shape[0]
    poses_upd = np.concatenate([orang_new, rest], axis=1).astype(np.float32)
    
    smpl_model = build_smplx_model_parahome(gender, num_betas=len(betas))
    
    # Extract pose components
    global_orient = poses_upd[:, :3]
    body_pose = poses_upd[:, 3:66]
    hand_pose = poses_upd[:, 66:156]
    lhand_pose = hand_pose[:, :45]
    rhand_pose = hand_pose[:, 45:90]
    
    # Reshape
    body_pose_reshaped = body_pose.reshape((T, 21, 3))
    lhand_pose_reshaped = lhand_pose.reshape((T, 15, 3))
    rhand_pose_reshaped = rhand_pose.reshape((T, 15, 3))
    
    with torch.no_grad():
        smpl_out = smpl_model(
            body_pose=torch.from_numpy(body_pose_reshaped).float().to(device),
            global_orient=torch.from_numpy(global_orient).float().to(device),
            left_hand_pose=torch.from_numpy(lhand_pose_reshaped).float().to(device),
            right_hand_pose=torch.from_numpy(rhand_pose_reshaped).float().to(device),
            betas=torch.from_numpy(betas[None, :]).repeat(T, 1).float().to(device),
            transl=torch.zeros([T, 3]).float().to(device),  # Zero translation
            jaw_pose=torch.zeros([T, 3]).float().to(device),
            leye_pose=torch.zeros([T, 3]).float().to(device),
            reye_pose=torch.zeros([T, 3]).float().to(device),
            expression=torch.zeros([T, 10]).float().to(device)
        )
    
    joints_local_new = smpl_out.joints.detach().cpu().numpy()
    r_local_new = joints_local_new[:, 0, :]  # Local pelvis
    
    # Solve for new translations
    trans_new = (r_target - r_local_new).astype(np.float32)
    
    # Transform all objects
    obj_angles_new_dict = {}
    obj_trans_new_dict = {}
    
    for obj_part_name, obj_angles in obj_angles_dict.items():
        obj_trans = obj_trans_dict[obj_part_name]
        
        # Rotate angles
        obj_angles_new = _leftmul_rotvec(obj_angles, R_can)
        
        # Rotate translations relative to pelvis
        obj_trans_new = (R_can @ (obj_trans - r).T).T + r_target
        
        obj_angles_new_dict[obj_part_name] = obj_angles_new.astype(np.float32)
        obj_trans_new_dict[obj_part_name] = obj_trans_new.astype(np.float32)
    
    # Get new vertices to find minimum Y
    with torch.no_grad():
        smpl_out_final = smpl_model(
            body_pose=torch.from_numpy(body_pose_reshaped).float().to(device),
            global_orient=torch.from_numpy(global_orient).float().to(device),
            left_hand_pose=torch.from_numpy(lhand_pose_reshaped).float().to(device),
            right_hand_pose=torch.from_numpy(rhand_pose_reshaped).float().to(device),
            betas=torch.from_numpy(betas[None, :]).repeat(T, 1).float().to(device),
            transl=torch.from_numpy(trans_new).float().to(device),
            jaw_pose=torch.zeros([T, 3]).float().to(device),
            leye_pose=torch.zeros([T, 3]).float().to(device),
            reye_pose=torch.zeros([T, 3]).float().to(device),
            expression=torch.zeros([T, 10]).float().to(device)
        )
    
    verts_new = smpl_out_final.vertices.detach().cpu().numpy()
    min_y_h = float(verts_new[0, :, 1].min())
    
    # Shift up so min_y >= 0
    shift_y = max(0.0, -min_y_h)
    if shift_y > 0:
        dy = np.array([0.0, shift_y, 0.0], dtype=np.float32)
        trans_new = trans_new + dy[None, :]
        for obj_part_name in obj_trans_new_dict:
            obj_trans_new_dict[obj_part_name] = obj_trans_new_dict[obj_part_name] + dy[None, :]
    
    # Combine new poses
    poses_new = np.concatenate([orang_new.astype(np.float32), rest.astype(np.float32)], axis=1)
    
    return poses_new, trans_new, obj_angles_new_dict, obj_trans_new_dict


def convert_parahome_to_interact(parahome_seq_root, output_dir, verbose=False):
    """
    Convert a single ParaHome sequence to interact format.
    
    Args:
        parahome_seq_root: Path to ParaHome smplx_seq directory (e.g., data/parahome/raw/smplx_seq/s1)
        output_dir: Output directory where human.npz and object_*.npz will be saved
        verbose: If True, print detailed information
    """
    if verbose:
        print(f"  Converting: {parahome_seq_root}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # ==================== Load ParaHome Data ====================
    
    # Load SMPL-X parameters
    smplx_params_path = os.path.join(parahome_seq_root, "smplx_params.pkl")
    if not os.path.exists(smplx_params_path):
        raise FileNotFoundError(f"smplx_params.pkl not found at {smplx_params_path}")
    
    with open(smplx_params_path, "rb") as f:
        smplx_params = pickle.load(f)
    
    smplx_beta = smplx_params['beta']  # Shape parameters
    gender = smplx_params['gender']     # Gender
    
    # Load SMPL-X pose
    smplx_pose_path = os.path.join(parahome_seq_root, "smplx_pose.pkl")
    if not os.path.exists(smplx_pose_path):
        raise FileNotFoundError(f"smplx_pose.pkl not found at {smplx_pose_path}")
    
    with open(smplx_pose_path, "rb") as f:
        smplx_pose = pickle.load(f)
    
    # Extract pose components (convert from torch tensors to numpy if needed)
    if hasattr(smplx_pose['body_pose'], 'cpu'):
        body_pose = smplx_pose['body_pose'].cpu().numpy()  # Shape: (T, 21, 3)
        global_orient = smplx_pose['global_orient'].cpu().numpy()  # Shape: (T, 3)
        transl = smplx_pose['transl'].cpu().numpy()  # Shape: (T, 3)
        hand_pose = smplx_pose['hand_pose'].cpu().numpy()  # Shape: (T, 30, 3)
    else:
        body_pose = smplx_pose['body_pose']
        global_orient = smplx_pose['global_orient']
        transl = smplx_pose['transl']
        hand_pose = smplx_pose['hand_pose']
    
    # Reshape if necessary
    if body_pose.ndim == 3:
        body_pose = body_pose.reshape((-1, 21, 3))
    if hand_pose.ndim == 3:
        hand_pose = hand_pose.reshape((-1, 30, 3))
    
    num_frames = body_pose.shape[0]
    
    # Construct full pose array
    # Format: [global_orient (3), body_pose (63), hand_pose (90)] = 156 per frame
    body_pose_flat = body_pose.reshape((num_frames, -1))  # (T, 63)
    hand_pose_flat = hand_pose.reshape((num_frames, -1))  # (T, 90)
    
    # Ensure global_orient is 2D
    if global_orient.ndim == 1:
        global_orient = global_orient.reshape((1, 3))
    
    # Concatenate: global_orient + body_pose + hand_pose
    poses = np.concatenate([global_orient, body_pose_flat, hand_pose_flat], axis=1)  # (T, 156)
    
    # Convert beta to numpy if needed
    if hasattr(smplx_beta, 'cpu'):
        betas = smplx_beta.cpu().numpy()
    else:
        betas = smplx_beta
    
    # Ensure betas is 1D
    if betas.ndim > 1:
        betas = betas.flatten()
    
    # ==================== Load Object Data ====================
    
    # Get the corresponding seq directory (not smplx_seq)
    # Assuming parahome_seq_root is like: data/parahome/raw/smplx_seq/s1
    # We need: data/parahome/raw/seq/s1
    seq_root = str(parahome_seq_root).replace('smplx_seq', 'seq')
    
    # Load object_in_scene.json
    object_in_scene_path = os.path.join(seq_root, "object_in_scene.json")
    if not os.path.exists(object_in_scene_path):
        if verbose:
            print(f"  Warning: object_in_scene.json not found at {object_in_scene_path}")
            print("  Skipping object conversion")
        # Still save human data
        human_output_path = os.path.join(output_dir, "human.npz")
        np.savez(human_output_path,
                 poses=poses,
                 betas=betas,
                 trans=transl,
                 gender=gender)
        return
    
    with open(object_in_scene_path, "r") as f:
        obj_in_scene = json.load(f)
    
    # Filter objects that are present (value is True)
    present_objects = [obj_name for obj_name, is_present in obj_in_scene.items() if is_present]
    
    if len(present_objects) == 0:
        if verbose:
            print("  Warning: No objects present in scene")
        # Still save human data
        human_output_path = os.path.join(output_dir, "human.npz")
        np.savez(human_output_path,
                 poses=poses,
                 betas=betas,
                 trans=transl,
                 gender=gender)
        return
    
    # Load object transformations
    object_transform_path = os.path.join(seq_root, "object_transformations.pkl")
    if not os.path.exists(object_transform_path):
        raise FileNotFoundError(f"object_transformations.pkl not found at {object_transform_path}")
    
    with open(object_transform_path, "rb") as f:
        object_transform = pickle.load(f)
    
    # Get all frame indices
    frame_indices = sorted(object_transform.keys())
    
    # Process each object and its parts (collect data first)
    obj_angles_dict = {}
    obj_trans_dict = {}
    obj_part_names = []
    
    for obj_name in present_objects:
        # Check for each possible part: base, part1, part2
        for part_name in ["base", "part1", "part2"]:
            obj_part_key = f"{obj_name}_{part_name}"
            
            # Check if this part exists in any frame
            part_exists = any(obj_part_key in object_transform[frame_idx] 
                            for frame_idx in frame_indices)
            
            if not part_exists:
                continue
            
            # Collect transformations across all frames
            obj_angles_list = []
            obj_trans_list = []
            
            for frame_idx in frame_indices:
                frame_transforms = object_transform[frame_idx]
                
                if obj_part_key in frame_transforms:
                    transform_matrix = frame_transforms[obj_part_key]
                    
                    # Extract rotation and translation
                    rot_vec, trans_vec = extract_rotation_translation_from_matrix(transform_matrix)
                    
                    obj_angles_list.append(rot_vec)
                    obj_trans_list.append(trans_vec)
                else:
                    # If object part not present in this frame, use identity transformation
                    obj_angles_list.append(np.zeros(3))
                    obj_trans_list.append(np.zeros(3))
            
            obj_angles = np.array(obj_angles_list)  # (T, 3)
            obj_trans = np.array(obj_trans_list)    # (T, 3)
            
            # Store in dictionaries
            obj_angles_dict[obj_part_key] = obj_angles
            obj_trans_dict[obj_part_key] = obj_trans
            obj_part_names.append(obj_part_key)
    
    # ==================== Apply Gravity Transformation ====================
    
    # Apply gravity transformation to align to interact coordinate system
    poses_transformed, trans_transformed, obj_angles_transformed, obj_trans_transformed = apply_gravity_transformation(
        poses, betas, transl, gender, obj_angles_dict, obj_trans_dict
    )
    
    # ==================== Save Transformed Data ====================
    
    # Save human.npz
    human_output_path = os.path.join(output_dir, "human.npz")
    np.savez(human_output_path,
             poses=poses_transformed,
             betas=betas,
             trans=trans_transformed,
             gender=gender)
    
    # Save all object parts
    object_files_created = []
    for obj_part_key in obj_part_names:
        # Parse object name and part
        parts = obj_part_key.split('_')
        part_name = parts[-1]
        obj_name = '_'.join(parts[:-1])
        
        # Save object part npz
        object_output_filename = f"object_{obj_name}_{part_name}.npz"
        object_output_path = os.path.join(output_dir, object_output_filename)
        
        np.savez(object_output_path,
                 angles=obj_angles_transformed[obj_part_key],
                 trans=obj_trans_transformed[obj_part_key],
                 name=obj_part_key)
        
        object_files_created.append(object_output_filename)
    
    if verbose:
        print(f"  Saved: human.npz + {len(object_files_created)} object files")


# ==================== Batch Processing Functions ====================

def find_all_sequences(smplx_seq_dir):
    """Find all sequence directories in data/parahome/raw/smplx_seq."""
    smplx_seq_dir = Path(smplx_seq_dir)
    
    if not smplx_seq_dir.exists():
        print(f"Error: Directory not found: {smplx_seq_dir}")
        return []
    
    # Find all directories starting with 's' followed by digits
    sequences = sorted([d for d in smplx_seq_dir.iterdir() 
                       if d.is_dir() and d.name.startswith('s')])
    
    return sequences


def process_single_sequence(seq_dir, output_root, verbose=False):
    """
    Wrapper function to process a single sequence (for multiprocessing).
    
    Args:
        seq_dir: Path to sequence directory
        output_root: Root output directory
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (seq_name, success, error_message)
    """
    seq_name = seq_dir.name
    output_dir = output_root / seq_name
    
    try:
        convert_parahome_to_interact(seq_dir, output_dir, verbose=verbose)
        return (seq_name, True, None)
    except Exception as e:
        return (seq_name, False, str(e))


def copy_scan_directories(scan_source_dir, objects_target_dir):
    """
    Copy all .obj files from data/parahome/raw/scan/{object_name}/simplified/ 
    to data/parahome/objects/{object_name}/ (preserving filenames).
    
    Args:
        scan_source_dir: Source directory (data/parahome/raw/scan)
        objects_target_dir: Target directory (data/parahome/objects)
    """
    scan_source_dir = Path(scan_source_dir)
    objects_target_dir = Path(objects_target_dir)
    
    if not scan_source_dir.exists():
        print(f"\nWarning: Scan source directory not found: {scan_source_dir}")
        print("Skipping scan directory copy")
        return
    
    
    # Create target directory
    objects_target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subdirectories in scan
    scan_dirs = [d for d in scan_source_dir.iterdir() if d.is_dir()]
    
    if not scan_dirs:
        print("No scan directories found to copy")
        return
    
    
    total_copied = 0
    total_skipped = 0
    not_found = 0
    
    for scan_dir in scan_dirs:
        object_name = scan_dir.name
        source_simplified_dir = scan_dir / "simplified"
        target_dir = objects_target_dir / object_name
        
        if not source_simplified_dir.exists():
            print(f"  ✗ Simplified directory not found for {object_name}")
            not_found += 1
            continue
        
        # Find all .obj files in the simplified directory
        obj_files = list(source_simplified_dir.glob("*.obj"))
        
        if not obj_files:
            print(f"  ✗ No .obj files found in {source_simplified_dir}")
            not_found += 1
            continue
        
        # Create target directory for this object
        target_dir.mkdir(parents=True, exist_ok=True)
        
        copied = 0
        skipped = 0
        
        for obj_file in obj_files:
            target_obj_path = target_dir / obj_file.name
            
            if target_obj_path.exists():
                skipped += 1
            else:
                try:
                    shutil.copy2(obj_file, target_obj_path)
                    copied += 1
                except Exception as e:
                    print(f"  ✗ Failed to copy {obj_file.name} from {object_name}: {e}")
        
        if copied > 0 or skipped > 0:
            print(f"  {object_name}: {copied} copied, {skipped} skipped")
        
        total_copied += copied
        total_skipped += skipped
    if args.verbose:
        print(f"\nScan copy complete: {total_copied} files copied, {total_skipped} skipped, {not_found} objects not found")


def main():
    parser = argparse.ArgumentParser(
        description="Process ParaHome dataset: convert sequences and copy scan directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script processes the ParaHome dataset by:
1. Converting all sequences from data/parahome/raw/smplx_seq to data/parahome/sequences
2. Copying scan directories from data/parahome/raw/scan to data/parahome/objects

The conversion includes gravity transformation to align with interact's coordinate system (+Y up).

Examples:
  # Process sequentially (single thread)
  python process_parahome.py
  
  # Process with specific number of workers
  python process_parahome.py --num_workers 4

  
  # Process with custom data root and verbose output
  python process_parahome.py --data_root /path/to/data --verbose --num_workers 8
        """
    )
    parser.add_argument(
        "--data_root",
        default="../data/parahome",
        help="Root directory for ParaHome data (default: data/parahome)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion information for each sequence"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    
    args = parser.parse_args()
    
    # Determine number of workers
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 1
    
    # Setup paths
    script_dir = Path(__file__).parent
    data_root = script_dir / args.data_root
    
    smplx_seq_dir = data_root / "raw" / "smplx_seq"
    output_root = data_root / "sequences"
    scan_source_dir = data_root / "raw" / "scan"
    objects_target_dir = data_root / "objects"
    print("="*60)
    print("ParaHome Dataset Processing")
    print("="*60)
    print(f"SMPL-X sequences: {smplx_seq_dir}")
    print(f"Output sequences: {output_root}")
    print(f"Scan source: {scan_source_dir}")
    print(f"Objects target: {objects_target_dir}")
    print()
    
    # ==================== Part 1: Convert Sequences ====================
    
    # Find all sequences
    sequences = find_all_sequences(smplx_seq_dir)
    
    if not sequences:
        print("No sequences found!")
    else:
        total = len(sequences)
        print(f"Found {total} sequences to convert\n")
        
        # Create output directory
        output_root.mkdir(parents=True, exist_ok=True)
        
        # Process sequences
        success = 0
        failed = 0
        failed_sequences = []
        
        if num_workers == 1:
            # Sequential processing
            for idx, seq_dir in enumerate(sequences, 1):
                seq_name = seq_dir.name
                if args.verbose:
                    print(f"[{idx}/{total}] Processing {seq_name}...")
                
                seq_name, is_success, error_msg = process_single_sequence(
                    seq_dir, output_root, verbose=args.verbose
                )
                
                if is_success:
                    success += 1
                else:
                    failed += 1
                    print(f"  ✗ Failed: {error_msg}")
                    failed_sequences.append((seq_name, error_msg))
                
                # Print progress every 10 sequences
                if idx % 10 == 0:
                    print(f"\n--- Progress: {idx}/{total} ({100*idx//total}%) ---")
                    print(f"    Success: {success}, Failed: {failed}\n")
        else:
            # Parallel processing
            print(f"Processing in parallel with {num_workers} workers...\n")
            
            # Create partial function with fixed arguments
            process_func = partial(process_single_sequence, 
                                 output_root=output_root, 
                                 verbose=args.verbose)
            
            # Process sequences in parallel
            with Pool(processes=num_workers) as pool:
                # Use imap for better progress tracking
                results = pool.imap(process_func, sequences)
                
                for idx, (seq_name, is_success, error_msg) in enumerate(results, 1):
                    if is_success:
                        success += 1
                        print(f"[{idx}/{total}] ✓ {seq_name}")
                    else:
                        failed += 1
                        print(f"[{idx}/{total}] ✗ {seq_name}: {error_msg}")
                        failed_sequences.append((seq_name, error_msg))
                    
                    # Print progress every 10 sequences
                    if idx % 10 == 0:
                        print(f"\n--- Progress: {idx}/{total} ({100*idx//total}%) ---")
                        print(f"    Success: {success}, Failed: {failed}\n")
        
        # Summary
        if args.verbose:
            print(f"\n{'='*60}")
            print("Sequence Conversion Complete!")
            print(f"Total: {total}, Success: {success}, Failed: {failed}")
            print(f"{'='*60}")
        
    
    # ==================== Part 2: Copy Scan Directories ====================
    
    copy_scan_directories(scan_source_dir, objects_target_dir)



if __name__ == "__main__":
    main()

