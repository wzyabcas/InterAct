#!/usr/bin/env python3
"""
Visualizer for ParaHome sequences with proper articulated object handling.

This script visualizes ParaHome sequences by properly handling articulated objects
using the stored 'arti' (revolute joint angles) and 'axis' (rotation axes) parameters.
Unlike the basic visualizer, this properly applies articulation relative to base parts.
"""

import os
import sys
import glob
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R
import argparse

# Add parent directory to path to import render module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from text2interaction.render.mesh_viz import visualize_body_obj
import smplx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

MODEL_PATH = 'models'

def build_smplx_model(gender, num_betas=20):
    """
    Build SMPLX model matching ParaHome's configuration.
    
    Args:
        gender: 'male', 'female', or 'neutral'
        num_betas: Number of beta parameters (default 20 as in ParaHome)
    
    Returns:
        SMPLX body model
    """
    smplx_model = smplx.create(
        model_path=MODEL_PATH,
        model_type="smplx",
        flat_hand_mean=True,
        use_pca=False,
        num_betas=num_betas,
        num_expression_coeffs=10,
        gender=gender,
        ext='pkl'
    ).to(device)
    
    return smplx_model


def process_smplx_frame(smpl_model, pose, betas, trans):
    """
    Process a single frame with SMPLX model.
    
    Args:
        smpl_model: SMPLX model instance
        pose: (156,) array of pose parameters for one frame
        betas: (num_betas,) array of shape parameters
        trans: (3,) array of global translation for one frame
    
    Returns:
        verts: (V, 3) human vertices
        joints: (J, 3) joint positions
    """
    # Extract pose components
    # Format: [global_orient (3), body_pose (63), hand_pose (90)] = 156
    global_orient = pose[:3]        # (3,)
    body_pose = pose[3:66]          # (63,)
    hand_pose = pose[66:156]        # (90,)
    
    # Split hand pose into left and right
    lhand_pose = hand_pose[:45]     # (45,) = 15 joints * 3
    rhand_pose = hand_pose[45:90]   # (45,) = 15 joints * 3
    
    # Reshape body pose for SMPLX: (1, 21, 3)
    body_pose_reshaped = body_pose.reshape((1, 21, 3))
    lhand_pose_reshaped = lhand_pose.reshape((1, 15, 3))
    rhand_pose_reshaped = rhand_pose.reshape((1, 15, 3))
    
    # Convert to tensors and move to device
    body_pose_torch = torch.from_numpy(body_pose_reshaped).float().to(device)
    global_orient_torch = torch.from_numpy(global_orient[None, :]).float().to(device)
    lhand_pose_torch = torch.from_numpy(lhand_pose_reshaped).float().to(device)
    rhand_pose_torch = torch.from_numpy(rhand_pose_reshaped).float().to(device)
    betas_torch = torch.from_numpy(betas[None, :]).float().to(device)
    transl_torch = torch.from_numpy(trans[None, :]).float().to(device)
    
    # Forward pass through SMPLX model
    with torch.no_grad():
        smplx_output = smpl_model(
            betas=betas_torch,
            return_verts=True,
            body_pose=body_pose_torch,
            left_hand_pose=lhand_pose_torch,
            right_hand_pose=rhand_pose_torch,
            global_orient=global_orient_torch,
            transl=transl_torch,
            jaw_pose=torch.zeros([1, 3]).float().to(device),
            leye_pose=torch.zeros([1, 3]).float().to(device),
            reye_pose=torch.zeros([1, 3]).float().to(device),
            expression=torch.zeros([1, 10]).float().to(device)
        )
    
    verts = to_cpu(smplx_output.vertices[0])  # Remove batch dimension
    joints = to_cpu(smplx_output.joints[0])
    
    return verts, joints


def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert axis-angle representation to rotation matrix using Rodrigues' formula.
    
    Args:
        axis: (3,) rotation axis vector (normalized)
        angle: rotation angle in radians (scalar)
    
    Returns:
        3x3 rotation matrix
    """
    axis = np.array(axis, dtype=np.float32)
    # Normalize axis
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-8:
        axis = axis / axis_norm
    else:
        axis = np.array([1, 0, 0], dtype=np.float32)  # Default axis
    
    # Use scipy's Rotation for robustness
    rot = R.from_rotvec(angle * axis)
    return rot.as_matrix()


def load_object_data_articulated(obj_part_npz, scan_root, start_frame=0, end_frame=None):
    """
    Load object mesh and transformation data with articulation support.
    
    Args:
        obj_part_npz: Path to object_{name}_{part}.npz file
        scan_root: Root directory containing object meshes
        start_frame: Start frame for slicing (default: 0)
        end_frame: End frame for slicing (default: None, meaning all frames)
    
    Returns:
        obj_verts_base: (V, 3) base object vertices
        object_faces: (F, 3) face indices
        obj_angles: (T, 3) rotation vectors (part's full transformation from object_transformations.pkl)
        obj_trans: (T, 3) translations (part's full transformation from object_transformations.pkl)
        obj_arti: (T,) revolute joint angles in radians (None for base parts)
        obj_axis: (3,) rotation axis vector in original object coordinates (None for base parts)
        obj_pivot: (3,) pivot point vector in original object coordinates (None for base parts)
        obj_part_name: Name of the object part
        is_articulated: Whether this part is articulated (has arti and axis)
    """
    # Load object transformation data
    with np.load(obj_part_npz, allow_pickle=True) as f:
        obj_angles = f['angles']  # (T, 3) rotation vectors (part's full transformation from object_transformations.pkl)
        obj_trans = f['trans']    # (T, 3) translations (part's full transformation from object_transformations.pkl)
        obj_part_name = str(f['name'])  # e.g., "refrigerator_part1"
        
        # Check if this is an articulated part
        is_articulated = 'arti' in f and 'axis' in f
        if is_articulated:
            obj_arti = f['arti']  # (T,) revolute joint angles
            
            # Use stored axis/pivot from npz file (now stored in original object coordinates)
            obj_axis = f['axis']  # (3,) rotation axis vector in original object coordinates
            obj_pivot = f.get('pivot', None)  # (3,) pivot point in original object coordinates
            
            # Handle scalar pivot if needed (convert to 3D)
            if obj_pivot is not None:
                pivot_arr = np.array(obj_pivot, dtype=np.float64)
                if pivot_arr.size == 1:
                    # Scalar pivot - distance along axis
                    axis_normalized = obj_axis / np.linalg.norm(obj_axis)
                    obj_pivot = (float(pivot_arr) * axis_normalized).astype(np.float32)
                else:
                    obj_pivot = pivot_arr.astype(np.float32)
        else:
            obj_arti = None
            obj_axis = None
            obj_pivot = None
    
    # Apply frame range selection
    if end_frame is None:
        end_frame = obj_angles.shape[0]
    obj_angles = obj_angles[start_frame:end_frame]
    obj_trans = obj_trans[start_frame:end_frame]
    if obj_arti is not None:
        obj_arti = obj_arti[start_frame:end_frame]
    
    print(f"\n  Loading object part: {obj_part_name}")
    print(f"    Angles shape: {obj_angles.shape}")
    print(f"    Translation shape: {obj_trans.shape}")
    if is_articulated:
        print(f"    Articulated: arti shape {obj_arti.shape}, axis {obj_axis}")
        if obj_pivot is not None:
            print(f"    Pivot: {obj_pivot}")
        else:
            print(f"    Pivot: None (using origin)")
    else:
        print(f"    Base part (no articulation)")
    
    # Parse object name and part from the stored name
    parts = obj_part_name.split('_')
    if len(parts) >= 2:
        part_name = parts[-1]  # 'base', 'part1', or 'part2'
        object_name = '_'.join(parts[:-1])  # Handle names with underscores
    else:
        raise ValueError(f"Invalid object part name format: {obj_part_name}")
    
    # Load object mesh
    obj_mesh_path = os.path.join(scan_root, object_name, f'{part_name}.obj')
    
    if not os.path.exists(obj_mesh_path):
        print(f"    Warning: Object mesh not found at {obj_mesh_path}")
        return None, None, None, None, None, None, obj_part_name, False
    
    OBJ_MESH = trimesh.load(obj_mesh_path)
    print(f"    Loaded mesh from: {obj_mesh_path}")
    print(f"    Mesh vertices: {len(OBJ_MESH.vertices)}, faces: {len(OBJ_MESH.faces)}")
    
    obj_verts_base = np.array(OBJ_MESH.vertices).astype(np.float32)
    object_faces = OBJ_MESH.faces.astype(np.int32)
    
    return obj_verts_base, object_faces, obj_angles, obj_trans, obj_arti, obj_axis, obj_pivot, obj_part_name, is_articulated


def transform_object_frame_articulated(obj_verts_base, obj_angle, obj_trans, 
                                       obj_arti=None, obj_axis=None, obj_pivot=None):
    """
    Transform object vertices for a single frame with articulation support.
    
    For base parts: Apply only global rotation and translation.
    For articulated parts: Apply base transformation first, then articulation rotation around pivot.
    
    The pivot point is in the object's coordinate system (same as base). The articulation rotation
    happens around the pivot point in world coordinates after the base transformation.
    
    Args:
        obj_verts_base: (V, 3) base object vertices (in part's local coordinate)
        obj_angle: (3,) rotation vector for base transformation
        obj_trans: (3,) translation for base transformation
        obj_arti: scalar revolute joint angle in radians (None for base parts)
        obj_axis: (3,) rotation axis vector in original object coordinates (None for base parts)
        obj_pivot: (3,) pivot point in original object coordinates (None for base parts)
    
    Returns:
        object_verts: (V, 3) transformed object vertices
    """
    # Convert base rotation vector to matrix
    base_rot_matrix = R.from_rotvec(obj_angle).as_matrix()  # (3, 3)
    
    if obj_arti is not None and obj_axis is not None:
        # This is an articulated part
        # IMPORTANT: When we compute the transformation from axis-angle with pivot in process_parahome.py,
        # we compute: transform_matrix = [[R, (I-R)@pivot], [0, 1]]
        # Then we extract: rotation = R, translation = (I-R)@pivot
        #
        # The extracted translation already includes the pivot effect!
        # So vertices_new = R @ vertices + t is equivalent to rotating around pivot.
        #
        # However, we store base transformation separately, so we need to:
        # 1. Apply articulation transformation: vertices_arti = R_arti @ vertices + t_arti
        #    where t_arti = (I - R_arti) @ pivot_obj (in object coordinates)
        # 2. Then apply base transformation: vertices_final = R_base @ vertices_arti + t_base
        
        # IMPORTANT: obj_axis and obj_pivot are in ORIGINAL object coordinates (stored correctly in npz files)
        # They are NOT transformed by gravity, so they can be used directly for articulation
        
        # Step 1: Use axis directly (in original object coordinates)
        axis_obj = obj_axis.astype(np.float32)
        
        # Step 2: Compute articulation rotation matrix in object coordinates
        arti_rot_matrix = axis_angle_to_rotation_matrix(axis_obj, obj_arti)  # (3, 3)
        
        # Step 3: Get pivot in object coordinates and compute articulation translation
        if obj_pivot is not None:
            # Use pivot directly (in original object coordinates)
            pivot_obj = np.array(obj_pivot, dtype=np.float32)
            # The effective translation from rotating around pivot: (I - R) @ pivot
            arti_trans_obj = (np.eye(3, dtype=np.float32) - arti_rot_matrix) @ pivot_obj
        else:
            # No pivot, so no translation from pivot rotation
            arti_trans_obj = np.zeros(3, dtype=np.float32)
        
        # Step 4: Apply articulation transformation in object coordinates
        # vertices_arti = R_arti @ vertices + t_arti
        # This is equivalent to rotating around pivot
        vertices_after_arti = obj_verts_base @ arti_rot_matrix.T + arti_trans_obj
        
        # Step 5: Apply base transformation to move articulated part to world coordinates
        object_verts = vertices_after_arti @ base_rot_matrix.T + obj_trans
    else:
        # This is a base part: just apply base rotation and translation
        object_verts = obj_verts_base @ base_rot_matrix.T + obj_trans
    
    return object_verts


def discover_object_files(sequence_path):
    """
    Discover all object_{name}_{part}.npz files in the sequence directory.
    
    Args:
        sequence_path: Path to the sequence directory
    
    Returns:
        List of object npz file paths
    """
    pattern = os.path.join(sequence_path, 'object_*.npz')
    object_files = sorted(glob.glob(pattern))
    return object_files


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ParaHome sequences with articulated object support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize a converted sequence with articulated objects
  python visualizer_articulated.py \\
      --sequence_path ./data/parahome/sequences_canonical/s1 \\
      --scan_root ./data/parahome/objects \\
      --output_path ./output/s1
  
  # Visualize full sequence with multi-angle rendering
  python visualizer_articulated.py \\
      --sequence_path ./data/parahome/sequences_canonical/s1 \\
      --scan_root ./data/parahome/objects \\
      --output_path ./output/s1_frames100-300.mp4 \\
      --multi_angle \\
      --show_frame
  
  Note: This visualizer properly handles articulated objects using 'arti' and 'axis' parameters.
        It renders the full sequence as a video.
        """
    )
    parser.add_argument(
        "--sequence_path",
        required=True,
        help="Path to the converted sequence directory containing human.npz and object_*.npz"
    )
    parser.add_argument(
        "--scan_root",
        required=True,
        help="Path to ParaHome objects directory containing object meshes"
    )
    parser.add_argument(
        "--output_path",
        default="./visualization_output",
        help="Output directory for rendered images (default: ./visualization_output/{sequence_name}/)"
    )
    parser.add_argument(
        "--multi_angle",
        action="store_true",
        help="Render from multiple angles"
    )
    parser.add_argument(
        "--show_frame",
        action="store_true",
        help="Show frame numbers in the video"
    )
    args = parser.parse_args()
    
    # Get sequence name
    sequence_name = os.path.basename(args.sequence_path.rstrip('/'))
    
    print("="*60)
    print(f"ParaHome Articulated Visualizer")
    print("="*60)
    print(f"Sequence: {sequence_name}")
    print(f"Sequence path: {args.sequence_path}")
    print(f"Scan root: {args.scan_root}")
    
    # ==================== Load Human Data ====================
    print("\n=== Loading Human Data ===")
    human_npz_path = os.path.join(args.sequence_path, "human.npz")
    
    if not os.path.exists(human_npz_path):
        raise FileNotFoundError(f"human.npz not found at {human_npz_path}")
    
    with np.load(human_npz_path, allow_pickle=True) as f:
        poses = f['poses']    # (T, 156)
        betas = f['betas']    # (num_betas,)
        trans = f['trans']    # (T, 3)
        gender = str(f['gender'])
    
    print(f"Gender: {gender}")
    print(f"Poses shape: {poses.shape}")
    print(f"Betas shape: {betas.shape}")
    print(f"Translation shape: {trans.shape}")
    
    num_frames = poses.shape[0]
    print(f"Total frames: {num_frames}")
    
    start_frame = 0
    end_frame = num_frames
    print(f"Visualizing full sequence: {num_frames} frames")
    
    # Build SMPLX model once
    print("Building SMPLX model...")
    smpl_model = build_smplx_model(gender, num_betas=len(betas))
    human_faces = smpl_model.faces.astype(np.int32)
    
    # ==================== Load Object Data ====================
    print("\n=== Loading Object Data ===")
    
    object_files = discover_object_files(args.sequence_path)
    
    print(f"Found {len(object_files)} object part files")
    
    # Load all object data (meshes and transformations)
    object_data_list = []  # List of dicts with object data
    
    if len(object_files) == 0:
        print("Warning: No object files found. Rendering human only.")
    else:
        for obj_file in object_files:
            obj_verts_base, obj_faces, obj_angles, obj_trans, obj_arti, obj_axis, obj_pivot, obj_name, is_articulated = load_object_data_articulated(
                obj_file, args.scan_root, start_frame, end_frame
            )
            
            if obj_verts_base is not None and obj_faces is not None:
                object_data_list.append({
                    'verts_base': obj_verts_base,
                    'faces': obj_faces,
                    'angles': obj_angles,
                    'trans': obj_trans,
                    'arti': obj_arti,  # None for base parts
                    'axis': obj_axis,  # None for base parts
                    'pivot': obj_pivot,  # None for base parts
                    'name': obj_name,
                    'is_articulated': is_articulated
                })
        
        print(f"\nSuccessfully loaded {len(object_data_list)} object parts")
        articulated_count = sum(1 for obj in object_data_list if obj['is_articulated'])
        base_count = len(object_data_list) - articulated_count
        print(f"  - Base parts: {base_count}")
        print(f"  - Articulated parts: {articulated_count}")
    
    # ==================== Render ====================
    print("\n=== Rendering ===")

    # Determine output video path
    if args.output_path is None:
        output_path = os.path.join("./output", f"{sequence_name}.mp4")
    else:
        if os.path.splitext(args.output_path)[1]:
            output_path = args.output_path
        else:
            output_path = os.path.join(args.output_path, f"{sequence_name}.mp4")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Output video: {output_path}")
    print(f"Multi-angle: {args.multi_angle}")
    print(f"Show frame numbers: {args.show_frame}")

    # Get frame indices to process (full sequence)
    num_frames_to_render = end_frame - start_frame
    frames_to_process = list(range(num_frames_to_render))
    print(f"\nPreparing {len(frames_to_process)} frames...")

    human_verts_seq = []
    object_verts_seq = []
    object_faces_merged = None

    for idx, frame_idx in enumerate(frames_to_process):
        if idx % 50 == 0:
            print(f"  Preparing frame {idx+1}/{len(frames_to_process)} (frame {start_frame + frame_idx})...")

        human_verts_frame, _ = process_smplx_frame(
            smpl_model, poses[frame_idx], betas, trans[frame_idx]
        )
        human_verts_seq.append(human_verts_frame.astype(np.float32))

        base_transforms = {}
        for obj_data in object_data_list:
            if not obj_data['is_articulated']:
                parts = obj_data['name'].split('_')
                if len(parts) >= 2:
                    obj_name = '_'.join(parts[:-1])
                    base_transforms[obj_name] = (
                        obj_data['angles'][frame_idx],
                        obj_data['trans'][frame_idx]
                    )

        obj_verts_list_frame = []
        obj_faces_list_frame = []
        for obj_data in object_data_list:
            if obj_data['is_articulated']:
                parts = obj_data['name'].split('_')
                if len(parts) >= 2:
                    obj_name = '_'.join(parts[:-1])
                    if obj_name in base_transforms:
                        base_angles, base_trans = base_transforms[obj_name]
                        obj_verts_transformed = transform_object_frame_articulated(
                            obj_data['verts_base'],
                            base_angles,
                            base_trans,
                            obj_data['arti'][frame_idx],
                            obj_data['axis'],
                            obj_data['pivot']
                        )
                    else:
                        obj_verts_transformed = transform_object_frame_articulated(
                            obj_data['verts_base'],
                            obj_data['angles'][frame_idx],
                            obj_data['trans'][frame_idx],
                            obj_data['arti'][frame_idx],
                            obj_data['axis'],
                            obj_data['pivot']
                        )
                else:
                    obj_verts_transformed = transform_object_frame_articulated(
                        obj_data['verts_base'],
                        obj_data['angles'][frame_idx],
                        obj_data['trans'][frame_idx],
                        obj_data['arti'][frame_idx] if obj_data['arti'] is not None else None,
                        obj_data['axis'],
                        obj_data['pivot']
                    )
            else:
                obj_verts_transformed = transform_object_frame_articulated(
                    obj_data['verts_base'],
                    obj_data['angles'][frame_idx],
                    obj_data['trans'][frame_idx],
                    None, None, None
                )

            obj_verts_list_frame.append(obj_verts_transformed.astype(np.float32))
            obj_faces_list_frame.append(obj_data['faces'])

        if len(obj_verts_list_frame) == 0:
            merged_obj_verts = np.zeros((0, 3), dtype=np.float32)
            merged_obj_faces = np.zeros((0, 3), dtype=np.int32)
        else:
            merged_obj_verts_parts = []
            merged_obj_faces_parts = []
            vert_offset = 0
            for verts_part, faces_part in zip(obj_verts_list_frame, obj_faces_list_frame):
                merged_obj_verts_parts.append(verts_part)
                merged_obj_faces_parts.append(faces_part + vert_offset)
                vert_offset += verts_part.shape[0]
            merged_obj_verts = np.concatenate(merged_obj_verts_parts, axis=0)
            merged_obj_faces = np.concatenate(merged_obj_faces_parts, axis=0).astype(np.int32)

        object_verts_seq.append(merged_obj_verts)
        if object_faces_merged is None:
            object_faces_merged = merged_obj_faces

    human_verts_seq = np.stack(human_verts_seq, axis=0)
    object_verts_seq = np.stack(object_verts_seq, axis=0) if len(object_verts_seq) > 0 else np.zeros((0, 0, 3), dtype=np.float32)
    if object_faces_merged is None:
        object_faces_merged = np.zeros((0, 3), dtype=np.int32)

    print(f"\nRendering video to: {output_path}")
    visualize_body_obj(
        human_verts_seq,
        human_faces,
        object_verts_seq,
        object_faces_merged,
        save_path=output_path,
        multi_angle=args.multi_angle,
        show_frame=args.show_frame
    )

    print("\n" + "="*60)
    print("Visualization complete!")
    print(f"Rendered {len(frames_to_process)} frames")
    print(f"Frame range: {start_frame} to {end_frame-1}")
    print(f"Video saved to: {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()

