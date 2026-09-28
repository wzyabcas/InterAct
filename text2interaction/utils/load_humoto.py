import pickle
import os
import numpy as np
import copy
from .rotation_helper import quaternion_to_matrix, matrix_to_quaternion
import yaml
import trimesh
from typing import Dict, List, Optional, Any
import torch

try:
    HUMOTO_OBJECT_DIR = os.environ.get('HUMOTO_OBJECT_DIR')
    HUMOTO_DATASET_DIR = os.environ.get('HUMOTO_DATASET_DIR')
except:
    HUMOTO_OBJECT_DIR = None
    HUMOTO_DATASET_DIR = None

Z_UP_TO_Y_UP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

def load_one_humoto_sequence(
    data_path: str, 
    include_text: bool = True, 
    y_up: bool = True,
    object_model: bool = True, 
    object_model_path: Optional[str] = HUMOTO_OBJECT_DIR, 
    object_modality: List[str] = ['mesh', 'pc'],
    pose_params: bool = True,
    bone_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Load one HUMOTO sequence from the specified data path.
    
    This function loads a complete HUMOTO sequence including human motion data,
    object poses, optional 3D models, and metadata. It handles coordinate system
    conversions and provides various data modalities.
    
    Args:
        data_path: Path to the sequence directory containing .pkl and .yaml files.
        include_text: Whether to load text metadata from .yaml file.
        y_up: If True, converts from Z-up to Y-up coordinate system.
        object_model: Whether to load 3D object models.
        object_model_path: Path to object model directory (uses HUMOTO_OBJECT_DIR env var if None).
        object_modality: List of object modalities to load ('mesh', 'pc').
        pose_params: Whether to extract pose parameters from armature data.
        bone_names: List of specific bone names to extract (None for all bones).
    
    Returns:
        Dictionary containing the loaded sequence data:
        - armature: List of dictionaries with bone transforms per frame
        - objects: Dictionary mapping object names to pose sequences
        - text: Text metadata (if include_text=True)
        - object_models: 3D object models (if object_model=True)
        - armature_pose_params: Extracted pose parameters (if pose_params=True)
        - object_pose_params: Object pose parameters (if pose_params=True)
    
    Raises:
        FileNotFoundError: If required files are missing.
        ValueError: If data format is invalid.
        KeyError: If required data keys are missing.
    """
    # Input validation
    if not os.path.exists(data_path):
        if HUMOTO_DATASET_DIR is not None:
            data_path = os.path.join(HUMOTO_DATASET_DIR, data_path)
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data path does not exist: {data_path}")
        else:
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path is not a directory: {data_path}")
    
    file_name = data_path.split('/')[-1]
    pk_file_path = os.path.join(data_path, f'{file_name}.pkl')
    
    if not os.path.exists(pk_file_path):
        raise FileNotFoundError(f"Pickle file not found: {pk_file_path}")
    
    # Load the main data
    with open(pk_file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Validate required keys
    required_keys = ['armature', 'objects']
    for key in required_keys:
        if key not in data:
            raise KeyError(f"Required key '{key}' not found in data")
    
    object_names = list(data['objects'].keys())
     
    if y_up:
        for obj_name in object_names:
            transform = np.array(data['objects'][obj_name], dtype=np.float32)
            transform_matrix = quaternion_to_matrix(transform)
            transform_matrix_y_up = Z_UP_TO_Y_UP @ transform_matrix
            transform_quat = matrix_to_quaternion(transform_matrix_y_up)
            data['objects'][obj_name] = transform_quat
    if object_model:
        if 'object_models' in data:
            if 'pc' in object_modality:
                for obj_name in data['object_models']:
                    v, f = data['object_models'][obj_name]['mesh']
                    obj_mesh = trimesh.Trimesh(v, f)
                    pcs, _ = trimesh.sample.sample_surface(obj_mesh, 10000)
                    pcs = np.array(pcs, dtype=np.float32)
                    data['object_models'][obj_name]['pc'] = pcs
        else:
            if object_model_path is None:
                raise ValueError("object_model_path must be provided when object_model=True")
            
            if not os.path.exists(object_model_path):
                raise FileNotFoundError(f"Object model path does not exist: {object_model_path}")
            
            data['object_models'] = {}
            for obj_name in object_names:
                obj_name_used = obj_name
                if '.' in obj_name:
                    obj_name_used = obj_name.split('.')[0]
                obj_path = os.path.join(object_model_path, obj_name_used, f'{obj_name_used}.obj')
                object_model = load_object_model(obj_path, object_modality)
                data['object_models'][obj_name] = object_model
    if include_text:
        text_file = os.path.join(data_path, f'{file_name}.yaml')
        if not os.path.exists(text_file):
            print(f"Warning: Text file not found: {text_file}")
            data['text'] = None
        else:
            try:
                with open(text_file, 'r') as f:
                    data['text'] = yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Failed to load text file: {e}")
                data['text'] = None
    if pose_params:
        data['armature_pose_params'] = get_pose_params(data['armature'], bone_names)
        data['object_pose_params'] = get_object_poses(data['objects'])
    return data,object_names
    
def load_object_model(obj_path: str, object_modality: List[str] = ['mesh', 'pc']) -> Dict[str, Any]:
    """
    Load the object model from the specified path.

    Args:
        obj_path: Path to the object model.
        object_modality: List of object modalities to load ('mesh', 'pc').

    Returns:
        Dictionary containing the object model data:
        - mesh: Mesh data (vertices, faces)
        - pc: Point cloud data (N, 3)
    """
    object_model = {}
    if not os.path.exists(obj_path):
        print(f"Warning: Object model not found: {obj_path}")
    try:
        obj_mesh = trimesh.load(obj_path)
        if isinstance(obj_mesh, trimesh.Scene):
            obj_mesh = trimesh.util.concatenate(obj_mesh.dump())
        verts = np.array(obj_mesh.vertices, dtype=np.float32)
        faces = np.array(obj_mesh.faces, dtype=np.int64)
        object_model['mesh'] = (verts, faces)
        
        if 'pc' in object_modality:
            pcs, _ = trimesh.sample.sample_surface(obj_mesh, 10000)
            pcs = np.array(pcs, dtype=np.float32)
            object_model['pc'] = pcs
    except Exception as e:
        print(f"Warning: Failed to load object model {obj_path}: {e}")
    return object_model


def list_all_sequences(data_path: str) -> List[str]:
    """
    List all sequence directories in the specified data path.
    
    This function scans the data directory and returns all subdirectories
    that contain HUMOTO sequences.
    
    Args:
        data_path: Path to the directory containing sequence folders.
    
    Returns:
        List of full paths to sequence directories.
    
    Raises:
        FileNotFoundError: If data_path does not exist.
        ValueError: If data_path is not a directory.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    if not os.path.isdir(data_path):
        raise ValueError(f"Data path is not a directory: {data_path}")
    
    sequences = os.listdir(data_path)
    sequences = [os.path.join(data_path, f) for f in sequences if os.path.isdir(os.path.join(data_path, f))]
    return sequences


def get_pose_params(armature_data: List[Dict[str, List[float]]], bone_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Extract pose parameters from armature data.
    
    This function extracts pose parameters for specified bones from the armature
    data, converting them into a dictionary of NumPy arrays.
    
    Args:
        armature_data: List of dictionaries containing bone transforms per frame.
                      Each dictionary maps bone names to [quat, loc] lists.
        bone_names: List of bone names to extract (None for all bones).
    
    Returns:
        Dictionary mapping bone names to pose parameter arrays of shape [T, 7].
    
    Raises:
        ValueError: If armature_data is empty or bone_names contains invalid names.
        KeyError: If specified bone names are not found in the data.
    """
    if not armature_data:
        raise ValueError("armature_data cannot be empty")
    
    pose_params = {}
    if bone_names is None:
        bone_names = list(armature_data[0].keys())
    
    for bone_name in bone_names:
        if bone_name not in armature_data[0]:
            raise KeyError(f"Bone '{bone_name}' not found in armature data")
        
        transform_data = [armature_data[i][bone_name] for i in range(len(armature_data))]
        transform_data = np.array(transform_data, dtype=np.float32)  # [B, 7]
        pose_params[bone_name] = transform_data
    return pose_params

def get_object_poses(object_data: Dict[str, List[List[float]]]) -> Dict[str, np.ndarray]:
    """
    Extract object pose parameters from object data.
    
    This function extracts pose parameters for all objects from the object
    data, converting them into a dictionary of NumPy arrays.
    
    Args:
        object_data: Dictionary mapping object names to pose sequences.
                    Each pose sequence is a list of [quat, loc] lists.
    
    Returns:
        Dictionary mapping object names to pose parameter arrays of shape [T, 7].
    
    Raises:
        ValueError: If object_data is empty or contains invalid data.
    """
    if not object_data:
        raise ValueError("object_data cannot be empty")
    
    object_pose = {}
    for obj in object_data:
        # print(object_data[obj].shape)
        # if not object_data[obj]:
        #     print(f"Warning: Empty pose data for object '{obj}'")
        #     continue
        
        transform_data = np.array(object_data[obj], dtype=np.float32)  # [B, 7]
        object_pose[obj] = transform_data
    return object_pose

def recenter_object(obj_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Recenter the object mesh. Object mesh in the HUMOTO dataset is centered by the center of its bounding box.
    Args:
        obj_mesh: trimesh.Trimesh
    Returns:
        obj_mesh: trimesh.Trimesh
    """
    obj_bbox = obj_mesh.bounds
    obj_bbox_center = (obj_bbox[0] + obj_bbox[1]) / 2
    obj_mesh.vertices -= obj_bbox_center
    return obj_mesh

def get_transformed_object(object_model: Dict[str, Any], object_pose_params: torch.Tensor) -> Dict[str, Any]:
    object_transformed = {}
    object_pose_rot_matrix = quaternion_to_matrix(object_pose_params[..., :4])
    if 'mesh' in object_model:
        # print(fuck)
        v, f = object_model['mesh']
        if len(v.shape) == 2:
            v = v.unsqueeze(0)
        if v.shape[0] != object_pose_params.shape[0]:
            v = v.repeat(object_pose_params.shape[0], 1, 1)
        v = torch.matmul(v, object_pose_rot_matrix.transpose(1, 2)) + object_pose_params[..., 4:][:, None]
        object_transformed['mesh'] = (v, f)
    if 'pc' in object_model:
        pcs = object_model['pc']
        if len(pcs.shape) == 2:
            pcs = pcs.unsqueeze(0)
        if pcs.shape[0] != object_pose_params.shape[0]:
            pcs = pcs.repeat(object_pose_params.shape[0], 1, 1)
        pcs = torch.matmul(pcs, object_pose_rot_matrix.transpose(1, 2)) + object_pose_params[..., 4:][:, None]
        object_transformed['pc'] = pcs
    return object_transformed
