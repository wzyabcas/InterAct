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
