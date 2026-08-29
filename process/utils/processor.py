import numpy as np
import trimesh
import smplx
import torch
import logging

from scipy.spatial.transform import Rotation
from typing import Dict, Tuple, List, Optional
from pathlib import Path

from .types import ObjectSequence, HumanSequence, Sequence

logger=logging.getLogger(__name__)

class Processor():
    def __init__(self, dict_path: Dict[str, str], smpl_init: bool = False):
        # put all variables in args into self
        for k, v in dict_path.items():
            setattr(self, k.lower(), v)
        if smpl_init:
            smplh10:Dict[str, smplx.SMPLH] = self._create_smplh_model(num_betas=10)
            self.dict_smpl_model = smplh10

    def _create_smplh_model(self, num_betas: int = 10) -> Dict[str, smplx.SMPLH]:
        """Copied from visualization/visualize.py"""
        smplh_model_male = smplx.create(self.model_path, model_type='smplh',
                                gender="male",
                                use_pca=False,
                                num_betas=num_betas,
                                flat_hand_mean=True,
                                ext='pkl')

        smplh_model_female = smplx.create(self.model_path, model_type='smplh',
                                gender="female",
                                use_pca=False,
                                num_betas=num_betas,
                                flat_hand_mean=True,
                                ext='pkl')

        smplh10 = {'male': smplh_model_male, 'female': smplh_model_female}
        return smplh10

    def _print(self):
        for k, v in vars(self).items():
            print(f"{k}: {v}")
            
    def _load_human_sequence(self, path_folder: str, sequence_name: str, double_dir:bool = True) -> HumanSequence:
        """Load and validate one HUMOTO SMPL-H parameter sequence.

        ------------------------------------------------
        Args:
            sequence_name (str): The name of the sequence, e.g., "drinking_from_mug1_and_talking-277".
            path_folder (str): The folder before the emergence of sequence_name
            double_dir (bool): If True, the path to the .npz file is constructed as path_folder/{sequence_name}/human.npz. \
                If False, the path is constructed as path_folder/sequence_name.npz. For example .../smplh/{sequence_name}.npz
        """
        if double_dir:
            human_path = Path(path_folder) / sequence_name / "human.npz"
        else:
            human_path = Path(path_folder) / f"{sequence_name}.npz"
        # print(human_path); import os; print(os.path.isfile(human_path)); return
        logger.debug(f"Loading human sequence from: {human_path}")
        if not human_path.is_file():
            raise FileNotFoundError(f"Missing HUMOTO human sequence file: {human_path}")

        with np.load(human_path, allow_pickle=True) as data:
            expected = {"poses", "betas", "trans", "gender"}
            if set(data.files) != expected:
                raise ValueError(f"Unexpected human fields in {human_path}: {data.files}")
            poses = np.asarray(data["poses"], dtype=np.float32)
            betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
            trans = np.asarray(data["trans"], dtype=np.float32)
            gender_raw = data["gender"]
            gender = str(gender_raw.item() if gender_raw.shape == () else gender_raw.reshape(-1)[0])

        human_sequence = HumanSequence(
            poses=poses,
            betas=betas,
            trans=trans,
            gender=gender
        )
        return human_sequence

    def _load_objects_sequence(self, pose_path_folder: str, mesh_path_folder: str, sequence_name: str) \
        -> List[ObjectSequence]:
        """Load every object instance from one HUMOTO sequence.

        Notes
        -----
        `objs_pose_path`: The path to the .npz file containing the poses of all object instances in the sequence.

        `object_path_raw`: The path to the directory containing all 735 object meshes.

        Each object instance in the sequence has a unique `instance_name`, like "mug.001", "mug.002", etc.  
        The `mesh_name` is the base name of the object mesh, like "mug", "bottle", etc.

        HAVE TO SORT the instance names, so that "mug" comes before "mug.001", which is aligned with the order in the YAML file.
        """

        objs_pose_path = Path(pose_path_folder) / f"{sequence_name}/obj_pose.npz"
        if not objs_pose_path.is_file():
            raise FileNotFoundError(f"Missing HUMOTO object poses: {objs_pose_path}")
        _objs_mesh_path = Path(mesh_path_folder)  # directory containing all 735 object meshes
        if not _objs_mesh_path.is_dir():
            raise FileNotFoundError(f"Missing HUMOTO object meshes: {_objs_mesh_path}")
        
        list_objects = []

        with np.load(objs_pose_path, allow_pickle=True) as pose_data:
            for instance_name in sorted(pose_data.files, key= lambda instance_name: f"object_{instance_name}.npz"): #E01  
                pose = np.asarray(pose_data[instance_name], dtype=np.float32)
                quaternion_wxyz = pose[:, :4]
                quaternion_norm = np.linalg.norm(quaternion_wxyz, axis=1)
                if not np.allclose(quaternion_norm, 1.0, atol=1e-4):
                    raise ValueError(f"{instance_name}: non-unit object quaternion")
                quaternion_xyzw = quaternion_wxyz[:, [1, 2, 3, 0]]
                rotation = Rotation.from_quat(quaternion_xyzw).as_matrix().astype(np.float32) # it's ok to skip that, and using this instead: angles = (Rotation.from_quat(quaternion_xyzw).as_rotvec().astype(np.float32) )
                angles = Rotation.from_matrix(rotation).as_rotvec().astype(np.float32)  

                mesh_name = instance_name.split(".", 1)[0]  # in case instance_name has a suffix like ".001"(mug.001)
                objs_mesh_path = _objs_mesh_path / mesh_name / f"{mesh_name}.obj"
                mesh = trimesh.load_mesh(objs_mesh_path, process=False)
                obj = ObjectSequence(
                    instance_name=instance_name,
                    mesh_name=mesh_name,
                    mesh=mesh,
                    angles=angles,
                    trans=pose[:, 4:7],
                )
                list_objects.append(obj)

        return list_objects

    @staticmethod
    def _load_object_sequences(object_sequence_path: Path) -> Dict[str, np.ndarray]:
        raise DeprecationWarning
        if not object_sequence_path.is_file():
            raise FileNotFoundError(f"Missing HUMOTO object sequence: {object_sequence_path}")
        with np.load(object_sequence_path, allow_pickle=True) as f:
            obj_sequence = {
                k: f[k] for k in f.files
            }
        return obj_sequence

    @staticmethod
    def resolve_object_sequence_paths(sequence_path: Path) -> List[Path]:
        raise DeprecationWarning
        return list(sequence_path.glob("object_*.npz"))

    def _get_body_vertices_from_smpl(self, human_sequence: HumanSequence, dict_smpl_model: Optional[Dict[str, smplx.SMPLH]] = None) -> np.ndarray:
        """Copied from process_behave.py.  Get vertices from SMPL-H parameters."""
        
        # For those who initialized the Processor with smpl params.
        if dict_smpl_model is None:
            assert hasattr(self, "dict_smpl_model"), "dict_smpl_model is not initialized. Please provide it or initialize it."
            dict_smpl_model = self.dict_smpl_model
            logger.debug("Using default SMPL-H model for vertex extraction.")
            
        smpl_model = dict_smpl_model[human_sequence.gender]
        
        smplx_output = smpl_model(
            body_pose=torch.from_numpy(human_sequence.poses[:, 3:66]).float(),
            global_orient=torch.from_numpy(human_sequence.poses[:, :3]).float(),
            left_hand_pose=torch.from_numpy(human_sequence.poses[:, 66:111]).float(),
            right_hand_pose=torch.from_numpy(human_sequence.poses[:, 111:156]).float(),
            betas=torch.from_numpy(human_sequence.betas).float(),
            transl=torch.from_numpy(human_sequence.trans).float(),
        )
        verts = smplx_output.vertices.detach().numpy()
        return verts

    def _rotate_HOI_at_given_angle(self, human_sequence: HumanSequence, objs_sequence: List[ObjectSequence], angle: float = 0.0):
        logger.debug(f"Rotating the whole HOI by {angle} radians.")
        return human_sequence, objs_sequence  # TODO: Implement rotation logic if needed

    def _make_sure_HOI_on_ground(
        self,
        human_sequence: HumanSequence,
        objs_sequence: List[ObjectSequence],
        dict_smpl_model: Optional[Dict[str, smplx.SMPLH]] = None,
    ):
        if not objs_sequence:
            raise ValueError("Cannot compute ground height without objects")

        # Only the first 30 frames are used for ground estimation.
        human_sample_count = min(
            30,
            human_sequence.frame_count,
        )
        human_sample = human_sequence.slice(
            0,
            human_sample_count,
        )

        with torch.no_grad():
            human_verts = self._get_body_vertices_from_smpl(
                human_sample,
                dict_smpl_model,
            )

        minimum_y = float(
            human_verts[..., 1].min()
        )

        for obj in objs_sequence:
            sample_count = min(
                30,
                obj.frame_count,
            )

            local_vertices = np.asarray(
                obj.mesh.vertices,
                dtype=np.float32,
            )
            rotations = (
                Rotation
                .from_rotvec(obj.angles[:sample_count])
                .as_matrix()
                .astype(np.float32)
            )
            translations = np.asarray(
                obj.trans[:sample_count],
                dtype=np.float32,
            )

            # Existing transformation:
            # world_vertices = local_vertices @ rotation.T + trans
            #
            # We only need its Y component:
            # world_y = local_vertices @ rotation[1, :] + trans_y
            for frame_index in range(sample_count):
                world_y = (
                    local_vertices
                    @ rotations[frame_index, 1, :]
                    + translations[frame_index, 1]
                )
                minimum_y = min(
                    minimum_y,
                    float(world_y.min()),
                )

        diff_fix = minimum_y

        for obj in objs_sequence:
            obj.trans[..., 1] -= diff_fix

        human_sequence.trans[..., 1] -= diff_fix

        return (
            human_sequence,
            objs_sequence,
            diff_fix,
        )


    # def _make_sure_HOI_on_ground(self, human_sequence: HumanSequence, objs_sequence: List[ObjectSequence], dict_smpl_model: Optional[Dict[str, smplx.SMPLH]] = None):
    #     """Copied from process_behave.py.  Lift the whole HOI so that it was above the line:y=0"""
    #     human_verts = self._get_body_vertices_from_smpl(human_sequence, dict_smpl_model)

    #     obj_verts_min_within_first_30_frames = np.inf
    #     for obj in objs_sequence:
    #         obj_verts = obj.mesh.vertices[None, ...]
    #         obj_trans = obj.trans
    #         angle_matrix = Rotation.from_rotvec(obj.angles).as_matrix()

    #         obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
    #         if obj_verts[:30,:,1].min() < obj_verts_min_within_first_30_frames:
    #             obj_verts_min_within_first_30_frames = obj_verts[:30,:,1].min()
                
    #     diff_fix = min(human_verts[:30,:,1].min(), obj_verts_min_within_first_30_frames)
    #     for obj in objs_sequence:
    #         obj.trans[..., 1] -= diff_fix
    #     human_sequence.trans[..., 1] -= diff_fix
        

    #     return human_sequence, objs_sequence, diff_fix # maybe these two have already been modified? because the params that passed in are pointer. CHECK!!!

if __name__ == "__main__":
    dict_test_path = {
        'MOTION_PATH': './data/humoto_test/sequences',
        'OBJECT_PATH': './data/humoto_test/objects',
        'METADATA_PATH': './data/humoto_test/raw/humoto_0805',
        'MODEL_PATH': './models',
        'MOTION_PATH_RAW_HUMAN': './data/humoto_test/raw/smplh',
        'MOTION_PATH_RAW_OBJECT': './data/humoto_test/raw/output_process',
        'OBJECT_PATH_RAW': './data/humoto_test/raw/humoto_objects_0805',
        'PATH_RAW': './data/humoto_test/raw',
        'PATH_ROOT': './data/humoto_test/',
    }
    dict_path = dict_test_path  # Switch between test and full dataset by changing this line
    procer = Processor(dict_path, smpl_init=False) 
    name = "drinking_from_mug1_and_talking-277"
    sequence_folder = dict_path['MOTION_PATH']
    procer._load_human_sequence(dict_path['MOTION_PATH_RAW_HUMAN'], name, False)
    procer._load_human_sequence(sequence_folder, name, True)
    exit()
    