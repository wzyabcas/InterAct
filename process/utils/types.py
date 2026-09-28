import numpy as np
import trimesh
import torch
import os
import logging

from scipy.spatial.transform import Rotation
from typing import Tuple, Dict, Any, List, Union
from pathlib import Path
from dataclasses import dataclass

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger=logging.getLogger(__name__)

@dataclass
class ObjectSequence():
    instance_name: str # spoon, spoon.001. spoon.002, etc.
    mesh_name: str  # spoon, spoon, spoon, etc. 
    mesh: trimesh.Trimesh   # vertices and faces
    angles: np.ndarray # Rotation Vectors in radians. ||r|| is the angle of rotation, r/||r|| is the axis of rotation. Shape: (T, 3)
    trans: np.ndarray # Shape: (T, 3)

    def __post_init__(self):
        
        # check type
        if not isinstance(self.instance_name, str):
            raise TypeError(f"Expected instance_name to be str, got {type(self.instance_name)}")
        if not isinstance(self.mesh_name, str):
            raise TypeError(f"Expected mesh_name to be str, got {type(self.mesh_name)}")
        if not isinstance(self.mesh, trimesh.Trimesh):
            raise TypeError(f"Expected mesh to be trimesh.Trimesh, got {type(self.mesh)}")
        if not isinstance(self.angles, np.ndarray):
            raise TypeError(f"Expected angles to be np.ndarray, got {type(self.angles)}")
        if not isinstance(self.trans, np.ndarray):
            raise TypeError(f"Expected trans to be np.ndarray, got {type(self.trans)}")
        # check shape
        if self.angles.ndim != 2 or self.angles.shape[1] != 3:
            raise ValueError(f"Expected angles (frame_count, 3), got {self.angles.shape}")
        self.frame_count = self.angles.shape[0]
        if self.trans.shape != (self.frame_count, 3):
            raise ValueError(f"Expected trans ({self.frame_count}, 3), got {self.trans.shape}")
        # if self.mesh_name != self.instance_name:
        #     print("#"*50)
        #     print("# Object name mismatch: ", self.mesh_name, self.instance_name)
        #     print("#"*50)
        #     # raise ValueError(f"Object name mismatch: {self.mesh_name} != {self.instance_name}")


    def slice(self, slice_start: int, slice_end: int) -> "ObjectSequence":
        """Return a new ObjectSequence for [slice_start, slice_end)."""
        if not isinstance(slice_start, int) or not isinstance(slice_end, int):
            raise TypeError(f"slice_start and slice_end must be integers, got {type(slice_start)} and {type(slice_end)}")

        if (
            slice_start < 0
            or slice_end > self.frame_count
            or slice_start >= slice_end
        ):
            raise ValueError(
                f"Invalid slice range "
                f"[{slice_start}, {slice_end}) "
                f"for object {self.instance_name!r} "
                f"with frame_count {self.frame_count}"
            )

        return ObjectSequence(
            instance_name=self.instance_name,
            mesh_name=self.mesh_name,
            mesh=self.mesh,
            angles=self.angles[slice_start:slice_end].copy(),
            trans=self.trans[slice_start:slice_end].copy()
        )

    def dump(self, path: Union[str, Path]) -> Path:
        """Dump one ObjectSequence to object_{instance_name}.npz."""

        if not isinstance(path, (str, Path)):
            raise TypeError(
                f"Expected path to be str or Path, got {type(path)}"
            )
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError(
                f"Expected an '.npz' path, got {path}"
            )
        if path.exists():
            raise FileExistsError(
                f"File already exists: {path}.\nPlease delete it."
            )
        if self.angles.shape != (self.frame_count, 3):
            raise ValueError(
                f"Expected angles ({self.frame_count}, 3), "
                f"got {self.angles.shape}"
            )
        if self.trans.shape != (self.frame_count, 3):
            raise ValueError(
                f"Expected trans ({self.frame_count}, 3), "
                f"got {self.trans.shape}"
            )

        np.savez(path, instance_name=self.instance_name, mesh_name=self.mesh_name, mesh=self.mesh, angles=self.angles, trans=self.trans)

        return path

    @staticmethod
    def merge_object_meshes(
        objs_sequence: List["ObjectSequence"],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bake all object poses into world vertices and merge their meshes."""
        if not objs_sequence:
            raise ValueError("Cannot merge an empty object sequence list")

        frame_count = objs_sequence[0].trans.shape[0]
        world_vertex_parts = []
        face_parts = []
        vertex_offset = 0

        for obj in objs_sequence:
            angles = obj.angles.astype(np.float32)
            translations = np.asarray(obj.trans, dtype=np.float32)
            if angles.shape != (frame_count, 3):
                raise ValueError(
                    f"{obj.instance_name}: expected angles "
                    f"({frame_count}, 3), got {angles.shape}"
                )
            if translations.shape != (frame_count, 3):
                raise ValueError(
                    f"{obj.instance_name}: expected translations "
                    f"({frame_count}, 3), got {translations.shape}"
                )

            rotation_matrices = (
                Rotation.from_rotvec(angles)
                .as_matrix()
                .astype(np.float32)
            )
            local_vertices = np.asarray(obj.mesh.vertices, dtype=np.float32)
            local_faces = np.asarray(obj.mesh.faces, dtype=np.int64)
            world_vertices = np.matmul(
                local_vertices[None],
                np.transpose(rotation_matrices, (0, 2, 1)),
            ) + translations[:, None, :]

            world_vertex_parts.append(world_vertices)
            face_parts.append(local_faces + vertex_offset)
            vertex_offset += local_vertices.shape[0]

        return (
            np.concatenate(world_vertex_parts, axis=1),
            np.concatenate(face_parts, axis=0),
        )

@dataclass
class HumanSequence():
    poses: np.ndarray
    betas: np.ndarray
    trans: np.ndarray  
    gender: str   

    def __post_init__(self):
        # check shape
        if self.poses.ndim != 2 or self.poses.shape[1] != 156:
            raise ValueError(f"Expected poses (T, 156), got {self.poses.shape}")
        self.frame_count = self.poses.shape[0]
        if self.betas.shape != (10,):
            raise ValueError(f"HUMOTO expected 10 SMPL-H betas, got {self.betas.shape}")
        if self.trans.shape != (self.frame_count,3):
            raise ValueError(f"Expected trans ({self.frame_count}, 3), got {self.trans.shape}")
        if self.gender not in {"male", "female"}:
            raise ValueError(f"Unsupported gender {self.gender!r}")

        if not np.isfinite(self.poses).all() or not np.isfinite(self.betas).all() or not np.isfinite(self.trans).all():
            raise ValueError(f"Human parameters contain NaN or Inf: poses {self.poses}, betas {self.betas}, trans {self.trans}")
        
        self.betas = np.repeat(self.betas[None], self.frame_count, axis=0)  # (T, 10)
        assert self.betas.shape == (self.frame_count, 10), f"Expected betas ({self.frame_count}, 10), got {self.betas.shape}"
    
    def get_body_verts_faces_smpl(self, dict_smpl_model) -> Tuple[np.ndarray, np.ndarray]:
        """Get body vertices and faces from SMPL-H model."""
        smpl_model = dict_smpl_model[self.gender]
        # make sure the model's flat_hand_mean param is set to True.
        if not smpl_model.flat_hand_mean:
            raise ValueError(f"Expected SMPL-H model with flat_hand_mean=True, got {smpl_model.flat_hand_mean}")
        smpl_output = smpl_model(
            body_pose=torch.from_numpy(self.poses[:, 3:66]).float(),
            global_orient=torch.from_numpy(self.poses[:, :3]).float(),
            left_hand_pose=torch.from_numpy(self.poses[:, 66:111]).float(),
            right_hand_pose=torch.from_numpy(self.poses[:, 111:156]).float(),
            betas=torch.from_numpy(self.betas).float(),
            transl=torch.from_numpy(self.trans).float(),
        )
        verts = smpl_output.vertices.detach().numpy()
        faces = smpl_model.faces
        return verts, faces

    def slice(self, slice_start: int, slice_end: int) -> 'HumanSequence':
        """Return a new HumanSequence for [slice_start, slice_end)."""
        if not isinstance(slice_start, int) or not isinstance(slice_end, int):
            raise TypeError(f"slice_start and slice_end must be integers, got {type(slice_start)} and {type(slice_end)}")
        if slice_start < 0 or slice_end > self.frame_count or slice_start >= slice_end:
            raise ValueError(f"Invalid slice range: [{slice_start}, {slice_end}) for frame_count {self.frame_count}")

        if self.betas.shape == (self.frame_count, 10):
            if not np.allclose(self.betas, self.betas[0][None]):
                raise ValueError("Expected betas to be constant across frames.")
            sliced_betas = self.betas[0].copy()
        elif self.betas.shape == (10,):
            sliced_betas = self.betas.copy()
        else: 
            raise ValueError(f"Unexpected betas shape: {self.betas.shape}")

        result = HumanSequence(
            poses=self.poses[slice_start:slice_end].copy(),
            betas=sliced_betas,
            trans=self.trans[slice_start:slice_end].copy(),
            gender=self.gender
        )    
        expected_frame_count = slice_end - slice_start
        if result.frame_count != expected_frame_count:
            raise RuntimeError(f"Sliced HumanSequence has frame_count {result.frame_count}, expected {expected_frame_count}")
        return result

    def dump(self, path:Union[str, Path]) -> Path:
        if not isinstance(path, (str, Path)):
            raise TypeError(f"Expected path to be str or Path, got {type(path)}")
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError(f"Expected path to end with .npz, got {path}")
        if path.exists():
            raise FileExistsError(f"File already exists: {path}")
        
        if self.betas.shape == (self.frame_count, 10):
            if not np.allclose(self.betas, self.betas[0][None]):
                raise ValueError("Expected betas to be constant across frames.")
            betas_to_save = self.betas[0].copy()
        elif self.betas.shape == (10,):
            betas_to_save = self.betas
        else: raise ValueError(f"Unexpected betas shape: {self.betas.shape}")

        if self.poses.shape != (self.frame_count, 156):
            raise ValueError(f"Expected poses shape ({self.frame_count}, 156), got {self.poses.shape}")
        if self.trans.shape != (self.frame_count, 3):
            raise ValueError(f"Expected trans shape ({self.frame_count}, 3), got {self.trans.shape}")

        if not isinstance(self.gender, str):
            raise ValueError(f"Expected gender to be a string, got {type(self.gender)}")
        
        np.savez(path, poses=self.poses, betas=betas_to_save, trans=self.trans, gender=np.asarray(self.gender))

        return path


class ObjectsSequence:
    def __init__(
        self,
        sequence_dir: Union[str, Path, None] = None,
        object_entries: Union[List[ObjectSequence],None] = None,
    ):
        # from sequence_dir XOR from object_entries
        if (
            (sequence_dir is None)
            == (object_entries is None)
        ):
            raise ValueError(
                "Provide exactly one of sequence_dir "
                "or object_entries"
            )
        
        # The first way: load from a directory containing object_*.npz files
        if sequence_dir is not None:
            sequence_dir = Path(sequence_dir)

            if not sequence_dir.is_dir():
                raise FileNotFoundError(
                    f"Sequence directory does not exist: "
                    f"{sequence_dir}"
                )

            object_files = sorted(
                sequence_dir.glob("object_*.npz")
            )

            if not object_files:
                raise FileNotFoundError(
                    f"No object_*.npz files found in "
                    f"{sequence_dir}"
                )

            entries = []

            for object_path in object_files:
                with np.load(
                    object_path,
                    allow_pickle=True,
                ) as data:
                    expected_fields = {
                        "instance_name",
                        "mesh_name",
                        "mesh",
                        "angles",
                        "trans",
                    }

                    if set(data.files) != expected_fields:
                        raise ValueError(
                            f"Unexpected object fields in "
                            f"{object_path}: {data.files}"
                        )

                    obj = ObjectSequence(
                        instance_name=str(
                            data["instance_name"].item()
                        ),
                        mesh_name=str(
                            data["mesh_name"].item()
                        ),
                        mesh=data["mesh"].item(),
                        angles=np.asarray(
                            data["angles"],
                            dtype=np.float32,
                        ),
                        trans=np.asarray(
                            data["trans"],
                            dtype=np.float32,
                        ),
                    )

                entries.append(obj)

        # The second way: load from a list of ObjectSequence instances
        else:
            entries = list(object_entries)

        # Validate entries
        if not entries:
            raise ValueError(
                "ObjectsSequence cannot be empty"
            )

        if not all(
            isinstance(obj, ObjectSequence)
            for obj in entries
        ):
            raise TypeError(
                "Every object entry must be an ObjectSequence"
            )

        instance_names = [
            obj.instance_name
            for obj in entries
        ]

        if len(instance_names) != len(set(instance_names)):
            raise ValueError(
                f"Duplicate object instance names: "
                f"{instance_names}"
            )

        frame_count = entries[0].frame_count

        for obj in entries:
            if obj.frame_count != frame_count:
                raise ValueError(
                    f"Object {obj.instance_name!r} has "
                    f"{obj.frame_count} frames, expected "
                    f"{frame_count}"
                )

        self.object_entries = entries
        self.frame_count = frame_count

    def __len__(self) -> int:
        return len(self.object_entries)

    def __iter__(self):
        return iter(self.object_entries)

    def __getitem__(self, index):
        return self.object_entries[index]

    def slice(self, slice_start: int, slice_end: int) -> "ObjectsSequence":
        """Slice every object using identical frame bounds."""

        sliced_entries = [
            obj.slice(slice_start, slice_end)
            for obj in self.object_entries
        ]

        return ObjectsSequence(
            object_entries=sliced_entries
        )

    def dump(self, output_dir: Union[str, Path]) -> List[Path]:
        """Dump every object into one sequence directory."""

        if not isinstance(output_dir, (str, Path)):
            raise TypeError(
                f"Expected output_dir to be str or Path, "
                f"got {type(output_dir)}"
            )

        output_dir = Path(output_dir)

        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(
                f"Output path is not a directory: "
                f"{output_dir}"
            )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        dumped_paths = []

        for obj in self.object_entries:
            object_path = (output_dir / f"object_{obj.instance_name}.npz")
            dumped_paths.append(
                obj.dump(object_path)
            )

        return dumped_paths

@dataclass
class Sequence:
    start_frame: int
    end_frame: int
    human_sequence: HumanSequence
    objs_sequence: ObjectsSequence
    text_description: List[Dict[str, Union[int,str]]]
    extra_text_description: str
    seg_times: int

    def __post_init__(self):
        if self.start_frame != 1:
            raise ValueError(f"Expected start_frame to be 1, got {self.start_frame}")
        if not self.text_description:
            raise ValueError("text_description cannot be empty")
        
        # E04
        if self.text_description[-1]["end_frame"] != self.end_frame:
            logger.warning(
                "-" * 50 + "\n"
                + f"Expected last text_description end_frame to match end_frame, "
                + f"got {self.text_description[-1]['end_frame']} and {self.end_frame}.\n"
                + "now->end_frame is set to min(end_frame, long_script[-1]['end_frame'])\n"
                + f"now->end_frame = {min(self.end_frame, self.text_description[-1]['end_frame'])}\n"
                + "-" * 50
            )
            self.end_frame = min(self.end_frame, self.text_description[-1]['end_frame'])

        for iseg in range(len(self.text_description) - 1, -1, -1):
            seg = self.text_description[iseg]

            if seg["start_frame"] > seg["end_frame"]:
                raise ValueError(
                    f"Invalid text_description range: "
                    f"[{seg['start_frame']}, {seg['end_frame']}]"
                )

            # A segment is completely beyond the overall end_frame, so we remove it.
            if seg["start_frame"] > self.end_frame:
                self.text_description.pop(iseg)

            # A part of the segment is valid, but part of it exceeds the overall end_frame.
            elif seg["end_frame"] > self.end_frame:
                seg["end_frame"] = self.end_frame

        if not self.text_description:
            raise ValueError(
                "No text_description remains after clipping"
            )

        self.seg_times = len(self.text_description)

        assert all(
            seg["end_frame"] <= self.end_frame
            for seg in self.text_description
        )
                    

if __name__ == "__main__":
    # E04
    from pathlib import Path; import yaml
    path_root = Path("/media/volume/sxu1/jianqi/code/test/InterAct/data/humoto_test/raw/humoto_0805")
    for i,yaml_file in enumerate(path_root.glob("**/*.yaml")):
        with open(yaml_file, 'r') as f:
            seg_info = yaml.safe_load(f)
            start_frame = seg_info['start_frame']; end_frame = seg_info['end_frame']
            objects:List[str] = seg_info['objects']
            long_script:List[Dict[str, Union[str, int]]] = seg_info['long_script']
            short_script:str = seg_info['short_script']
            seg_times = len(long_script)
        logger.info(f"Loaded {yaml_file.resolve()}")
        sequence = Sequence(
            start_frame=start_frame,
            end_frame=end_frame,
            human_sequence=None,
            objs_sequence=None,
            text_description=long_script,
            extra_text_description=short_script,
            seg_times=seg_times
        )
