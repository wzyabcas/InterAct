import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _object_name_to_str(obj_name_raw):
    if isinstance(obj_name_raw, np.ndarray):
        if obj_name_raw.shape == ():
            return str(obj_name_raw.item())
        return str(obj_name_raw.reshape(-1)[0])
    return str(obj_name_raw)


def _load_object_entries(sequence_dir):
    """Load either single-object or multi-object sequence format."""
    single_object_path = os.path.join(sequence_dir, "object.npz")
    object_entries = []
    if os.path.exists(single_object_path):
        with np.load(single_object_path, allow_pickle=True) as f:
            object_data = {k: f[k] for k in f.files}
        object_entries.append(
            {
                "filename": "object.npz",
                "filepath": single_object_path,
                "data": object_data,
                "name": _object_name_to_str(object_data["name"]),
            }
        )
        return object_entries

    object_files = sorted(
        [f for f in os.listdir(sequence_dir) if f.startswith("object_") and f.endswith(".npz")]
    )
    for object_file in object_files:
        object_path = os.path.join(sequence_dir, object_file)
        with np.load(object_path, allow_pickle=True) as f:
            object_data = {k: f[k] for k in f.files}
        object_entries.append(
            {
                "filename": object_file,
                "filepath": object_path,
                "data": object_data,
                "name": _object_name_to_str(object_data["name"]),
            }
        )
    return object_entries


def resolve_object_mesh_path(object_root, object_name, object_filename=None):
    """
    Resolve mesh path for:
      - single mesh object: objects/<name>/<name>.obj
      - ARCTIC style: objects/<name>/mesh.obj (or mesh_tex/top/bottom)
      - part object: objects/<base_name>/<part>.obj
    """
    name = str(object_name)
    file_stem = None
    if object_filename is not None:
        file_stem = os.path.splitext(os.path.basename(object_filename))[0]
        if file_stem.startswith("object_"):
            file_stem = file_stem[len("object_"):]

    dir_candidates = [os.path.join(object_root, name)]
    part_hints = []
    part_tokens = {"base", "part1", "part2", "top", "bottom"}
    if "_" in name:
        base, suffix = name.rsplit("_", 1)
        if suffix in part_tokens:
            dir_candidates.append(os.path.join(object_root, base))
            part_hints.append(suffix)
    if file_stem is not None and "_" in file_stem:
        _, suffix = file_stem.rsplit("_", 1)
        if suffix in part_tokens and suffix not in part_hints:
            part_hints.append(suffix)

    for object_dir in dir_candidates:
        candidates = [
            os.path.join(object_dir, f"{name}.obj"),
            os.path.join(object_dir, "mesh.obj"),
            os.path.join(object_dir, "mesh_tex.obj"),
            os.path.join(object_dir, "top.obj"),
            os.path.join(object_dir, "bottom.obj"),
        ]
        for part in part_hints:
            candidates.append(os.path.join(object_dir, f"{part}.obj"))
        for mesh_path in candidates:
            if os.path.exists(mesh_path):
                return mesh_path
    return None


def canonicalize_mesh_and_get_center(mesh_path, mesh_center_cache):
    """
    Center a mesh at origin once, and return the original center.
    """
    if mesh_path in mesh_center_cache:
        return mesh_center_cache[mesh_path]

    mesh_obj = trimesh.load(mesh_path, force='mesh')
    obj_verts = mesh_obj.vertices
    center = obj_verts.mean(axis=0)
    mesh_obj.vertices = obj_verts - center[None, :]
    mesh_obj.export(mesh_path)
    mesh_center_cache[mesh_path] = center
    return center

if __name__ == "__main__":
    datasets = ['behave', 'intercap', 'grab', 'omomo', 'arctic', 'parahome']
    data_root = './data'
    for dataset in datasets:
        print("Processing dataset:", dataset)
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences')
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        if not os.path.isdir(MOTION_PATH) or not os.path.isdir(OBJECT_PATH):
            print(f"Skip dataset {dataset}: missing sequences or objects folder.")
            continue

        # Mesh path -> pre-centering centroid
        mesh_center_cache = {}

        data_name = os.listdir(MOTION_PATH)
        for name in data_name:
            print("Processing sequence:", name)
            seq_dir = os.path.join(MOTION_PATH, name)
            if not os.path.isdir(seq_dir):
                continue

            object_entries = _load_object_entries(seq_dir)
            if not object_entries:
                continue

            for entry in object_entries:
                object_data = entry["data"]
                obj_name = entry["name"]

                if "angles" not in object_data or "trans" not in object_data:
                    print(f"  Skip {entry['filename']}: missing 'angles' or 'trans'.")
                    continue

                mesh_path = resolve_object_mesh_path(
                    object_root=OBJECT_PATH,
                    object_name=obj_name,
                    object_filename=entry["filename"],
                )
                if mesh_path is None:
                    print(f"  Skip {entry['filename']}: cannot find mesh for '{obj_name}'.")
                    continue

                center = canonicalize_mesh_and_get_center(mesh_path, mesh_center_cache)

                obj_angles = object_data["angles"]
                obj_trans = object_data["trans"]
                rotation = Rotation.from_rotvec(obj_angles)
                new_obj_trans = obj_trans + rotation.apply(center)

                # Preserve all existing fields (e.g. ARCTIC's 'arti'), update only translation.
                object_data["trans"] = new_obj_trans
                np.savez(entry["filepath"], **object_data)
