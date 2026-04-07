# NOTE: Canonicalize the first human pose

import os
import os.path
import numpy as np
import torch
import smplx
import trimesh
from scipy.spatial.transform import Rotation



import shutil
import sys
sys.path.append('.')
from human_body_prior.body_model.body_model import BodyModel

from process.markerset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


MODEL_PATH = './models'
SUBJECT_SMPLX_CACHE = {}


def _safe_subject_id(name):
    if '_' not in name:
        return None
    return name.split('_', 1)[0]


def _get_subject_template_path(dataset, subject_id):
    dataset = dataset.lower()
    candidates = [
        os.path.join('./data', dataset, 'raw', 'meta', 'subject_vtemplates', f'{subject_id}.obj'),
        os.path.join('./data', dataset, 'meta', 'subject_vtemplates', f'{subject_id}.obj'),
        os.path.join('./data', dataset, 'subject_vtemplates', f'{subject_id}.obj'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_subject_smplx_model(dataset, subject_id, gender):
    cache_key = (dataset.lower(), subject_id, gender)
    if cache_key in SUBJECT_SMPLX_CACHE:
        return SUBJECT_SMPLX_CACHE[cache_key]

    vtemplate_path = _get_subject_template_path(dataset, subject_id)
    if vtemplate_path is None:
        raise FileNotFoundError(
            f'Cannot find subject template for {dataset}/{subject_id} in expected subject_vtemplates paths.'
        )
    mesh = trimesh.load_mesh(vtemplate_path, force='mesh')
    vtemplate = mesh.vertices
    smpl_model = smplx.create(
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=gender,
        num_pca_comps=45,
        v_template=vtemplate,
        flat_hand_mean=True,
        use_pca=False,
    )
    SUBJECT_SMPLX_CACHE[cache_key] = smpl_model
    return smpl_model


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
                "data": object_data,
                "name": _object_name_to_str(object_data["name"]),
            }
        )
        return object_entries

    object_files = sorted(
        [f for f in os.listdir(sequence_dir) if f.startswith("object_") and f.endswith(".npz")]
    )
    for object_file in object_files:
        with np.load(os.path.join(sequence_dir, object_file), allow_pickle=True) as f:
            object_data = {k: f[k] for k in f.files}
        object_entries.append(
            {
                "filename": object_file,
                "data": object_data,
                "name": _object_name_to_str(object_data["name"]),
            }
        )
    if not object_entries:
        raise FileNotFoundError(
            f"Cannot find object parameters in '{sequence_dir}'. "
            "Expected 'object.npz' or 'object_*.npz'."
        )
    return object_entries


def resolve_object_mesh_path(object_root, object_name, object_filename=None):
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
    raise FileNotFoundError(
        f"Cannot find an object mesh for '{object_name}' under '{object_root}'. "
        f"Tried dirs: {dir_candidates}"
    )


def _prepare_betas_for_model(betas, smpl_model):
    """Trim/pad betas to match the target model expectation."""
    betas_arr = np.asarray(betas, dtype=np.float32)
    if betas_arr.ndim == 0:
        betas_arr = betas_arr.reshape(1)
    elif betas_arr.ndim > 1:
        betas_arr = betas_arr.reshape(-1, betas_arr.shape[-1])[0]

    target_betas = getattr(smpl_model, 'num_betas', None)
    if target_betas is None:
        shapedirs = getattr(smpl_model, 'shapedirs', None)
        if shapedirs is not None:
            target_betas = int(shapedirs.shape[-1])

    if target_betas is None:
        return betas_arr.astype(np.float32)

    target_betas = int(target_betas)
    if betas_arr.shape[0] > target_betas:
        betas_arr = betas_arr[:target_betas]
    elif betas_arr.shape[0] < target_betas:
        betas_arr = np.pad(betas_arr, (0, target_betas - betas_arr.shape[0]), mode='constant')
    return betas_arr.astype(np.float32)

######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl')


smplh10 = {'male': smplh_model_male, 'female': smplh_model_female}
######################################## smplx 10 ########################################
smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                        gender = 'male',
                        use_pca=False,
                        ext='pkl')
                           
smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="female",
                        use_pca=False,
                        ext='pkl')

smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl')

smplx10 = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}
######################################## smplx 10 pca 12 ########################################
smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="male",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')

smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="female",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')
smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="neutral",
                          num_pca_comps=12,
                          use_pca=True,
                          flat_hand_mean = True,
                          ext='pkl')
smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
######################################## smplh 16 ########################################
SMPLH_PATH = MODEL_PATH+'/smplh'
surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 

smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}
######################################## smplx 16 ########################################
SMPLX_PATH = MODEL_PATH+'/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                num_betas=num_betas,
                num_expressions=num_expressions,
                num_dmpls=num_dmpls,
                dmpl_fname=dmpl_fname)
smplx16 = {'male': smplx16_model_male, 'female': smplx16_model_female, 'neutral': smplx16_model_neutral}
########################################################################################
results_folder = "./results"
os.makedirs(results_folder, exist_ok=True)

######################################## Visualize SMPL ########################################
def visualize_smpl(name, MOTION_PATH, model_type, num_betas, num_pca_comps=None, dataset=None, subject_id=None):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 10 PCA 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    
    
    frame_times = poses.shape[0]
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            model_betas = _prepare_betas_for_model(betas, smpl_model)
            smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                betas=torch.from_numpy(model_betas[None, :]).repeat(frame_times, 1).float(),
                                transl=torch.from_numpy(trans).float(),) 
        elif model_type == 'smplx':
            if num_pca_comps == 12:
                smpl_model = smplx12[gender]
                model_betas = _prepare_betas_for_model(betas, smpl_model)
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:78]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 78:90]).float(),
                              jaw_pose=torch.zeros(frame_times, 3).float(),
                              leye_pose=torch.zeros(frame_times, 3).float(),
                              reye_pose=torch.zeros(frame_times, 3).float(),
                              expression=torch.zeros(frame_times, 10).float(),
                              betas=torch.from_numpy(model_betas[None, :]).repeat(frame_times, 1).float(),
                              transl=torch.from_numpy(trans).float(),)
            else:
                use_subject_template = dataset is not None and dataset.upper() in ['ARCTIC'] and subject_id is not None
                if use_subject_template:
                    smpl_model = get_subject_smplx_model(dataset, subject_id, gender)
                else:
                    smpl_model = smplx10[gender]
                model_betas = _prepare_betas_for_model(betas, smpl_model)
                if poses.shape[1] >= 165:
                    jaw_pose = torch.from_numpy(poses[:, 156:159]).float()
                    leye_pose = torch.from_numpy(poses[:, 159:162]).float()
                    reye_pose = torch.from_numpy(poses[:, 162:165]).float()
                else:
                    jaw_pose = torch.zeros([frame_times,3]).float()
                    leye_pose = torch.zeros([frame_times,3]).float()
                    reye_pose = torch.zeros([frame_times,3]).float()
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                    global_orient=torch.from_numpy(poses[:, :3]).float(),
                                    left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                    right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                    jaw_pose = jaw_pose,
                                    reye_pose = reye_pose,
                                    leye_pose = leye_pose,
                                    expression = torch.zeros([frame_times,10]).float(),
                                    betas=torch.from_numpy(model_betas[None, :]).repeat(frame_times, 1).float(),
                                    transl=torch.from_numpy(trans).float(),)
        verts = to_cpu(smplx_output.vertices)
        faces = smpl_model.faces
        joints = to_cpu(smplx_output.joints)
    elif num_betas == 16: 
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        model_betas = _prepare_betas_for_model(betas, smpl_model)
        smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
                            pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
                            betas=torch.from_numpy(model_betas[None, :]).repeat(frame_times, 1).float(), 
                            root_orient=torch.from_numpy(poses[:, :3]).float(), 
                            trans=torch.from_numpy(trans).float())
        verts = to_cpu(smplx_output.v)
        faces = smpl_model.f
        joints = to_cpu(smplx_output.Jtr)
    
    return verts, faces, joints


######################################## Visualize GRAB ########################################
def visualize_grab(name, MOTION_PATH):
    """
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, vtemp, trans, gender = f['poses'], f['vtemp'], f['trans'], str(f['gender'])
    n_comps = 24
    T = len(poses)

    smpl_model = smplx.create( 
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=gender,
        num_pca_comps=n_comps,
        v_template = vtemp,
        batch_size=T)

    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                            global_orient=torch.from_numpy(poses[:, :3]).float(),
                            left_hand_pose=torch.from_numpy(poses[:, 66:90]).float(),
                            right_hand_pose=torch.from_numpy(poses[:, 90:114]).float(),
                            transl=torch.from_numpy(trans).float(),)
    verts = to_cpu(smplx_output.vertices)
    faces = smpl_model.faces
    joints = to_cpu(smplx_output.joints)

    return verts, faces, joints



if __name__ == "__main__":
    datasets = ['behave', 'intercap', 'omomo', 'grab', 'arctic', 'parahome']
    data_root = './data'
    for dataset in datasets:
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences')
        NEW_MOTION_PATH = os.path.join(dataset_path, 'sequences_canonical')
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        data_name = os.listdir(MOTION_PATH)
        for name in data_name:
            try:
                print('Processing sequence:', dataset, name)
                subject_id = _safe_subject_id(name)
                if dataset.upper() == 'GRAB':
                    verts, faces, joints = visualize_grab(name, MOTION_PATH)
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'BEHAVE':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
                    markers = verts[:,markerset_smplh]
                elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
                    markers = verts[:,markerset_smplh]
                elif dataset.upper() == 'CHAIRS':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'INTERCAP':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplx', 10, 12)
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'ARCTIC':
                    verts, faces, joints = visualize_smpl(
                        name, MOTION_PATH, 'smplx', 10, dataset=dataset, subject_id=subject_id
                    )
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'PARAHOME':
                    verts, faces, joints = visualize_smpl(
                        name, MOTION_PATH, 'smplx', 10, dataset=dataset, subject_id=subject_id
                    )
                    markers = verts[:,markerset_smplx]
                elif dataset.upper() == 'OMOMO':
                    verts, faces, joints = visualize_smpl(name, MOTION_PATH, 'smplx', 16)
                    markers = verts[:,markerset_smplx]
                np.save(os.path.join(MOTION_PATH, name, 'markers.npy'), markers)
                centroid = joints[0,0]
            
                object_entries = _load_object_entries(os.path.join(MOTION_PATH, name))

                with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
                    if dataset.upper() == 'GRAB':
                        poses, vtemp, trans, gender = f['poses'], f['vtemp'], f['trans'], str(f['gender'])
                    else:
                        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
                global_orient = Rotation.from_rotvec(poses[0,:3]).as_matrix()
                rotation_v = np.eye(3).astype(np.float32)
                cos, sin = global_orient[0, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2), global_orient[2, 0] / np.sqrt(global_orient[0, 0]**2 + global_orient[2, 0]**2)
                rotation_v[[0, 2, 0, 2], [0, 2, 2, 0]] = np.array([cos, cos, -sin, sin])
                rotation = np.linalg.inv(rotation_v).astype(np.float32)

                new_poses = []
                new_trans = []
                new_objects_state = []
                for entry in object_entries:
                    new_objects_state.append(
                        {
                            "filename": entry["filename"],
                            "name": entry["name"],
                            "data": entry["data"],
                            "new_angles": [],
                            "new_trans": [],
                        }
                    )
                for i in range(poses.shape[0]):
                    smplfit_params = {'pose': poses[i].copy(), 'trans': trans[i].copy()}
                    pelvis = joints[i,0]
                    smplfit_params['trans'] = smplfit_params['trans'] - centroid
                    pelvis = pelvis - centroid
                    pelvis_original = pelvis - smplfit_params['trans'] # pelvis position in original smpl coords system
                    smplfit_params['trans'] = np.dot(smplfit_params['trans'] + pelvis_original, rotation.T) - pelvis_original
                    pelvis = np.dot(pelvis, rotation.T)
                    # smpl pose parameter in the canonical system
                    r_ori = Rotation.from_rotvec(smplfit_params['pose'][:3])
                    r_new = Rotation.from_matrix(rotation) * r_ori
                    smplfit_params['pose'][:3] = r_new.as_rotvec()
                    new_poses.append(smplfit_params['pose'])
                    new_trans.append(smplfit_params['trans'])

                    for obj_state in new_objects_state:
                        src_angles = obj_state["data"]["angles"][i].copy()
                        src_trans = obj_state["data"]["trans"][i].copy()
                        src_trans = src_trans - centroid
                        src_trans = np.dot(src_trans, rotation.T)
                        r_ori = Rotation.from_rotvec(src_angles)
                        r_new = Rotation.from_matrix(rotation) * r_ori
                        obj_state["new_angles"].append(r_new.as_rotvec())
                        obj_state["new_trans"].append(src_trans)
                
                if dataset.upper() == 'GRAB':   
                    new_human = {
                        'poses': np.array(new_poses),
                        'vtemp': vtemp,
                        'trans': np.array(new_trans),
                        'gender': gender
                    }
                else:
                    new_human = {
                        'poses': np.array(new_poses),
                        'betas': betas,
                        'trans': np.array(new_trans),
                        'gender': gender
                    }
                os.makedirs(os.path.join(NEW_MOTION_PATH, name), exist_ok=True)
                np.savez(os.path.join(NEW_MOTION_PATH, name, 'human.npz'), **new_human)

                # get smpl vertices and object vertices
                if dataset.upper() == 'GRAB':
                    verts, faces, joints = visualize_grab(name, NEW_MOTION_PATH)
                elif dataset.upper() == 'BEHAVE':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplh', 10)
                elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplh', 16)
                elif dataset.upper() == 'CHAIRS':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplx', 10)
                elif dataset.upper() == 'INTERCAP':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplx', 10, 12)
                elif dataset.upper() == 'ARCTIC':
                    verts, faces, joints = visualize_smpl(
                        name, NEW_MOTION_PATH, 'smplx', 10, dataset=dataset, subject_id=subject_id
                    )
                elif dataset.upper() == 'PARAHOME':
                    verts, faces, joints = visualize_smpl(
                        name, NEW_MOTION_PATH, 'smplx', 10, dataset=dataset, subject_id=subject_id
                    )
                elif dataset.upper() == 'OMOMO':
                    verts, faces, joints = visualize_smpl(name, NEW_MOTION_PATH, 'smplx', 16)
                
                object_min_y = []
                for obj_state in new_objects_state:
                    obj_state["new_angles"] = np.array(obj_state["new_angles"])
                    obj_state["new_trans"] = np.array(obj_state["new_trans"])
                    mesh_obj_path = resolve_object_mesh_path(
                        OBJECT_PATH, obj_state["name"], object_filename=obj_state["filename"]
                    )
                    mesh_obj = trimesh.load(mesh_obj_path, force='mesh')
                    angle_matrix = Rotation.from_rotvec(obj_state["new_angles"]).as_matrix()
                    sample_frames = min(30, angle_matrix.shape[0], obj_state["new_trans"].shape[0])
                    if sample_frames > 0:
                        obj_verts = mesh_obj.vertices[None, ...]
                        obj_verts = (
                            np.matmul(obj_verts, np.transpose(angle_matrix[:sample_frames], (0, 2, 1)))
                            + obj_state["new_trans"][:sample_frames, None, :]
                        )
                        object_min_y.append(float(obj_verts[..., 1].min()))

                diff_candidates = [float(verts[:30, ..., 1].min())]
                diff_candidates.extend(object_min_y)
                diff_fix = min(diff_candidates)
                new_trans = np.array(new_trans)
                for obj_state in new_objects_state:
                    obj_state["new_trans"][..., 1] -= diff_fix
                new_trans[..., 1] -= diff_fix

                if dataset.upper() == 'GRAB':
                    new_human = {
                        'poses': np.array(new_poses),
                        'vtemp': vtemp,
                        'trans': np.array(new_trans),
                        'gender': gender
                    }
                else:
                    new_human = {
                        'poses': np.array(new_poses),
                        'betas': betas,
                        'trans': np.array(new_trans),
                        'gender': gender
                    }
                np.savez(os.path.join(NEW_MOTION_PATH, name, 'human.npz'), **new_human)
                for obj_state in new_objects_state:
                    new_obj = {
                        'angles': np.array(obj_state["new_angles"]),
                        'trans': np.array(obj_state["new_trans"]),
                        'name': obj_state["name"]
                    }
                    for k, v in obj_state["data"].items():
                        if k not in new_obj:
                            new_obj[k] = v
                    np.savez(os.path.join(NEW_MOTION_PATH, name, obj_state["filename"]), **new_obj)
                if os.path.exists(os.path.join(MOTION_PATH, name, 'text.txt')):
                    shutil.copy(os.path.join(MOTION_PATH, name, 'text.txt'), os.path.join(NEW_MOTION_PATH, name, 'action.txt'))
                    shutil.copy(os.path.join(MOTION_PATH, name, 'text.txt'), os.path.join(NEW_MOTION_PATH, name, 'action.npy'))
                    shutil.copy(os.path.join(MOTION_PATH, name, 'text.txt'), os.path.join(NEW_MOTION_PATH, name, 'text.txt'))
            except Exception as e:
                print(e)
                print(name)

                
            




                
