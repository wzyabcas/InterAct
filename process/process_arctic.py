"""
Combined script to process ARCTIC dataset:
1. Convert raw sequences to InterAct format
2. Transform to canonical coordinate system
3. Scale object files from mm to meters
"""

import os
import os.path as op
import numpy as np
import torch
import json
import trimesh
import shutil
from glob import glob
from scipy.spatial.transform import Rotation as R
import smplx
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'text2interaction'))
from render.mesh_utils import Mesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from object_tensors import ObjectTensors
from preprocess_dataset import construct_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()

MODEL_PATH = './models'
SMPLX_MODEL_P = {
    "male": "./models/smplx/SMPLX_MALE.npz",
    "female": "./models/smplx/SMPLX_FEMALE.npz",
    "neutral": "./models/smplx/SMPLX_NEUTRAL.npz",
}

# ============ Part 1: arctic_to_inter utilities ============
def build_smplx(batch_size, gender, vtemplate):
    subj_m = smplx.create(
        model_path=SMPLX_MODEL_P[gender],
        model_type="smplx",
        gender=gender,
        num_pca_comps=45,
        v_template=vtemplate,
        flat_hand_mean=True,
        use_pca=False,
        batch_size=batch_size,
    )
    return subj_m

def construct_layers(dev, subject_id):
    with open("./data/arctic/raw/meta/misc.json", "r") as f:
        misc = json.load(f)
    vtemplate_p = f"./data/arctic/raw/meta/subject_vtemplates/{subject_id}.obj"
    mesh = Mesh(filename=vtemplate_p)
    vtemplate = mesh.vertices
    gender = misc[subject_id]["gender"]
    mano_layers = {
        "smplx": build_smplx(1, gender, vtemplate),
    }
    for layer in mano_layers.values():
        layer.to(dev)
    return mano_layers

def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing

def _resolve_gender(subject_id):
    with open("./data/arctic/raw/meta/misc.json", "r") as f:
        misc = json.load(f)
    gender = misc[subject_id]["gender"]
    return gender

def _pack_poses_axis_angle(batch):
    parts = [
        batch["smplx_global_orient"],
        batch["smplx_body_pose"],
        batch["smplx_left_hand_pose"],
        batch["smplx_right_hand_pose"],
        batch["smplx_jaw_pose"],
        batch["smplx_leye_pose"],
        batch["smplx_reye_pose"],
    ]
    parts = [p.reshape(p.shape[0], -1) for p in parts]
    poses = torch.cat(parts, dim=1)
    return poses

def _concat(list_of_arrays, axis=0):
    if not list_of_arrays:
        return None
    return np.concatenate(list_of_arrays, axis=axis)


def process_seq_params_direct(mano_p, dev, statcams, layers):
    with torch.no_grad():
        sid, seqname = mano_p.split("/")[-2:]
        seqname = seqname.replace(".mano.npy", "")
        out_root = f"./data/arctic/sequences/{sid}/{seqname}"
        os.makedirs(out_root, exist_ok=True)

        loader = construct_loader(mano_p)

        curr_bs = None
        smplx_m = None
        gender_str = "neutral"
        betas_subject = None

        poses_seq = []
        transl_seq = []
        obj_angles_seq = []
        obj_transl_seq = []
        obj_arti_seq = []
        qname_seq = []

        for batch in loader:
            batch = thing2dev(batch, dev)
            B = batch["smplx_transl"].shape[0]

            if B != curr_bs:
                curr_bs = B
                smplx_m = layers["smplx"]
                gender_str = _resolve_gender(sid)

                betas_tensor = getattr(smplx_m, "betas", None)
                if betas_tensor is None:
                    num_betas = getattr(smplx_m, "num_betas", 16)
                    betas_subject = np.zeros((num_betas,), dtype=np.float32)
                else:
                    bt = betas_tensor.detach().cpu().numpy()
                    betas_subject = (bt[0] if bt.ndim == 2 else bt).astype(np.float32)

            poses = _pack_poses_axis_angle(batch).detach().cpu().numpy()
            transl = batch["smplx_transl"].detach().cpu().numpy()

            poses_seq.append(poses)
            transl_seq.append(transl)

            obj_arti = batch["obj_arti"].view(-1, 1).detach().cpu().numpy()
            obj_grot = batch["obj_rot"].detach().cpu().numpy()
            obj_trans_m = (batch["obj_trans"] / 1000.0).detach().cpu().numpy()

            obj_angles_seq.append(obj_grot)
            obj_transl_seq.append(obj_trans_m)
            obj_arti_seq.append(obj_arti)

            qnames = batch["query_names"]
            if isinstance(qnames, (list, tuple)):
                qname_seq.extend([str(x) for x in qnames])
            else:
                qname_seq.append(str(qnames))

        poses_all = _concat(poses_seq, axis=0)
        trans_all = _concat(transl_seq, axis=0)
        angles_all = _concat(obj_angles_seq, axis=0)
        otrans_all = _concat(obj_transl_seq, axis=0)
        obj_arti_all = _concat(obj_arti_seq, axis=0)
        obj_name = qname_seq[0] if len(qname_seq) > 0 else ""

        human_path = op.join(out_root, "human.npz")
        np.savez(human_path,
                 poses=poses_all,
                 trans=trans_all,
                 betas=betas_subject,
                 gender=gender_str)

        object_path = op.join(out_root, "object.npz")
        np.savez(object_path,
                 angles=angles_all,
                 trans=otrans_all,
                 arti=obj_arti_all,
                 name=np.array(obj_name, dtype=object))

        return out_root, sid, seqname

# ============ Part 2: transform_arc utilities ============
def build_smpl_model(gender, subject_id):
    vtemplate_p = f"./data/arctic/raw/meta/subject_vtemplates/{subject_id}.obj"
    mesh = Mesh(filename=vtemplate_p)
    vtemplate = mesh.vertices
    smpl_model = smplx.create(
        model_path=MODEL_PATH,
        model_type="smplx",
        gender=gender,
        num_pca_comps=45,
        v_template=vtemplate,
        flat_hand_mean=True,
        use_pca=False,
    )
    return smpl_model

def visualize_smpl_arctic(poses, betas, trans, gender, subject_id):
    frame_times = poses.shape[0]
    smpl_model = build_smpl_model(gender, subject_id)

    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses[:, :3]).float(),
        left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
        right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
        jaw_pose=torch.from_numpy(poses[:, 156:159]).float(),
        leye_pose=torch.from_numpy(poses[:, 159:162]).float(),
        reye_pose=torch.from_numpy(poses[:, 162:165]).float(),
        expression=torch.zeros([frame_times,10]).float(),
        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
        transl=torch.from_numpy(trans).float(),)
    verts = to_cpu(smplx_output.vertices)
    joints= to_cpu(smplx_output.joints)
    faces = smpl_model.faces.astype(np.int32)
    
    return verts, joints, faces, poses

def _normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def _rot_between(a, b):
    a = _normalize(a).reshape(3,)
    b = _normalize(b).reshape(3,)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = _normalize(np.cross(a, [1,0,0])) if abs(a[0]) < 0.9 else _normalize(np.cross(a, [0,1,0]))
        return R.from_rotvec(np.pi * axis).as_matrix()
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3, dtype=np.float64)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c)/(s*s))

def _leftmul_rotvec(rotvecs, R_left):
    Rt = R.from_rotvec(rotvecs)
    Rl = R.from_matrix(R_left)
    return (Rl * Rt).as_rotvec()

def closest_axis_unit(v):
    v = _normalize(v).reshape(3,)
    axes = np.array([[ 1,0,0],[0, 1,0],[0,0, 1],
                     [-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    dots = axes @ v
    return axes[np.argmax(np.abs(dots))]

def transform_sequence_to_interact(human_npz, obj_npz, pivot='pelvis0', subject_id="s01"):
    poses  = human_npz['poses']
    trans  = human_npz['trans']
    betas  = human_npz['betas']
    gender = human_npz['gender'].item() if human_npz['gender'].shape == () else human_npz['gender']

    verts0, joints0, _, _ = visualize_smpl_arctic(poses, betas, trans, gender, subject_id)
    joints0_np = joints0.cpu().numpy()
    r = joints0_np[:, 0, :]

    j0 = joints0_np[0]
    up_src_cont = -_normalize(0.5*(j0[7]+j0[8]) - j0[0])
    up_src = closest_axis_unit(up_src_cont)

    up_tgt = np.array([0,1,0], dtype=np.float64)
    R_can  = _rot_between(up_src, up_tgt)

    if pivot == 'pelvis0':
        p0 = r[0].astype(np.float64)
    else:
        p0 = np.zeros(3, dtype=np.float64)

    orang = poses[:, :3]
    rest  = poses[:, 3:]

    orang_new = _leftmul_rotvec(orang, R_can)

    r_target = (R_can @ (r - p0[None,:]).T).T + p0[None,:]

    T = poses.shape[0]
    poses_upd = np.concatenate([orang_new, rest], axis=1).astype(np.float32)

    smpl_model = build_smpl_model(gender, subject_id)
    smpl_out = smpl_model(
        body_pose=torch.from_numpy(poses_upd[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses_upd[:, :3]).float(),
        left_hand_pose=torch.from_numpy(poses_upd[:, 66:111]).float(),
        right_hand_pose=torch.from_numpy(poses_upd[:, 111:156]).float(),
        jaw_pose=torch.from_numpy(poses_upd[:, 156:159]).float(),
        leye_pose=torch.from_numpy(poses_upd[:, 159:162]).float(),
        reye_pose=torch.from_numpy(poses_upd[:, 162:165]).float(),
        expression=torch.zeros([T,10]).float(),
        betas=torch.from_numpy(betas[None,:]).repeat(T,1).float(),
        transl=torch.zeros([T,3]).float(),
    )
    joints_local_new = smpl_out.joints.detach().cpu().numpy()
    r_local_new = joints_local_new[:, 0, :]

    trans_new = (r_target - r_local_new).astype(np.float32)

    obj_angles = obj_npz['angles']
    obj_trans  = obj_npz['trans']
    name       = obj_npz['name'].item() if obj_npz['name'].shape == () else str(obj_npz['name'])
    arti       = obj_npz.get('arti', None)

    obj_angles_new = _leftmul_rotvec(obj_angles, R_can)
    obj_trans_new = (R_can @ (obj_trans - r).T).T + r_target

    new_poses = np.concatenate([orang_new.astype(np.float32), rest.astype(np.float32)], axis=1)

    verts_h, _, _, _ = visualize_smpl_arctic(new_poses, betas, trans_new, gender, subject_id)
    min_y_h = float(verts_h[0, :, 1].min().cpu().numpy())

    min_y = min_y_h

    shift_y = max(0.0, -min_y)
    if shift_y > 0:
        dy = np.array([0.0, shift_y, 0.0], dtype=np.float32)
        trans_new = trans_new + dy[None, :]
        obj_trans_new = obj_trans_new + dy[None, :]

    new_human = dict(
        poses=new_poses,
        betas=betas,
        trans=trans_new,
        gender=gender
    )
    new_obj = dict(
        angles=obj_angles_new.astype(np.float32),
        trans=obj_trans_new.astype(np.float32),
        name=name
    )

    if arti is not None:
        new_obj['arti'] = arti

    return new_human, new_obj

# ============ Part 3: scale_to_meters utilities ============
def scale_obj_file_inplace(input_path, scale_factor):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    temp_path = input_path + '.tmp'
    with open(temp_path, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x_scaled = x * scale_factor
                y_scaled = y * scale_factor
                z_scaled = z * scale_factor
                f.write(f"v {x_scaled} {y_scaled} {z_scaled}\n")
            else:
                f.write(line)
    
    os.replace(temp_path, input_path)

def scale_json_coordinates_inplace(input_path, scale_factor):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    def scale_recursive(obj):
        if isinstance(obj, list):
            if all(isinstance(x, (int, float)) for x in obj):
                return [x * scale_factor for x in obj]
            else:
                return [scale_recursive(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: scale_recursive(value) for key, value in obj.items()}
        else:
            return obj
    
    scaled_data = scale_recursive(data)
    
    temp_path = input_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(scaled_data, f, indent=4)
    
    os.replace(temp_path, input_path)

def scale_object_files(input_dir, output_dir, scale_factor=0.001):
    """
    Copy all files from input_dir to output_dir, then scale specific files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy all files from input to output
    for item in os.listdir(input_dir):
        src_path = os.path.join(input_dir, item)
        dst_path = os.path.join(output_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
    
    # Scale specific files in the output directory
    obj_files = ['mesh.obj', 'mesh_tex.obj', 'top.obj', 'bottom.obj']
    json_files = ['object_params.json', 'top_keypoints_300.json', 'bottom_keypoints_300.json']
    
    for obj_file in obj_files:
        file_path = os.path.join(output_dir, obj_file)
        if os.path.exists(file_path):
            scale_obj_file_inplace(file_path, scale_factor)
    
    for json_file in json_files:
        file_path = os.path.join(output_dir, json_file)
        if os.path.exists(file_path):
            scale_json_coordinates_inplace(file_path, scale_factor)

# ============ Main processing logic ============
def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load metadata
    with open(f"./data/arctic/raw/meta/misc.json", "r") as f:
        misc = json.load(f)

    statcams = {}
    for sub in misc.keys():
        statcams[sub] = {
            "world2cam": torch.FloatTensor(np.array(misc[sub]["world2cam"])),
            "intris_mat": torch.FloatTensor(np.array(misc[sub]["intris_mat"])),
        }

    # Find all .mano.npy files
    mano_ps = glob(f"./data/arctic/raw/raw_seqs/*/*.mano.npy")
    
    
    # Process each sequence
    for mano_p in mano_ps:
        sid = mano_p.split("/")[-2]
        
        # Initialize layers for this subject
        layers = construct_layers(dev, sid)
        object_tensor = ObjectTensors()
        object_tensor.to(dev)
        layers["object"] = object_tensor
        
        # Step 1: Process raw sequence (arctic_to_inter)
        out_root, subject_id, seqname = process_seq_params_direct(mano_p, dev, statcams, layers)
        
        # Step 2: Transform to InterAct coordinate system
        human_path = os.path.join(out_root, "human.npz")
        object_path = os.path.join(out_root, "object.npz")
        
        with np.load(human_path, allow_pickle=True) as H, np.load(object_path, allow_pickle=True) as O:
            human_out, object_out = transform_sequence_to_interact(H, O, subject_id=subject_id)
        
        # Save to data/arctic/sequences
        output_dir = os.path.join("./data/arctic/sequences", subject_id + "_" + seqname)
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "human.npz"), **human_out)
        np.savez(os.path.join(output_dir, "object.npz"), **object_out)
        
        # Delete intermediate files
        if os.path.exists(out_root):
            shutil.rmtree(out_root)
            # Also remove parent directory if it's empty
            parent_dir = os.path.dirname(out_root)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
    
    # Step 3: Scale all object files from mm to meters
    input_objects_root = "./data/arctic/raw/meta/object_vtemplates"
    output_objects_root = "./data/arctic/objects"
    
    for obj_name in os.listdir(input_objects_root):
        input_obj_dir = os.path.join(input_objects_root, obj_name)
        if os.path.isdir(input_obj_dir):
            output_obj_dir = os.path.join(output_objects_root, obj_name)
            scale_object_files(input_obj_dir, output_obj_dir, scale_factor=0.001)

if __name__ == "__main__":
    main()

