import os
import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation
import argparse
import smplx
import sys
sys.path.append('.')
sys.path.append('..')
from text2interaction.render.mesh_viz import visualize_body_obj

from human_body_prior.body_model.body_model import BodyModel
import json
from mesh import Mesh

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()

MODEL_PATH = 'models'

def build_smpl_model(gender, subject_id):
    vtemplate_p = f"./data/arctic/raw/meta/subject_vtemplates/{subject_id}.obj"
    mesh = Mesh(filename=vtemplate_p)
    vtemplate = mesh.v
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
    """
    Load and visualize SMPL data for a single sequence
    """
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

def align_table_mesh(table_mesh_path, table_trans):
    """
    Load table mesh and align it using transformed table pose from InterAct coordinate system.
    table_trans: (T, 3) array of table translations in InterAct coordinates (meters)
    Returns aligned vertices and faces.
    """
    table_mesh = trimesh.load(table_mesh_path)
    mesh_vertices = np.array(table_mesh.vertices)  # (N, 3) in meters
    mesh_faces = table_mesh.faces.astype(np.int32)
    
    # Find mesh top surface (max Z) and centroid
    mesh_max_z = np.max(mesh_vertices[:, 2])
    mesh_centroid_xy = np.mean(mesh_vertices[:, :2], axis=0)
    
    # table_trans[0] is the first frame's table position (table is static, so all frames same)
    # table_trans[0] = [centroid_x, centroid_y, top_z] in InterAct coordinates
    table_centroid_xy = table_trans[0, :2]  # (2,)
    table_top_z = table_trans[0, 2]  # scalar
    
    # Calculate translation needed
    # 1. Translate so mesh top aligns with table top
    z_translation = table_top_z - mesh_max_z
    
    # 2. Translate so mesh centroid aligns with table centroid
    xy_translation = table_centroid_xy - mesh_centroid_xy
    
    translation = np.array([xy_translation[0], xy_translation[1], z_translation])
    
    # Apply translation
    aligned_vertices = mesh_vertices + translation[None, :]
    
    return aligned_vertices, mesh_faces
    
def main():
    parser = argparse.ArgumentParser(description="Visualize a single sequence from a dataset")
    parser.add_argument("--dataset_path", required=True, help="Path to the root of the dataset")
    parser.add_argument("--sequence_name", required=True, help="Name of the sequence to visualize")
    parser.add_argument("--output_path", default="./visualization_output", help="Output directory for rendered video")
    
    args = parser.parse_args()
    
    # Derived paths
    human_path = os.path.join(args.dataset_path, 'sequences_canonical')
    object_path = os.path.join(args.dataset_path, 'objects')
    dataset_name = args.dataset_path.split('/')[-1]
    # Check if sequence exists
    sequence_path = os.path.join(human_path, args.sequence_name)
    subject_id = args.sequence_name.split("_")[0]
    print(subject_id)
    with np.load(os.path.join(human_path, args.sequence_name, "human.npz"), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])

    print(f"Motion loaded: {os.path.join(human_path, args.sequence_name, 'human.npz')}")
    verts, joints, faces, poses = visualize_smpl_arctic(poses, betas, trans, gender, subject_id)

    print(f"Visualizing sequence: {args.sequence_name}")
    print(f"Dataset: {dataset_name}")
    
    
    # Load object data
    try:
        with np.load(os.path.join(human_path, args.sequence_name, 'object.npz'), allow_pickle=True) as f:
            obj_angles, obj_trans, obj_name, obj_arti = f['angles'], f['trans'], str(f['name']), f['arti']
    except FileNotFoundError:
        print(f"Error: object.npz not found in {sequence_path}")
        return
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    # Load object mesh
    obj_mesh_path = os.path.join(object_path, obj_name, 'mesh.obj')
    
    OBJ_MESH = trimesh.load(obj_mesh_path)
    # Transform object vertices (with articulation on top-part only)
    # ---- Transform object vertices: articulate top, then apply global to all ----
    ov = np.array(OBJ_MESH.vertices).astype(np.float32)
    object_faces = OBJ_MESH.faces.astype(np.int32)

    parts_json = os.path.join(object_path, obj_name, "parts.json")
    if not os.path.exists(parts_json):
        raise FileNotFoundError(f"parts.json not found for {obj_name} at {parts_json}")
    parts_bool = np.array(json.load(open(parts_json, "r")), dtype=bool)  # True/False 标注

    # NOTE: INVERT_TOP_MASK must be True to match object_tensors.py behavior
    # In object_tensors.py: parts.json (1=top, 0=bottom) -> bool -> LongTensor (+1) -> parts_ids
    # After +1: original top (1->True->1->2), original bottom (0->False->0->1)
    # But object_tensors.py uses "parts_ids == 1" for top, which actually selects bottom!
    # So we need to invert the mask here to match that behavior.
    INVERT_TOP_MASK = True
    top_mask_np = parts_bool if not INVERT_TOP_MASK else ~parts_bool
    # ---------------------------------------------------------------

    ov_torch = torch.from_numpy(ov).float().to(device)           # (N,3)
    top_mask = torch.from_numpy(top_mask_np).to(device)          # (N,)
    bot_mask = ~top_mask                                         # (N,)

    ov_top = ov_torch[top_mask]                                  # (Nt,3)
    ov_bot = ov_torch[bot_mask]                                  # (Nb,3)


    # NOTE: z_axis must be [0,0,1] not [0,0,-1] because we transpose the rotation matrix below
    # In object_tensors.py: uses [0,0,-1] with quaternion_apply (standard rotation: R @ p)
    # Here we use .permute(0,2,1) which applies R^T (inverse rotation: R^T @ p = R^-1 @ p)
    # To compensate: R^-1([0,0,1], θ) = R([0,0,-1], θ), so we need opposite axis sign
    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    arti_vec = (obj_arti.reshape(-1, 1) * z_axis[None, :]).astype(np.float32)  # (T,3)
    arti_matrix = Rotation.from_rotvec(arti_vec).as_matrix()                   # (T,3,3)
    rot_arti = torch.from_numpy(arti_matrix).float().to(device)                # (T,3,3)

    
    rot_global = torch.tensor(angle_matrix).float().to(device)   # (T,3,3) 由 obj_angles 得到
    obj_trans_tensor = torch.tensor(obj_trans).float().to(device)  # (T,3) (米)

    T = obj_trans_tensor.shape[0]

    
    top_after_arti = torch.einsum('nj,tij->tni', ov_top, rot_arti.permute(0, 2, 1))              # (T,Nt,3)
    top_after_all  = torch.einsum('tni,tij->tnj', top_after_arti, rot_global.permute(0, 2, 1))   # (T,Nt,3)
    top_after_all  = top_after_all + obj_trans_tensor[:, None, :]                                 # (T,Nt,3)

   
    bot_rep        = ov_bot.unsqueeze(0).expand(T, -1, -1)                                        # (T,Nb,3)
    bot_after_all  = torch.einsum('tbi,tij->tbj', bot_rep, rot_global.permute(0, 2, 1))           # (T,Nb,3)
    bot_after_all  = bot_after_all + obj_trans_tensor[:, None, :]                                  # (T,Nb,3)

    
    object_verts = torch.zeros(T, ov_torch.shape[0], 3, device=device, dtype=torch.float32)
    object_verts[:, bot_mask, :] = bot_after_all
    object_verts[:, top_mask, :] = top_after_all


    # Prepare human mesh data
    human_verts = torch.from_numpy(verts.float().cpu().numpy()).to(device).float()
    human_face = torch.from_numpy(faces.astype(np.int32)).to(device)
    
    T = poses.shape[0]
    

    output_filename = f"{args.sequence_name}.mp4"
    # output_path = os.path.join(args.output_path, args.sequence_name)
    output_path = os.path.join(args.output_path, output_filename)
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
 
    
    # Render visualization
    print(f"Rendering to: {output_path}")
    visualize_body_obj(
        human_verts.float().detach().cpu().numpy(),
        faces.astype(np.int32),
        object_verts.detach().cpu().numpy(),
        object_faces,
        save_path=output_path,
        multi_angle=True,
        show_frame=True
    )
    
    print(f"Visualization complete: {output_path}")

if __name__ == "__main__":
    main() 