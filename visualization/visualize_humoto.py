import json
import math
import os
import os.path
os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
import imageio
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from copy import copy
from PIL import Image, ImageDraw

import sys
sys.path.append('.')
sys.path.append('..')
from text2interaction.render.mesh_viz import visualize_body_obj
from text2interaction.render.mesh_utils import MeshViewer
from text2interaction.render.utils import colors
from human_body_prior.body_model.body_model import BodyModel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

dataset = sys.argv[1].upper()

MOTION_PATH = './data/{}/sequences_canonical'.format(dataset.lower())
OBJECT_PATH = './data/{}/objects'.format(dataset.lower())
MODEL_PATH = './models'

data_name = [sys.argv[2]] if len(sys.argv) > 2 else os.listdir(MOTION_PATH)

"""
```bash
  python visualization/visualize_modified.py humoto_test \
    carrying_plastic_bowl_stacked_with_both_hands-328_210_292_003
    pull_laptop_backward_put_mug_and_organizer_on_utility_cart_move_dining_chair-705_1_115_000
    drinking_from_mug1_and_talking-277_455_576_002
    cut_food_using_knife_and_eat_with_fork-150_1_113_000
    activating_floor_lamp_with_right_hand-485_1_181_000
```
"""

######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        flat_hand_mean=True,
                        ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        flat_hand_mean=True,
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
######################################## smplx 12 ########################################
smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="male",
                          num_pca_comps=12,
                          ext='pkl')

smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="female",
                          num_pca_comps=12,
                          ext='pkl')

smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="neutral",
                          num_pca_comps=12,
                          ext='pkl')

smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
######################################## smplh 16 ########################################
SMPLH_PATH = MODEL_PATH+'/smplh'
surface_model_male_fname = os.path.join(SMPLH_PATH,'male', "model.npz")
surface_model_female_fname = os.path.join(SMPLH_PATH, "female","model.npz")
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

######################################## Visualize SMPL ##############
###########################
def visualize_smpl(name, MOTION_PATH, model_type, num_betas, num_pca_comps=None):
    """
    BEHAVE for SMPLH 10
    NEURALDOME or IMHD for SMPLH 16
    vertices: (N, 6890, 3)
    Chairs for SMPLX 10
    InterCap for SMPLX 12
    OMOMO for SMPLX 16
    vertices: (N, 10475, 3)
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
        
    frame_times = poses.shape[0]
    if num_betas == 10:
        if model_type == 'smplh':
            smpl_model = smplh10[gender]
            smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                transl=torch.from_numpy(trans).float(),) 
        elif model_type == 'smplx':
            if num_pca_comps == 12:
                smpl_model = smplx12[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                    global_orient=torch.from_numpy(poses[:, :3]).float(),
                                    left_hand_pose=torch.from_numpy(poses[:, 66:78]).float(),
                                    right_hand_pose=torch.from_numpy(poses[:, 78:90]).float(),
                                    jaw_pose=torch.zeros(frame_times, 3).float(),
                                    leye_pose=torch.zeros(frame_times, 3).float(),
                                    reye_pose=torch.zeros(frame_times, 3).float(),
                                    expression=torch.zeros(frame_times, 10).float(),
                                    betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                    transl=torch.from_numpy(trans).float(),)
            else:
                smpl_model = smplx10[gender]
                smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                    global_orient=torch.from_numpy(poses[:, :3]).float(),
                                    left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                    right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                    jaw_pose = torch.zeros([frame_times,3]).float(),
                                    reye_pose = torch.zeros([frame_times,3]).float(),
                                    leye_pose = torch.zeros([frame_times,3]).float(),
                                    expression = torch.zeros([frame_times,10]).float(),
                                    betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                    transl=torch.from_numpy(trans).float(),)
        verts = to_cpu(smplx_output.vertices)
        faces = smpl_model.faces
    elif num_betas == 16: 
        if model_type == 'smplh':
            smpl_model = smplh16[gender]
        elif model_type == 'smplx':
            smpl_model = smplx16[gender]
        smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).float(), 
                            pose_hand=torch.from_numpy(poses[:, 66:156]).float(), 
                            betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(), 
                            root_orient=torch.from_numpy(poses[:, :3]).float(), 
                            trans=torch.from_numpy(trans).float())
        verts = to_cpu(smplx_output.v)
        faces = smpl_model.f
    
    return verts, faces



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

    return verts, faces


def _np_string(value):
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return str(value.item())
        return str(value.reshape(-1)[0])
    return str(value)


def load_humoto_objects(name, motion_path, object_path, frame_times):
    sequence_path = os.path.join(motion_path, name)
    object_files = sorted(
        filename
        for filename in os.listdir(sequence_path)
        if filename.startswith('object_') and filename.endswith('.npz')
    )
    if not object_files:
        raise FileNotFoundError(
            f"No object_*.npz files found in HUMOTO sequence '{sequence_path}'."
        )

    objects = []
    total_vertices = 0
    total_faces = 0
    for object_file in object_files:
        object_file_path = os.path.join(sequence_path, object_file)
        with np.load(object_file_path, allow_pickle=True) as f:
            if 'angles' not in f.files or 'trans' not in f.files:
                raise KeyError(
                    f"{object_file}: expected 'angles' and 'trans', got {f.files}."
                )

            obj_angles = np.asarray(f['angles'], dtype=np.float32)
            obj_trans = np.asarray(f['trans'], dtype=np.float32)
            if 'mesh_name' in f.files:
                mesh_name = _np_string(f['mesh_name'])
            elif 'name' in f.files:
                mesh_name = _np_string(f['name'])
            else:
                raise KeyError(
                    f"{object_file}: expected 'mesh_name' or 'name', got {f.files}."
                )

            if 'instance_name' in f.files:
                instance_name = _np_string(f['instance_name'])
            else:
                instance_name = os.path.splitext(object_file)[0][len('object_'):]

        expected_shape = (frame_times, 3)
        if obj_angles.shape != expected_shape or obj_trans.shape != expected_shape:
            raise ValueError(
                f"{object_file}: expected angles/trans shape {expected_shape}, "
                f"got {obj_angles.shape}/{obj_trans.shape}."
            )
        if not np.isfinite(obj_angles).all() or not np.isfinite(obj_trans).all():
            raise ValueError(f"{object_file}: angles/trans contain NaN or Inf.")

        mesh_path = os.path.join(object_path, mesh_name, f'{mesh_name}.obj')
        if not os.path.isfile(mesh_path):
            raise FileNotFoundError(
                f"{object_file}: cannot find HUMOTO mesh '{mesh_path}'."
            )
        mesh_obj = trimesh.load(mesh_path, force='mesh')
        local_vertices = np.asarray(mesh_obj.vertices, dtype=np.float32)
        local_faces = np.asarray(mesh_obj.faces, dtype=np.int64)
        if local_vertices.ndim != 2 or local_vertices.shape[1] != 3:
            raise ValueError(
                f"{object_file}: invalid mesh vertices shape {local_vertices.shape}."
            )
        if local_faces.ndim != 2 or local_faces.shape[1] != 3:
            raise ValueError(
                f"{object_file}: invalid mesh faces shape {local_faces.shape}."
            )

        objects.append(
            {
                'instance_name': instance_name,
                'angles': obj_angles,
                'trans': obj_trans,
                'vertices': local_vertices,
                'faces': local_faces,
            }
        )
        total_vertices += local_vertices.shape[0]
        total_faces += local_faces.shape[0]

    merged_vertices = np.empty(
        (frame_times, total_vertices, 3), dtype=np.float32
    )
    merged_faces = np.empty((total_faces, 3), dtype=np.int64)
    vertex_offset = 0
    face_offset = 0
    for obj in objects:
        num_vertices = obj['vertices'].shape[0]
        num_faces = obj['faces'].shape[0]
        angle_matrix = Rotation.from_rotvec(obj['angles']).as_matrix().astype(np.float32)
        merged_vertices[:, vertex_offset:vertex_offset + num_vertices] = (
            np.matmul(
                obj['vertices'][None, ...],
                np.transpose(angle_matrix, (0, 2, 1)),
            )
            + obj['trans'][:, None, :]
        )
        merged_faces[face_offset:face_offset + num_faces] = (
            obj['faces'] + vertex_offset
        )
        vertex_offset += num_vertices
        face_offset += num_faces

    instance_names = [obj['instance_name'] for obj in objects]
    return merged_vertices, merged_faces, instance_names


def debug_visualize_body_obj(
    body_verts,
    body_faces,
    obj_verts,
    obj_faces,
    save_path,
    axis_length=0.5,
    multi_angle=False,
    h=256,
    w=256,
    bg_color='white',
    show_frame=False,
):
    """Render the human, objects, and the canonical world coordinate axes.

    The fixed marker is located at world (0, 0, 0).  It uses the standard
    colors: +X is red, +Y is green, and +Z is blue.  The ground is y=0.
    """
    body_verts = np.asarray(body_verts)
    body_faces = np.asarray(body_faces)
    obj_verts = np.asarray(obj_verts)
    obj_faces = np.asarray(obj_faces)

    if axis_length <= 0:
        raise ValueError(f'axis_length must be positive, got {axis_length}.')
    if body_verts.ndim != 3 or body_verts.shape[-1] != 3:
        raise ValueError(
            f'Expected body vertices with shape (T, V, 3), got {body_verts.shape}.'
        )
    if body_faces.ndim != 2 or body_faces.shape[-1] != 3:
        raise ValueError(
            f'Expected body faces with shape (F, 3), got {body_faces.shape}.'
        )
    if obj_verts.ndim != 3 or obj_verts.shape[-1] != 3:
        raise ValueError(
            f'Expected object vertices with shape (T, V, 3), got {obj_verts.shape}.'
        )
    if obj_faces.ndim != 2 or obj_faces.shape[-1] != 3:
        raise ValueError(
            f'Expected object faces with shape (F, 3), got {obj_faces.shape}.'
        )
    if body_verts.shape[0] != obj_verts.shape[0]:
        raise ValueError(
            'Body and object sequences must have the same frame count, got '
            f'{body_verts.shape[0]} and {obj_verts.shape[0]}.'
        )

    # Keep the same horizontal centering as visualize_body_obj.  The axis mesh
    # receives the same translation, so it still represents the true world
    # origin after the scene is shifted for rendering.
    min_x, _, min_z = body_verts.min(axis=(0, 1))
    max_x, _, max_z = body_verts.max(axis=(0, 1))
    center_x = (min_x + max_x) / 2
    center_z = (min_z + max_z) / 2

    body_verts = body_verts.copy()
    obj_verts = obj_verts.copy()
    body_verts[:, :, 0] -= center_x
    body_verts[:, :, 2] -= center_z
    obj_verts[:, :, 0] -= center_x
    obj_verts[:, :, 2] -= center_z

    axis_transform = np.eye(4)
    axis_transform[0, 3] = -center_x
    axis_transform[2, 3] = -center_z
    axis_mesh = trimesh.creation.axis(
        origin_size=axis_length * 0.06,
        axis_radius=axis_length * 0.02,
        axis_length=axis_length,
        transform=axis_transform,
    )

    viewer = MeshViewer(
        width=w,
        height=h,
        add_ground_plane=True,
        plane_mins=(min_x, max_x, min_z, max_z),
        use_offscreen=True,
        bg_color=bg_color,
    )
    viewer.render_wireframe = False

    object_rgb = np.asarray(colors['pink'][:3], dtype=np.float32) / 255.0
    body_rgb = np.asarray(colors['yellow_pale'][:3], dtype=np.float32) / 255.0
    rotate_y_90 = trimesh.transformations.rotation_matrix(
        math.radians(90), [0, 1, 0]
    )

    video_writer = imageio.get_writer(save_path, fps=30)
    try:
        for frame_idx in range(body_verts.shape[0]):
            object_mesh = trimesh.Trimesh(
                vertices=obj_verts[frame_idx],
                faces=obj_faces,
                vertex_colors=np.tile(object_rgb, (obj_verts.shape[1], 1)),
                process=False,
            )
            body_mesh = trimesh.Trimesh(
                vertices=body_verts[frame_idx],
                faces=body_faces,
                vertex_colors=np.tile(body_rgb, (body_verts.shape[1], 1)),
                process=False,
            )

            object_and_axes = trimesh.util.concatenate(
                [object_mesh, axis_mesh.copy()]
            )
            viewer.set_meshes(
                [object_and_axes, body_mesh],
                group_name='static',
            )
            rendered_views = [viewer.render()]

            if multi_angle:
                rotated_object_and_axes = object_and_axes.copy()
                rotated_body = body_mesh.copy()
                rotated_object_and_axes.apply_transform(rotate_y_90)
                rotated_body.apply_transform(rotate_y_90)
                viewer.set_meshes(
                    [rotated_object_and_axes, rotated_body],
                    group_name='static',
                )
                rendered_views.append(viewer.render())

            frame = np.concatenate(rendered_views, axis=1)
            image = Image.fromarray(frame.astype(np.uint8))
            draw = ImageDraw.Draw(image)
            if show_frame:
                draw.text((5, 5), f'{frame_idx:04d}', fill='red')

            legend_y = max(5, image.height - 15)
            draw.text((5, legend_y), '+X', fill=(255, 0, 0))
            draw.text((23, legend_y), '+Y', fill=(0, 160, 0))
            draw.text((41, legend_y), '+Z', fill=(0, 0, 255))
            draw.text((62, legend_y), 'world axes', fill=(40, 40, 40))
            video_writer.append_data(np.asarray(image, dtype=np.uint8))
    finally:
        video_writer.close()
        del viewer


# visualize surface motion of smpl model
i = 0
for k, name in tqdm(enumerate(data_name)):
    print(name)
    try:
        if dataset == 'GRAB':
            verts, faces = visualize_grab(name, MOTION_PATH)
        elif dataset == 'BEHAVE':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
        elif dataset == 'NEURALDOME' or dataset == 'IMHD':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
        elif dataset == 'CHAIRS':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
        elif dataset == 'INTERCAP':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10, 12)
        elif dataset == 'OMOMO':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 16)
        elif dataset == 'HUMOTO_TEST':
            verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 10)

        if dataset != 'HUMOTO_TEST':
            with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

            mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
            obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

            angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
            obj_verts = (obj_verts)[None, ...]
            obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
            rend_video_path = os.path.join(results_folder, '{}_{}_{}.mp4'.format(dataset, name, obj_name))
            visualize_body_obj(verts, faces, obj_verts, obj_faces, save_path=rend_video_path, show_frame=True, multi_angle=True)
        elif dataset == 'HUMOTO_TEST':
            obj_verts, obj_faces, instance_names = load_humoto_objects(
                name,
                MOTION_PATH,
                OBJECT_PATH,
                verts.shape[0],
            )
            print(
                f"HUMOTO objects ({len(instance_names)}): "
                + ', '.join(instance_names)
            )
            rend_video_path = os.path.join(
                results_folder,
                '0808_{}_{}_all_objects_debug_axes.mp4'.format(dataset, name),
            )
            print(f"Saving HUMOTO debug video to '{rend_video_path}'.")
            debug_visualize_body_obj(
                verts,
                faces,
                obj_verts,
                obj_faces,
                save_path=rend_video_path,
                axis_length=0.5,
                show_frame=True,
                multi_angle=True,
                h=512,
                w=512
            )
            i += 1
            if i == 2:
                exit()
    except Exception as e:
        print(e)
        continue
