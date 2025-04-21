import os
import os.path
import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation
import pickle
import trimesh
import torch

RAW_PATH = './data/intercap/raw'
MOTION_PATH = './data/intercap/sequences'
OBJECT_PATH = './data/intercap/objects'
MODEL_PATH = './models'

# ### Objects
# * 01 suitcase
# * 02 skateboard
# * 03 soccerball
# * 04 umbrella
# * 05 racket
# * 06 toolbox
# * 07 chair
# * 08 fantabottle
# * 09 cup
# * 10 stool

# * male: 01 02 04 05 10 
# * female: 03 06 07 08 09

obj_dic = {
    '01': 'suitcase',
    '02': 'skateboard',
    '03': 'soccerball',
    '04': 'umbrella',
    '05': 'racket',
    '06': 'toolbox',
    '07': 'chair',
    '08': 'fantabottle',
    '09': 'cup',
    '10': 'stool'
}

gender_dic = {
    '01': 'male',
    '02': 'male',
    '03': 'female',
    '04': 'male',
    '05': 'male',
    '06': 'female',
    '07': 'female',
    '08': 'female',
    '09': 'female',
    '10': 'male'
}


# Arguments
# Sub_id = '10'
# gender = 'female'
# Obj_id = '05'
# obj_name = 'racket'

smpl_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="male",
                          num_pca_comps=12,
                          ext='pkl')

smpl_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                          gender="female",
                          num_pca_comps=12,
                         
                          ext='pkl')


smpl_models = {
    'male': smpl_model_male,
    'female': smpl_model_female
}

def process(human, obj):
    poses, betas, trans, gender = human['poses'], human['betas'], human['trans'], str(human['gender'])
    obj_angles, obj_trans, obj_name = obj['angles'], obj['trans'], str(obj['name'])
    frame_times = poses.shape[0]
    smpl_model = smpl_models[gender]
    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:78]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 78:90]).float(),
                              jaw_pose=torch.zeros(frame_times, 3).float(),
                              leye_pose=torch.zeros(frame_times, 3).float(),
                              reye_pose=torch.zeros(frame_times, 3).float(),
                              expression=torch.zeros(frame_times, 10).float(),
                              betas=torch.from_numpy(betas).float(),
                              transl=torch.from_numpy(trans).float(),)
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
    rotvecs = poses[:, :3]
    rotations = Rotation.from_rotvec(rotvecs)
    # camera extrinsic
    rotation_matrix = Rotation.from_euler('z', np.pi, degrees=False)
    # Apply the rotation to the batch of rotations
    rotated_rotations = rotation_matrix * rotations
    # Convert the rotated rotations back to rotation vectors
    poses[:, :3] = rotated_rotations.as_rotvec()

    trans = rotation_matrix.apply(trans)

    rotations2 = Rotation.from_rotvec(obj_angles)

    # Apply the rotation to the batch of rotations
    rotated_rotations2 = rotation_matrix * rotations2
    # Convert the rotated rotations back to rotation vectors
    obj_angles = rotated_rotations2.as_rotvec()
    obj_trans_delta = rotation_matrix.apply(obj_trans - pelvis)
    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                              global_orient=torch.from_numpy(poses[:, :3]).float(),
                              left_hand_pose=torch.from_numpy(poses[:, 66:78]).float(),
                              right_hand_pose=torch.from_numpy(poses[:, 78:90]).float(),
                              jaw_pose=torch.zeros(frame_times, 3).float(),
                              leye_pose=torch.zeros(frame_times, 3).float(),
                              reye_pose=torch.zeros(frame_times, 3).float(),
                              expression=torch.zeros(frame_times, 10).float(),
                              betas=torch.from_numpy(betas).float(),
                              transl=torch.from_numpy(trans).float(),)
    
    verts = smplx_output.vertices.detach().numpy()
    pelvis = smplx_output.joints.detach().numpy()[:, 0, :]
    faces = smpl_model.faces
    
    obj_trans = pelvis + obj_trans_delta
    
    mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
    obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = mesh_obj.vertices[None, ...]
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    diff = min(verts[:, :, 1].min(), obj_verts[:, :, 1].min())
    obj_trans[..., 1] -= diff
    trans[..., 1] -= diff
    

    obj = {
        'angles': np.array(obj_angles),
        'trans': np.array(obj_trans),
        'name': obj_name,
    }
    human = {
        'poses': np.array(poses),
        'betas': np.array(betas[0]),
        'trans': np.array(trans),
        'gender': gender,
    }
    return human, obj




# process sequences
for Sub_id in ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']:
    Sub_motion_path = os.path.join(RAW_PATH, Sub_id)
    dirs = os.listdir(Sub_motion_path)
    for Obj_id in dirs:
        Obj_motion_path = os.path.join(Sub_motion_path, Obj_id)
        segs = os.listdir(Obj_motion_path)
        for Seg_id in segs:
            Seg_motion_path = os.path.join(Obj_motion_path, Seg_id)
            print('processing ' + Seg_motion_path)
            if os.path.exists(Seg_motion_path+'/res.pkl'):
                with open(Seg_motion_path+'/res.pkl', 'rb') as f:
                    human = pickle.load(f, encoding='utf8')
                body_pose = human['body_pose']
                global_orient = human['global_orient']
                left_hand_pose = human['left_hand_pose']
                right_hand_pose = human['right_hand_pose']
                body_pose = body_pose.reshape(-1, 63)
                poses = np.concatenate([global_orient, body_pose, left_hand_pose, right_hand_pose], axis=1)
                betas = human['betas']
                trans = human['transl']
                obj_angles = human['ob_pose']
                obj_trans = human['ob_trans']

                # process object
                Obj_path = os.path.join(OBJECT_PATH, obj_dic[Obj_id])
                if not os.path.exists(os.path.join(Obj_path, obj_dic[Obj_id]+'.obj')):
                    mesh_path = os.path.join(Seg_motion_path, 'Mesh', '00000_second_obj.ply')
                    mesh = trimesh.load(mesh_path, force='mesh') 
                    mesh.vertices -= obj_trans[0]
                    mesh.vertices = mesh.vertices @ Rotation.from_rotvec(-obj_angles[0]).as_matrix().T
                    os.makedirs(Obj_path, exist_ok=True)
                    mesh.export(os.path.join(Obj_path,obj_dic[Obj_id]+'.obj'))
                obj = {
                    'angles': np.array(obj_angles),
                    'trans': np.array(obj_trans),
                    'name': obj_dic[Obj_id],
                }
                human = {
                    'poses': np.array(poses),
                    'betas': np.array(betas),
                    'trans': np.array(trans),
                    'gender': gender_dic[Sub_id],
                }
                name = 'Sub'+ Sub_id + '_Object' + Obj_id + '_' + Seg_id + '_' + obj_dic[Obj_id]

                human, obj = process(human, obj)

                os.makedirs(os.path.join(MOTION_PATH, name), exist_ok=True)
                np.savez(os.path.join(MOTION_PATH, name,'object.npz'), **obj)
                np.savez(os.path.join(MOTION_PATH, name, 'human.npz'), **human)
                print('save ' + os.path.join(MOTION_PATH, name))
        
       