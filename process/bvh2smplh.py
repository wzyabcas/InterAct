import torch
import open3d as o3d
import numpy as np
import trimesh
import os
import argparse

from pytorch3d.transforms import matrix_to_axis_angle
from scipy.spatial.transform import Rotation as R

from text2interaction.sample.marker2smpl import SmplhOptmize10_fulljoints_mixamo
from text2interaction.render.mesh_viz import visualize_body_objs
# from text2interaction.utils.rotation_helper import *

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

PATH_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
mixamo_to_smplh = {
    # Root and spine
    "Hips": 0,                 # pelvis
    "Spine": 3,                # spine1
    "Spine1": 6,               # spine2
    "Spine2": 9,               # spine3
    "Neck": 12,                # neck
    "Head": 15,                # head

    # Left leg
    "LeftUpLeg": 1,            # left_hip
    "LeftLeg": 4,              # left_knee
    "LeftFoot": 7,             # left_ankle
    "LeftToeBase": 10,         # left_foot

    # Right leg
    "RightUpLeg": 2,           # right_hip
    "RightLeg": 5,             # right_knee
    "RightFoot": 8,            # right_ankle
    "RightToeBase": 11,        # right_foot

    # Left arm
    "LeftShoulder": 13,        # left_collar
    "LeftArm": 16,             # left_shoulder
    "LeftForeArm": 18,         # left_elbow
    "LeftHand": 20,            # left_wrist

    # Right arm
    "RightShoulder": 14,       # right_collar
    "RightArm": 17,            # right_shoulder
    "RightForeArm": 19,        # right_elbow
    "RightHand": 21,           # right_wrist

    # Left hand fingers (SMPL-H / MANO order)
    "LeftHandIndex1": 22,
    "LeftHandIndex2": 23,
    "LeftHandIndex3": 24,
    "LeftHandMiddle1": 25,
    "LeftHandMiddle2": 26,
    "LeftHandMiddle3": 27,
    "LeftHandPinky1": 28,
    "LeftHandPinky2": 29,
    "LeftHandPinky3": 30,
    "LeftHandRing1": 31,
    "LeftHandRing2": 32,
    "LeftHandRing3": 33,
    "LeftHandThumb1": 34,
    "LeftHandThumb2": 35,
    "LeftHandThumb3": 36,

    # Right hand fingers
    "RightHandIndex1": 37,
    "RightHandIndex2": 38,
    "RightHandIndex3": 39,
    "RightHandMiddle1": 40,
    "RightHandMiddle2": 41,
    "RightHandMiddle3": 42,
    "RightHandPinky1": 43,
    "RightHandPinky2": 44,
    "RightHandPinky3": 45,
    "RightHandRing1": 46,
    "RightHandRing2": 47,
    "RightHandRing3": 48,
    "RightHandThumb1": 49,
    "RightHandThumb2": 50,
    "RightHandThumb3": 51,
}

# rename the keys to include the "mixamorig:" prefix
mapping ={}
for key,value in mixamo_to_smplh.items():
    mapping[f'mixamorig:{key}'] =value


def numpy_to_pd(points,path):
    # 1000 points in 3D
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # (Optional) Add color if you have RGB values
    # colors = np.random.rand(1000, 3)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    # o3d.visualization.draw_geometries([pcd])

    # Save to file
    o3d.io.write_point_cloud(path, pcd)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="示例：解析 --number 参数")

    # d= dict(np.load('/projects/bbsg/ziyin/humoto/human_model/human_pos.npz',allow_pickle=True))
    # print(len(list(d.keys())))
    
    # T=0
    # for key,value in d.items():
    #     T = value.shape[0]
    #     break
    # A = np.zeros((52,3))
    # # gender = 'female'
    # # model=SmplhOptmize10_fulljoints_mixamo(gender, 1, T,extra=[],joint_nums=52)
    # for key,value in mapping.items():
    #     A[value] = d[key]
    # np.save('/projects/bbsg/ziyin/humoto/human_model/human_pos2smplh.npy',A)
    
    # A =np.load('/projects/bbsg/ziyin/humoto/human_model/human_pos2smplh.npy').reshape(1,-1,3)
    # gender = 'female'
    # model=SmplhOptmize10_fulljoints(gender, 1, 1,extra=[],joint_nums=52)
    # V,F,poses,betas,trans = model(torch.from_numpy(A).float().cuda())
    # print(betas.shape)
    # np.save('/projects/bbsg/ziyin/humoto/human_model/human_betas.npy',betas.reshape(-1))
    
    # visualize_body(V.detach().cpu().numpy(),F,save_path =f'./zm2smplh/at.mp4')
    
    betas = np.load(os.path.join(PATH_PROJECT, "data", "human_betas.npy"))

    # 添加 --number 参数，类型为 int，可以通过 required=True 强制用户必须传入
    parser.add_argument('--number', type=int, required=True, help="输入一个整数")
    args = parser.parse_args()
    
    path_sequence_name = os.path.join(PATH_PROJECT, "data", "output_process")
    list_sequence_name = sorted(os.listdir(path_sequence_name))
    sequence_length = len(list_sequence_name)
    
    chunk_size = sequence_length // 10 + 1
    start_index = args.number * chunk_size
    end_index = min(start_index + chunk_size, sequence_length)
    list_sequence_name = list_sequence_name[start_index:end_index]
    
    for sequence_name in list_sequence_name:
        npz_p = os.path.join(PATH_PROJECT, 'data', 'smplh', sequence_name+'.npz')    # smplh -> smplh_test
        if os.path.isfile(npz_p):
            continue
        # sequence_name ='activating_floor_lamp_with_right_hand-485'
        
        path_joint_pos = os.path.join(PATH_PROJECT, "data", "output_process", sequence_name, "human_joints_mixamo.pt")
        path_joint_transform = os.path.join(PATH_PROJECT, "data", "output_process", sequence_name, "human_pose_params_matrix.pt")
        path_obj_pose = os.path.join(PATH_PROJECT, "data", "output_process", sequence_name, "obj_pose.npz")
        
        dict_joint_pos=torch.load(path_joint_pos)
        joint_transform = torch.load(path_joint_transform)
        dict_obj_pose =  dict(np.load(path_obj_pose,allow_pickle=True))
        # print(data_obj['floor_lamp'].shape)
       

        obj_vs =[]
        obj_fs =[]
        bp=os.path.join(PATH_PROJECT, "data",'humoto_objects_0805')
        for obj_name,object_pose_params in dict_obj_pose.items():
            mesh_name = obj_name.split(".", 1)[0]
            MESH = trimesh.load(
                os.path.join(bp, mesh_name, f"{mesh_name}.obj"),
                process=False,
                force="mesh",
            )
            obj_v = np.array(MESH.vertices)[None,:]
            obj_fs.append(MESH.faces)
            # object_pose_rot_matrix = quaternion_to_matrix(object_pose_params[..., :4])
            object_pose_rot_matrix = R.from_quat(object_pose_params[:, [1,2,3,0]]).as_matrix()
            
            # if 'mesh' in object_model:
            #    
            #     v, f = object_model['mesh']
            #     if len(v.shape) == 2:
            #         v = v.unsqueeze(0)
            #     if v.shape[0] != object_pose_params.shape[0]:
            #         v = v.repeat(object_pose_params.shape[0], 1, 1)

            obj_v = np.matmul(obj_v, np.transpose(object_pose_rot_matrix, (0, 2, 1))) + object_pose_params[..., 4:][:, None,:]
            # print(obj_v.shape)
            obj_vs.append(obj_v)

        T=0
        for key,value in dict_joint_pos.items():
            T = value.shape[0]
        A = torch.zeros((T,52,3)).cuda()
        rotA = torch.zeros((T,52,3)).cuda()
        gender = 'female'
        
        for key,value in mapping.items():
            if 'Hips' in key:
                transA = joint_transform[key].cuda()[:,:3,3]
            A[:,value] = dict_joint_pos[key].cuda()
            # print(quaternion_to_matrix(joint_transform[key].cuda()).shape,'AA')
            rotA[:,value] = matrix_to_axis_angle((joint_transform[key][:,:3,:3].cuda()).float()).float()

            
            
        # line 984 set requires_grad to False for left_hand_pose, which may be a bug.
        model = SmplhOptmize10_fulljoints_mixamo(
            gender=gender,
            batch_size=1,
            frame_times=T,
            extra=[],joint_nums=52,betas=betas,
            init_pose=rotA,
            init_trans=transA
        )
        V,F,poses,betas,trans = model(A)
        ODCT={}
        ODCT['poses'] = poses
        ODCT['betas'] = betas
        ODCT['trans'] = trans
        ODCT['gender'] = np.array(gender)
        
        # np.savez(os.path.join('/projects/bbsg/ziyin/HUMOTO/smplh',seq_id+'.npz'),**ODCT)
        np.savez(os.path.join(PATH_PROJECT, "data", 'smplh', sequence_name+'.npz'),**ODCT)
        
        
        # ODCT={}
        # ODCT['poses']
        # print(poses.shape,betas.shape,trans.shape)

        vis_path = os.path.join(PATH_PROJECT, "results", "bvh2smplh", f"{sequence_name}.mp4")
        visualize_body_objs(V.detach().cpu().numpy(),F,obj_vs,obj_fs,save_path=vis_path)
        print(f"Processed sequence: {sequence_name}, saved to {vis_path}")
    # key = key.replace(':','_')
    # numpy_to_pd(value[0:1].detach().cpu().numpy(),f'./pcds/{key}.ply')
    # print(value.shape)


# keys=np.array(list(d.keys()))
# idx_mapping =[0,55,60,1,56,61,2,57,62,3,59,64,4,3,3,5,7,31,8,32,9,33,10,34]
# print(keys[idx_mapping])

# for key,value in d.items():
#     key = key.replace(':','_')
#     numpy_to_pd(value[0:1].detach().cpu().numpy(),f'./pcds/{key}.ply')
    # print(value.shape)

# print(keys[33])


# 'mixamorig:Hips', 'mixamorig:Spine', 'mixamorig:Spine1', 'mixamorig:Spine2', 'mixamorig:Neck', 'mixamorig:Head', 'mixamorig:HeadTop_End', 'mixamorig:LeftShoulder', 'mixamorig:LeftArm', 'mixamorig:LeftForeArm', 'mixamorig:LeftHand'