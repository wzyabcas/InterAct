import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import trimesh
from bps_torch.bps import bps_torch


# visualize markers motion of smpl model
if __name__ == "__main__":
    # bps 
    bps_torch = bps_torch()
    bps_obj = np.load('assets/bps_basis_set_1024_1.npy')
    bps_obj = torch.from_numpy(bps_obj).float().cuda()
   
    datasets = ['behave', 'intercap', 'grab', 'omomo',]
    data_root = 'data'
    for dataset in datasets:
        print(f'Loading {dataset} ...')
        frame_num = 0
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences_canonical')
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        OBJECT_BPS_PATH = os.path.join(dataset_path, 'objects_bps')
        
        os.makedirs(OBJECT_BPS_PATH, exist_ok=True)  # create folder if not exist
        data_name = os.listdir(OBJECT_PATH)
        for k, name in tqdm(enumerate(data_name)):
            mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{name}/{name}.obj"), force='mesh')
            obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
            obj_verts = (obj_verts)[None, ...]
            torch_obj_verts = torch.from_numpy(obj_verts).float().cuda()
            
            bps_object_geo = bps_torch.encode(x=torch_obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=bps_obj[None,...])['deltas'] # T X N X 3 
            bps_object_geo_np = bps_object_geo.data.detach().cpu().numpy()
            
            np.save(os.path.join(OBJECT_BPS_PATH, f"{name}/{name}_1024.npy"), bps_object_geo_np)
            
            
            

        
        