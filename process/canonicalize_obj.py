import os
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    datasets = ['behave', 'intercap', 'grab', 'omomo']
    data_root = './data'
    for dataset in datasets:
        print("Processing dataset:", dataset)
        obj_dic = {}
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences')
        OBJECT_PATH = os.path.join(dataset_path, 'objects')
        obj_names = os.listdir(OBJECT_PATH)
        for name in obj_names:
            print("Processing object:", name)
            mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{name}/{name}.obj"), force='mesh')
            obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
            obj_dic[name] = obj_verts.mean(axis=0)
            mesh_obj.vertices = (obj_verts - obj_verts.mean(axis=0, keepdims=True))
            os.makedirs(os.path.join(OBJECT_PATH, name), exist_ok=True)
            mesh_obj.export(os.path.join(OBJECT_PATH, f"{name}/{name}.obj"))
            
        data_name = os.listdir(MOTION_PATH)
        for name in data_name:
            print("Processing sequence:", name)
            if not os.path.exists(os.path.join(MOTION_PATH, name, 'object.npz')):
                continue
            with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])

            rotation = Rotation.from_rotvec(obj_angles)

            
            new_obj_trans = obj_trans + rotation.apply(obj_dic[obj_name])
            
            
            obj = {
                'angles': obj_angles,
                'trans': new_obj_trans,
                'name': obj_name,
            }
            
            np.savez(os.path.join(MOTION_PATH, name, 'object.npz'), **obj)
            


        