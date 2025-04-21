import trimesh
import os
import numpy as np

datasets = ['behave', 'intercap', 'grab', 'omomo']
data_root = './data'
id_root = './assets/sample_objids'
for dataset in datasets:
    dataset_path = os.path.join(data_root, dataset)
    OBJECT_PATH = os.path.join(dataset_path, 'objects')
    object_name = os.listdir(OBJECT_PATH)
    for obj_name in object_name:
        mesh_obj = trimesh.load(os.path.join(OBJECT_PATH, f"{obj_name}/{obj_name}.obj"), force='mesh')
        obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces
        sample_ids = np.load(os.path.join(id_root,dataset,f'{obj_name}.npy'))
        obj_points = obj_verts[sample_ids]
        np.save(os.path.join(OBJECT_PATH, f"{obj_name}/sample_points.npy"), obj_points)
   
        

