import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from copy import copy
from pytorch3d.transforms import *
import codecs as cs
from bps_torch.bps import bps_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()


MODEL_PATH = './models'

######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=True)

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=True)

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=True)

smplh10 = {'male': smplh_model_male, 'female': smplh_model_female, 'neutral': smplh_model_neutral}


######################################## Visualize SMPL ########################################
def visualize_smpl(name, MOTION_PATH, dataset):
    if dataset =='grab':
        with np.load(os.path.join(MOTION_PATH.replace('grab','grab_smplh'), name, 'human.npz'), allow_pickle=True) as f:
            global_pose, body_pose, lhand_pose, rhand_pose, betas, trans, gender = f['global_orient'],f['body_pose'], f['left_hand_pose'], f['right_hand_pose'], f['betas'], f['transl'], str(f['gender'])
        betas = betas[0]
    else:
        with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
            poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
        global_pose,body_pose,lhand_pose,rhand_pose = poses[:,:3],poses[:,3:66],poses[:,66:111],poses[:,111:156]
    frame_times = global_pose.shape[0]
    
    smpl_model = smplh10[gender]
    smplx_output = smpl_model(body_pose=torch.from_numpy(body_pose).float(),
        global_orient=torch.from_numpy(global_pose).float(),
        left_hand_pose=torch.from_numpy(lhand_pose).float(),
        right_hand_pose=torch.from_numpy(rhand_pose).float(),
        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
        transl=torch.from_numpy(trans).float(),) 
        # global_pose = poses[:,:3]
        # body_pose = poses[:,3:66]
        # lhand_pose = poses[:,66:111]
        # rhand_pose = poses[:,111:156]
    hand_rot_scalar = np.linalg.norm(np.concatenate([lhand_pose,rhand_pose],-1).reshape(-1,30,3),axis=-1)
    joints =smplx_output.joints.detach().cpu().numpy()
    
    return joints[:,:52],betas,hand_rot_scalar,frame_times,gender

######################################## utils for GRAB ########################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

    points = points.reshape(-1,3)
    n_points = points.shape[0]

    spheres = []
    for p in range(n_points):
        sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
        sphs.apply_translation(points[p])
        sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

        spheres.append(sphs)

    spheres = Mesh.concatenate_meshes(spheres)
    return spheres

class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 process = False,
                 visual = None,
                 wireframe=False,
                 smooth = False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process = process)
            vertices = mesh.vertices
            faces= mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices*vscale

        if faces is None:
            mesh = points2sphere(vertices)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rot_verts(self, vertices, rxyz):
        return np.array(vertices * rxyz.T)

    def colors_like(self,color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self,vc, vertex_ids = None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self,fc, face_ids = None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)
def DotDict(in_dict):
    
    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}


######################################## Visualize GRAB ########################################







def matrix_to_rotation_6d_np(mat):
    """
    Convert rotation matrices to continuous 6D representations
    :param mat: Rotation matrices of shape [B, T, 3, 3]
    :return: Continuous 6D representations of shape [B, T, 6]
    """
    batch_dim = mat.shape[:-2]
    return mat[..., :2, :].reshape(batch_dim + (6,))



def rotation_6d_to_matrix_np(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of shape [B, T, 6]

    Returns:
        Rotation matrices of shape [B, T, 3, 3]

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    
    d6 = torch.from_numpy(d6)

    return rotation_6d_to_matrix(d6).numpy()

def relative_representation_np(markers, rotation_matrix, obj_rot6D, obj_trans):
    # Input: markers: BxTx77x3, rotation_matrix: BxTx77x3x3, obj_rot6D: BxTx6, obj_trans: BxTx3
    # return: obj_rot6D: BxTx77x6, obj_trans: BxTx77x3

    # apply the rotation matrix to the normal vector
    trans_delta = obj_trans[...,None,:] - markers
    rel_obj_trans = np.matmul(rotation_matrix, trans_delta[...,None]).squeeze(-1)
    obj_matrix = rotation_6d_to_matrix_np(obj_rot6D)
    rel_obj_matrix = np.matmul(rotation_matrix, obj_matrix[...,None,:,:])
    rel_obj_rot6D = matrix_to_rotation_6d_np(rel_obj_matrix)

    return rel_obj_rot6D, rel_obj_trans

gender_dict ={'male':0,'female':1,'neutral':2}
# visualize markers motion of smpl model
if __name__ == "__main__":
    
    bps_func = bps_torch()
    bps_obj = np.load('assets/bps_basis_set_1024_1.npy')
    bps_obj = torch.from_numpy(bps_obj).float().cuda()
    
    datasets = ['behave_correct', 'grab',]
    data_root = './data'
    for dataset in datasets:
        print(f'Loading {dataset} ...')
        frame_num = 0
        dataset_path = os.path.join(data_root, dataset)
        dataset_path_raw = os.path.join(data_root, dataset.split('_')[0])
        MOTION_PATH = os.path.join(dataset_path, 'sequences_canonical')
        MOTION_PATH_raw = os.path.join(dataset_path_raw, 'sequences_canonical')
        
        OBJECT_PATH = os.path.join(data_root, dataset.split('_')[0], 'objects')
        OBJECT_BPS_PATH =  os.path.join(data_root, dataset.split('_')[0], 'objects_bps')
        data_name = os.listdir(MOTION_PATH)
        for k, name in tqdm(enumerate(data_name)):
            
            ## SMPL-H FK
            joints,betas,hand_rot_scalar,L,gender = visualize_smpl(name, MOTION_PATH, dataset)
            if L<60 or L >= 400:
                continue
             
            
            ## object poses
            with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
            angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
            obj_rot6D = matrix_to_rotation_6d_np(angle_matrix)            

            obj_data = np.concatenate([obj_rot6D, obj_trans], axis=-1)
            obj_points = np.load(os.path.join(OBJECT_PATH, obj_name, 'sample_points.npy'))
            
        representation = np.concatenate([joints.reshape(-1,52*3),hand_rot_scalar,obj_rot6D.reshape(-1,6),obj_trans.reshape(-1,3)],-1)
        
        obj_sample_path = os.path.join(OBJECT_PATH, obj_name, 'sample_points.npy')
        
        obj_points = np.load(obj_sample_path)
        
        
        angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
        angle_matrix = angle_matrix[:L]
        obj_trans = obj_trans[:L]
        rotated = np.matmul(obj_points, angle_matrix.transpose(0,2,1)) + obj_trans[:,None]
        
        ## relative distance, contact labels, contact corresponding ids
        bps_dct = bps_func.encode(torch.from_numpy(rotated).float().cuda(),feature_type=['deltas'],custom_basis=torch.from_numpy(joints).float().cuda())
        
        object_cids = bps_dct['ids'].detach().cpu().numpy()
        delta = bps_dct['deltas'].detach().cpu().numpy()
        
        dist = np.exp(-5*np.linalg.norm(delta,axis=-1))
        contact_ids = (np.linalg.norm(delta,axis=-1)<0.03).astype(np.float32)
        
        
        ## object BPS and nomralized BPS w/ scale
        
        M = trimesh.load(os.path.join(OBJECT_PATH,obj_name,obj_name+'.obj'),force='mesh')
        obj_vertices_static = torch.from_numpy(M.vertices).float()
        margin = 0.05
        s = torch.norm(obj_vertices_static,p=2,dim=1)
        max_norm, max_idx = torch.max(s, dim=0)
        scale = max_norm/(1-margin)
        obj_vertices_static = obj_vertices_static/scale
        
        bps_object_geo = bps_func.encode(x=obj_vertices_static, \
                feature_type=['deltas'], \
                custom_basis=bps_obj[None,...])['deltas'] # T X N X 3 
        
        
        
        bps_object_geo_np = bps_object_geo.data.detach().cpu().numpy().reshape(-1)
        bps_object_geo_np_wscale = np.concatenate([bps_object_geo_np,scale.detach().cpu().numpy().reshape(-1)],-1)
        
        obj_bps = np.load(os.path.join(OBJECT_BPS_PATH, obj_name, f'{obj_name}_1024.npy'))
        
        # seq name / beta / gender / text
        
        seq_name = dataset+'_'+obj_name
        
        gender_onehot = np.zeros((3),dtype=np.float32)
        gender_onehot[gender_dict[gender]] = 1
        
        beta_f = np.concatenate([betas,gender_onehot],axis=-1)
        
        text_data = []
        with cs.open(os.path.join(MOTION_PATH_raw, name, 'text.txt')) as f:
                        
            for line in f.readlines():
                text_dict = {}
                line_split = line.strip().split('#')
                caption = line_split[0]
                tokens = line_split[1].split(' ')
                f_tag = 0.0
                to_tag = 0.0
                

                text_dict['caption'] = caption
                
                text_data.append(text_dict)
        
        dct = {}
        
        L = min(L,300)
        dct['motion'] = representation[:L]
        dct['length'] = L
        dct['text'] = text_data
        dct['seq_name'] = seq_name
        dct['beta'] = beta_f
        dct['obj_points'] = obj_points
        dct['object_cids'] = object_cids[:L]
        dct['contact_ids'] = contact_ids[:L]
        dct['dist'] = dist[:L]
        dct['obj_bps'] = obj_bps
        dct['bps_normalize'] = bps_object_geo_np_wscale
        
        np.savez(os.path.join(MOTION_PATH_raw, name, 'data.npz'), **dct)
        
        
        
        


    
