import os
import os.path
import numpy as np
import torch
from tqdm import tqdm
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from copy import copy
from process.markerset import *
from human_body_prior.body_model.body_model import BodyModel
from pytorch3d.transforms import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()


MODEL_PATH = './models'

######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl')

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl')

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl')

smplh10 = {'male': smplh_model_male, 'female': smplh_model_female, 'neutral': smplh_model_neutral}
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
def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False):
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
            smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
                                global_orient=torch.from_numpy(poses[:, :3]).float(),
                                left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
                                right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
                                betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
                                transl=torch.from_numpy(trans).float(),) 
        elif model_type == 'smplx':
            if use_pca:
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
        faces = smpl_model.faces.astype(np.int32)
        faces = torch.from_numpy(faces).unsqueeze(0).repeat(frame_times, 1, 1)
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
        faces = smpl_model.f.unsqueeze(0).repeat(frame_times, 1, 1)
    
    return verts, faces

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

sbj_info = {}
def load_sbj_verts(sbj_id, seq_data, data_root_folder = '../data/grab/'):
    
    mesh_path = os.path.join(data_root_folder,seq_data.body.vtemp)
    if sbj_id in sbj_info:
        sbj_vtemp = sbj_info[sbj_id]
    else:
        sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
        sbj_info[sbj_id] = sbj_vtemp
    return sbj_vtemp
######################################## Visualize GRAB ########################################
def visualize_grab(name, MOTION_PATH):
    """
    vertices: (N, 10475, 3)
    Modified to work with processed GRAB data format
    """
    with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
        poses, vtemp, trans, gender = f['poses'], f['vtemp'], f['trans'], str(f['gender'])

    n_comps = 24  # GRAB uses 24 PCA components for hands
    T = len(poses)

    smpl_model = smplx.create(
        model_path=MODEL_PATH,
        model_type='smplx',
        gender=gender,
        num_pca_comps=n_comps,
        v_template=vtemp,
        batch_size=T).cuda()

    smplx_output = smpl_model(
        body_pose=torch.from_numpy(poses[:, 3:66]).float().cuda(),
        global_orient=torch.from_numpy(poses[:, :3]).float().cuda(),
        left_hand_pose=torch.from_numpy(poses[:, 66:90]).float().cuda(),
        right_hand_pose=torch.from_numpy(poses[:, 90:114]).float().cuda(),
        transl=torch.from_numpy(trans).float().cuda(),
    )
    verts = to_cpu(smplx_output.vertices)
    faces = smpl_model.faces.astype(np.int32)

    return verts, torch.from_numpy(faces).unsqueeze(0).repeat(T,1,1)

def get_representation_canonical(positions, positions_obj, obj_points):

    # Right/Left foot
    fid_r, fid_l = [61, 52, 53, 40, 34, 49, 40], [29, 30, 18, 19, 7, 2, 15]

    obj_angles = positions_obj[:,:6]
    obj_trans = positions_obj[:,6:9]
    ground_marker = obj_trans[:,None,:].copy() # T x 1 x 3
    ground_marker[...,1] = 0
    angle_matrix = rotation_6d_to_matrix_np(obj_angles)
    obj_verts = (obj_points)[None, ...]
    freeze_obj_verts = (obj_verts)[None, ...] # 1x1xMx3
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

    velocity = (positions[1: ] - positions[:-1]).copy()
    velocity_obj = (positions_obj[1: ] - positions_obj[:-1]).copy()
    # pad the last velocity using torch
    velocity = np.concatenate([velocity, np.zeros(velocity.shape[1:])[None]], axis=0)
    velocity_obj = np.concatenate([velocity_obj, np.zeros(velocity_obj.shape[1:])[None]], axis=0)


    def interaction_aware(distance, omega=5.0):
        return np.exp(-omega * distance)
    """ Get Foot Contacts """


    feet_l = positions[:,fid_l,1]
    # feet_l = interaction_aware(feet_l_h) 

    feet_r = positions[:,fid_r,1]
    # feet_r = interaction_aware(feet_r_h) 


    def interaction_aware_torch(distance, omega=5.0):
        return torch.exp(-omega * distance)

    def contact_detect_torch(verts, obj_points, freeze_obj_points):
        # verts: T x N+1 X 3 
        # obj_points: T x M x 3
        # return T x N+1
        verts = torch.tensor(verts).cuda()
        obj_points = torch.tensor(obj_points).cuda()
        freeze_obj_points = torch.tensor(freeze_obj_points).cuda()
        
        contact_vector = obj_points[:, None, :, :] - verts[:, :, None, :]  # T x N+1 x M x 3
        contact = torch.norm(contact_vector, dim=-1) # T x N+1 x M 
        _, contact_label = torch.min(contact, dim=-1, keepdim=True) # T x N+1 x 1

        expanded_contact_label = contact_label.unsqueeze(-1).expand(-1, -1, -1, 3) # T x N+1 x M x 3
        freeze_obj_verts = freeze_obj_points.expand(contact_label.size(0), contact_label.size(1), -1, -1)  #  T x (N+1) x M x 3
        
        selected_contact_vector = torch.gather(contact_vector, 2, expanded_contact_label).squeeze() # T x (N+1) x 1 x 3
        selected_can_obj_verts = torch.gather(freeze_obj_verts, 2, expanded_contact_label).squeeze() # T x (N+1) x 1 x 3

        contact_data = torch.cat([selected_contact_vector,selected_can_obj_verts],dim=-1)

        
        # print(contact[0])
        # contact = interaction_aware_torch(contact)
        return to_cpu(contact_data).numpy()

    
    contact_data = contact_detect_torch(np.concatenate([positions, ground_marker], axis=1), obj_verts, freeze_obj_verts)

    
    velocity = velocity.reshape(len(velocity), -1)
    # plot_markers("./results/{}_markes.mp4".format(int(x_offset*100)),positions,obj_verts)
    positions = positions.reshape(len(positions), -1)
    contact_data = contact_data.reshape(len(positions), -1)

    data = np.concatenate([positions, velocity, feet_l, feet_r, positions_obj, velocity_obj, contact_data], axis=-1)
    return data


def get_representation(positions, positions_obj, feet_thre, obj_points):

    minx, _, miny = positions.min(axis=(0, 1))
    maxx, _, maxy = positions.max(axis=(0, 1))

    x_offset = (minx+maxx)/2
    y_offset = (miny+maxy)/2

    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    positions_obj[:, 4] -= floor_height
    positions[:, :, 0] -= x_offset
    positions_obj[:, 3] -= x_offset
    positions[:, :, 2] -= y_offset
    positions_obj[:, 5] -= y_offset

    obj_angles = positions_obj[:,:3]
    obj_trans = positions_obj[:,3:]
    angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
    obj_verts = (obj_points)[None, ...]
    obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]
    # plot_markers("./results/{}_markes.mp4".format(int(x_offset*100)),positions,obj_verts)
    velocity = (positions[1: ] - positions[:-1]).copy()

    def interaction_aware(distance, omega=5.0):
        return np.exp(-omega * distance)
    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor = thres
        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        feet_l_h = positions[:-1,fid_l,1]
        feet_l = interaction_aware(feet_l_h) * ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        feet_r_h = positions[:-1,fid_r,1]
        feet_r = interaction_aware(feet_r_h) * ((feet_r_x + feet_r_y + feet_r_z) < velfactor).astype(np.float32)
        return feet_l, feet_r
    #
    def contact_detect(verts, obj_points):
        # verts: T x N X 3 
        # obj_points: M x 3
        # return T x M
        contact = verts[:, :, np.newaxis, :] - obj_points[:, np.newaxis, :, :]
        contact = np.linalg.norm(contact, axis=-1)
        contact = np.min(contact, axis=1)
        contact = interaction_aware(contact)
        return contact[:-1]

    def interaction_aware_torch(distance, omega=5.0):
        return torch.exp(-omega * distance)

    def contact_detect_torch(verts, obj_points):
        # verts: T x N X 3 
        # obj_points: M x 3
        # return T x M
        verts = torch.tensor(verts).cuda()
        obj_points = torch.tensor(obj_points).cuda()
        contact = verts[:, :, None, :] - obj_points[:, None, :, :]
        contact = torch.norm(contact, dim=-1)
        contact, _ = torch.min(contact, dim=-1)
        # print(contact[0])
        contact = interaction_aware_torch(contact)
        return to_cpu(contact[:-1])

    
    contact_aware = contact_detect_torch(positions, obj_verts)

    feet_l, feet_r = foot_detect(positions, feet_thre)
    
    velocity = velocity.reshape(len(velocity), -1)
    # plot_markers("./results/{}_markes.mp4".format(int(x_offset*100)),positions,obj_verts)
    positions = positions.reshape(len(positions), -1)
    data = np.concatenate([positions[:-1], velocity, feet_l, feet_r, contact_aware, positions_obj[:-1]], axis=-1)
    return data
############################ utils ############################


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


# visualize markers motion of smpl model
if __name__ == "__main__":
    
    all_clips = 0
    all_frames = 0
    fid_r, fid_l = [61, 52, 53, 40, 34, 49, 40], [29, 30, 18, 19, 7, 2, 15]
    datasets = ['behave','intercap', 'grab', 'omomo']
    data_root = './data'
    for dataset in datasets:
        print(f'Loading {dataset} ...')
        frame_num = 0
        dataset_path = os.path.join(data_root, dataset)
        MOTION_PATH = os.path.join(dataset_path, 'sequences_canonical')
        OBJECT_PATH = os.path.join(data_root, dataset, 'objects')
        data_name = os.listdir(MOTION_PATH)
        for k, name in tqdm(enumerate(data_name)):
            
            if os.path.exists(os.path.join(MOTION_PATH, name, 'motion.npy')):
                continue
            if dataset.upper() == 'GRAB':
                verts, faces = visualize_grab(name, MOTION_PATH)
                markers = verts[:,markerset_smplx]
                
            elif dataset.upper() == 'BEHAVE':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
                markers = verts[:,markerset_smplh]
              
            elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
                markers = verts[:,markerset_smplh]
          
            elif dataset.upper() == 'CHAIRS':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
                markers = verts[:,markerset_smplx]
                
            elif dataset.upper() == 'INTERCAP':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 10, True)
                markers = verts[:,markerset_smplx]
           
            elif dataset.upper() == 'OMOMO':
                verts, faces = visualize_smpl(name, MOTION_PATH, 'smplx', 16)
                markers = verts[:,markerset_smplx]
           
            
           
            with np.load(os.path.join(MOTION_PATH, name, 'object.npz'), allow_pickle=True) as f:
                obj_angles, obj_trans, obj_name = f['angles'], f['trans'], str(f['name'])
            angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
            obj_rot6D = matrix_to_rotation_6d_np(angle_matrix)            

            obj_data = np.concatenate([obj_rot6D, obj_trans], axis=-1)
            obj_points = np.load(os.path.join(OBJECT_PATH, obj_name, 'sample_points.npy'))
            data = get_representation_canonical(markers.detach().cpu().numpy(), obj_data, obj_points)
            
            
            # 476 human + 18 obj + 6*78 rel obj
            np.save(os.path.join(MOTION_PATH, name, 'motion.npy'), data)
            
            seq_len = data.shape[0]
            frame_num += seq_len
            
        print('Total clips: %d, Frames: %d, Duration: %fm' %
            (len(data_name), frame_num, frame_num / 30 / 60))
        all_clips += len(data_name)
        all_frames += frame_num
    print('All clips: %d, Frames: %d, Duration: %fm' %
        (all_clips, all_frames, all_frames / 30 / 60))

    
