import json
import os
import os.path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import yaml
import smplx
import trimesh
from scipy.spatial.transform import Rotation
from torch.autograd import Variable
import copy
from render.mesh_viz import visualize_mesh


#from visualize_marker import plot_markers 
#from utils.markerset import *
# from render.mesh_viz import visualize_body_obj
import sys
import shutil
from sample.prior import *
from utils.markerset import *
from utils.markerset_mosh import all_marker_vids
import pickle
# from human_body_prior.body_model.body_model import BodyModel


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


MODEL_PATH = '/scratch/bcnt/sirui/dongting/models'


######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=False).to(device)

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=False).to(device)

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=False).to(device)

smplh10 = {'male': smplh_model_male,'female':smplh_model_female,'neutral':smplh_model_neutral}
######################################## smplx 10 ########################################
# smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
#                         gender = 'male',
#                         use_pca=False,
#                         ext='pkl')
                           
# smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
#                         gender="female",
#                         use_pca=False,
#                         ext='pkl')

# smplx_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
#                         gender="neutral",
#                         use_pca=False,
#                         ext='pkl')

# smplx10 = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}
# ######################################## smplx 10 pca 12 ########################################
# smplx12_model_male = smplx.create(MODEL_PATH, model_type='smplx',
#                           gender="male",
#                           num_pca_comps=12,
#                           use_pca=True,
#                           flat_hand_mean = True,
#                           ext='pkl')

# smplx12_model_female = smplx.create(MODEL_PATH, model_type='smplx',
#                           gender="female",
#                           num_pca_comps=12,
#                           use_pca=True,
#                           flat_hand_mean = True,
#                           ext='pkl')
# smplx12_model_neutral = smplx.create(MODEL_PATH, model_type='smplx',
#                           gender="neutral",
#                           num_pca_comps=12,
#                           use_pca=True,
#                           flat_hand_mean = True,
#                           ext='pkl')
# smplx12 = {'male': smplx12_model_male, 'female': smplx12_model_female, 'neutral': smplx12_model_neutral}
# ######################################## smplh 16 ########################################
# SMPLH_PATH = MODEL_PATH+'/smplh'
# surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
# surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
# surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral", "model.npz")
# dmpl_fname = None
# num_dmpls = None 
# num_expressions = None
# num_betas = 16 

# smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname,
#                 num_betas=num_betas,
#                 num_expressions=num_expressions,
#                 num_dmpls=num_dmpls,
#                 dmpl_fname=dmpl_fname)
# smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname,
#                 num_betas=num_betas,
#                 num_expressions=num_expressions,
#                 num_dmpls=num_dmpls,
#                 dmpl_fname=dmpl_fname)
# smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
#                 num_betas=num_betas,
#                 num_expressions=num_expressions,
#                 num_dmpls=num_dmpls,
#                 dmpl_fname=dmpl_fname)
# smplh16 = {'male': smplh16_model_male, 'female': smplh16_model_female, 'neutral': smplh16_model_neutral}
# ######################################## smplx 16 ########################################
# SMPLX_PATH = MODEL_PATH+'/smplx'
# surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
# surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
# surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

# smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname,
#                 num_betas=num_betas,
#                 num_expressions=num_expressions,
#                 num_dmpls=num_dmpls,
#                 dmpl_fname=dmpl_fname)
# smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname,
#                 num_betas=num_betas,
#                 num_expressions=num_expressions,
#                 num_dmpls=num_dmpls,
#                 dmpl_fname=dmpl_fname)
# smplx16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
#                 num_betas=num_betas,
#                 num_expressions=num_expressions,
#                 num_dmpls=num_dmpls,
#                 dmpl_fname=dmpl_fname)
# smplx16 = {'male': smplx16_model_male, 'female': smplx16_model_female, 'neutral': smplx16_model_neutral}
# ########################################################################################
# results_folder = "./results"
# os.makedirs(results_folder, exist_ok=True)

# ######################################## Visualize SMPL ########################################
# def visualize_smpl(name, MOTION_PATH, model_type, num_betas, use_pca=False):
#     """
#     BEHAVE for SMPLH 10
#     NEURALDOME or IMHD for SMPLH 16
#     vertices: (N, 6890, 3)
#     Chairs for SMPLX 10
#     InterCap for SMPLX 10 PCA 12
#     OMOMO for SMPLX 16
#     vertices: (N, 10475, 3)
#     """
#     with np.load(os.path.join(MOTION_PATH, name, 'human.npz'), allow_pickle=True) as f:
#         poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
        
#     frame_times = poses.shape[0]
#     if num_betas == 10:
#         if model_type == 'smplh':
#             smpl_model = smplh10[gender].cuda()
#             smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).cuda().float(),
#                                 global_orient=torch.from_numpy(poses[:, :3]).cuda().float(),
#                                 left_hand_pose=torch.from_numpy(poses[:, 66:111]).cuda().float(),
#                                 right_hand_pose=torch.from_numpy(poses[:, 111:156]).cuda().float(),
#                                 betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).cuda().float(),
#                                 transl=torch.from_numpy(trans).cuda().float(),) 
#         elif model_type == 'smplx':
#             if use_pca:
#                 smpl_model = smplx12[gender].cuda()
#                 smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).cuda().float(),
#                               global_orient=torch.from_numpy(poses[:, :3]).cuda().float(),
#                               left_hand_pose=torch.from_numpy(poses[:, 66:78]).cuda().float(),
#                               right_hand_pose=torch.from_numpy(poses[:, 78:90]).cuda().float(),
#                               jaw_pose=torch.zeros(frame_times, 3).cuda().float(),
#                               leye_pose=torch.zeros(frame_times, 3).cuda().float(),
#                               reye_pose=torch.zeros(frame_times, 3).cuda().float(),
#                               expression=torch.zeros(frame_times, 10).cuda().float(),
#                               betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).cuda().float(),
#                               transl=torch.from_numpy(trans).cuda().float(),)
#             else:
#                 smpl_model = smplx10[gender].cuda()
#                 smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).cuda().float(),
#                                     global_orient=torch.from_numpy(poses[:, :3]).cuda().float(),
#                                     left_hand_pose=torch.from_numpy(poses[:, 66:111]).cuda().float(),
#                                     right_hand_pose=torch.from_numpy(poses[:, 111:156]).cuda().float(),
#                                     jaw_pose = torch.zeros([frame_times,3]).cuda().float(),
#                                     reye_pose = torch.zeros([frame_times,3]).cuda().float(),
#                                     leye_pose = torch.zeros([frame_times,3]).cuda().float(),
#                                     expression = torch.zeros([frame_times,10]).cuda().float(),
#                                     betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).cuda().float(),
#                                     transl=torch.from_numpy(trans).cuda().float(),)
#         verts = to_cpu(smplx_output.vertices)
#         faces = smpl_model.faces
#         joints = to_cpu(smplx_output.joints)
#     elif num_betas == 16: 
#         if model_type == 'smplh':
#             smpl_model = smplh16[gender].cuda()
#         elif model_type == 'smplx':
#             smpl_model = smplx16[gender].cuda()
#         smplx_output = smpl_model(pose_body=torch.from_numpy(poses[:, 3:66]).cuda().float(), 
#                             pose_hand=torch.from_numpy(poses[:, 66:156]).cuda().float(), 
#                             betas=torch.from_numpy(betas[None, :]).cuda().repeat(frame_times, 1).float(), 
#                             root_orient=torch.from_numpy(poses[:, :3]).cuda().float(), 
#                             trans=torch.from_numpy(trans).cuda().float())
#         verts = to_cpu(smplx_output.v)
#         faces = smpl_model.f.detach().cpu().numpy()
#         joints = to_cpu(smplx_output.Jtr)
#     angles = torch.from_numpy(poses[:, :3]).float()
#     trans = torch.from_numpy(trans).float()
#     return verts, faces, joints, angles, trans

# ######################################## utils for GRAB ########################################
# class dotdict(dict):
#     """dot.notation access to dictionary attributes"""
#     __getattr__ = dict.get
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


# def points2sphere(points, radius = .001, vc = [0., 0., 1.], count = [5,5]):

#     points = points.reshape(-1,3)
#     n_points = points.shape[0]

#     spheres = []
#     for p in range(n_points):
#         sphs = trimesh.creation.uv_sphere(radius=radius, count = count)
#         sphs.apply_translation(points[p])
#         sphs = Mesh(vertices=sphs.vertices, faces=sphs.faces, vc=vc)

#         spheres.append(sphs)

#     spheres = Mesh.concatenate_meshes(spheres)
#     return spheres

# class Mesh(trimesh.Trimesh):

#     def __init__(self,
#                  filename=None,
#                  vertices=None,
#                  faces=None,
#                  vc=None,
#                  fc=None,
#                  vscale=None,
#                  process = False,
#                  visual = None,
#                  wireframe=False,
#                  smooth = False,
#                  **kwargs):

#         self.wireframe = wireframe
#         self.smooth = smooth

#         if filename is not None:
#             mesh = trimesh.load(filename, process = process)
#             vertices = mesh.vertices
#             faces= mesh.faces
#             visual = mesh.visual
#         if vscale is not None:
#             vertices = vertices*vscale

#         if faces is None:
#             mesh = points2sphere(vertices)
#             vertices = mesh.vertices
#             faces = mesh.faces
#             visual = mesh.visual

#         super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

#         if vc is not None:
#             self.set_vertex_colors(vc)
#         if fc is not None:
#             self.set_face_colors(fc)

#     def rot_verts(self, vertices, rxyz):
#         return np.array(vertices * rxyz.T)

#     def colors_like(self,color, array, ids):

#         color = np.array(color)

#         if color.max() <= 1.:
#             color = color * 255
#         color = color.astype(np.int8)

#         n_color = color.shape[0]
#         n_ids = ids.shape[0]

#         new_color = np.array(array)
#         if n_color <= 4:
#             new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
#         else:
#             new_color[ids, :] = color

#         return new_color

#     def set_vertex_colors(self,vc, vertex_ids = None):

#         all_ids = np.arange(self.vertices.shape[0])
#         if vertex_ids is None:
#             vertex_ids = all_ids

#         vertex_ids = all_ids[vertex_ids]
#         new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
#         self.visual.vertex_colors[:] = new_vc

#     def set_face_colors(self,fc, face_ids = None):

#         if face_ids is None:
#             face_ids = np.arange(self.faces.shape[0])

#         new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
#         self.visual.face_colors[:] = new_fc

#     @staticmethod
#     def concatenate_meshes(meshes):
#         return trimesh.util.concatenate(meshes)
# def DotDict(in_dict):
    
#     out_dict = copy(in_dict)
#     for k,v in out_dict.items():
#        if isinstance(v,dict):
#            out_dict[k] = DotDict(v)
#     return dotdict(out_dict)

# def parse_npz(npz, allow_pickle=True):
#     npz = np.load(npz, allow_pickle=allow_pickle)
#     npz = {k: npz[k].item() for k in npz.files}
#     return DotDict(npz)

# def params2torch(params, dtype = torch.float32):
#     return {k: torch.from_numpy(v).type(dtype).to(device) for k, v in params.items()}

# sbj_info = {}
# def load_sbj_verts(sbj_id, seq_data, data_root_folder = './data/grab/'):
    
#     mesh_path = os.path.join(data_root_folder,seq_data.body.vtemp)
#     if sbj_id in sbj_info:
#         sbj_vtemp = sbj_info[sbj_id]
#     else:
#         sbj_vtemp = np.array(Mesh(filename=mesh_path).vertices)
#         sbj_info[sbj_id] = sbj_vtemp
#     return sbj_vtemp
# ######################################## Visualize GRAB ########################################
# def visualize_grab(name, MOTION_PATH):
#     """
#     vertices: (N, 10475, 3)
#     """
#     motion_file = os.path.join(MOTION_PATH,name,'motion.npz')
#     seq_data = parse_npz(motion_file)
#     n_comps = seq_data['n_comps']
#     gender = seq_data['gender']
#     sbj_id = seq_data['sbj_id']
#     T = seq_data.n_frames
#     sbj_vtemp = load_sbj_verts(sbj_id, seq_data, os.path.dirname(MOTION_PATH))

#     smpl_model = smplx.create( 
#         model_path=MODEL_PATH,
#         model_type='smplx',
#         gender=gender,
#         num_pca_comps=n_comps,
#         v_template = sbj_vtemp,
#         batch_size=T).cuda()
#     sbj_parms = params2torch(seq_data.body.params)

#     angles = to_cpu(sbj_parms.global_orient)
#     trans = to_cpu(sbj_parms.transl)
#     smplx_output = smpl_model(**sbj_parms)
#     verts = to_cpu(smplx_output.vertices)
#     faces = smpl_model.faces
#     joints = to_cpu(smplx_output.joints)

#     return verts, faces, joints, angles, trans


# def visualize_dataset(dataset,MOTION_PATH,name):
#     if dataset.upper() == 'GRAB':
#         verts, faces, joints, angles, trans = visualize_grab(name, MOTION_PATH)
#     elif dataset.upper() == 'BEHAVE':
#         verts, faces, joints, angles, trans = visualize_smpl(name, MOTION_PATH, 'smplh', 10)
#     elif dataset.upper() == 'NEURALDOME' or dataset.upper() == 'IMHD':
#         verts, faces, joints, angles, trans = visualize_smpl(name, MOTION_PATH, 'smplh', 16)
#     elif dataset.upper() == 'CHAIRS':
#         verts, faces, joints, angles, trans = visualize_smpl(name, MOTION_PATH, 'smplx', 10)
#     elif dataset.upper() == 'INTERCAP':
#         verts, faces, joints, angles, trans = visualize_smpl(name, MOTION_PATH, 'smplx', 10, True)
#     elif dataset.upper() == 'OMOMO':
#         verts, faces, joints, angles, trans = visualize_smpl(name, MOTION_PATH, 'smplx', 16)
#     return verts, faces, joints, angles, trans

class SmplhOptmize10_betas(nn.Module):
    def __init__(self, gender, batch_size, frame_times, betas):
        device=torch.device('cuda:0')
        super(SmplhOptmize10_betas, self).__init__()
        self.smpl_model = smplh10[gender]
        self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
        self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)

        #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
        self.pred_pose.requires_grad=True
        # omomo sub9 betas:  
        # [ 1.2644597   0.4629662  -0.9876839  -0.6337372   1.4846485  -0.05660084  1.6636678  -0.7218272   2.580027    2.314394]
        self.pred_betas = Variable(torch.tensor(np.tile(betas, (batch_size, 1))).float().to(device),requires_grad=False)
        self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='./assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            # print(verts.shape,markers.shape)
            H=(torch.sum((verts[:,[1861,5322,1058,4544]]-markers[:,[28,60,19,53]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
        # print(self.pred_trans.shape)


            
        # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
        #                                     lr=1e-2, line_search_fn='strong_wolfe')
        #self.optimizer= torch.optim.Adam([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], lr=0.01)
            

    def ankle_loss(self):
        return torch.sum(torch.exp(self.pred_pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=self.pred_pose.device)) ** 2)
    def smooth(self):
        return torch.sum((self.pred_pose[1:]-self.pred_pose[:-1])**2)+torch.sum((self.left_hand_pose[1:]-self.left_hand_pose[:-1])**2)+\
        torch.sum((self.right_hand_pose[1:]-self.right_hand_pose[:-1])**2)+\
        torch.sum((self.pred_trans[1:]-self.pred_trans[:-1])**2)+torch.sum((self.glo_pose[1:]-self.glo_pose[:-1])**2)
    def forward_human(self):
        smpl_output = self.smpl_model(body_pose=self.pred_pose[:, :],
            global_orient=self.glo_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10),
            transl=self.pred_trans,)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    # def gmof(self,x, sigma):
    
    #     x_squared = x ** 2
    #     sigma_squared = sigma ** 2
    #     return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                pred_markers = verts[:,markerset_smplh]
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                loss2=5*self.smooth()
                loss=loss1+loss2
                loss.backward()
                return loss
            cam_t_optimizer.step(closure)
            
    def beta_restrict(self):
        return torch.sum(self.pred_betas**2)
    def optimize_whole(self,markers_gt):
        body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        
        for i in tqdm(range(100)):
            def closure():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                pred_markers = verts[:,markerset_smplh]
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                loss2=5*self.smooth()
                loss3=5*self.ankle_loss()
                # loss4=5*self.beta_restrict()
                loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
                        self.prior.forward(self.pred_pose)
                
                loss=loss1+loss2+loss3+loss5
                loss.backward()
                #print(loss.shape)
                return loss

            body_optimizer.step(closure)
        with torch.no_grad():
            verts,joints=self.forward_human()
            return verts.detach(), self.smpl_model.faces
            
        
        

    def forward(self,markers_gt):
        self.init_guess(markers_gt)
        self.optimize_cam(markers_gt)
        return self.optimize_whole(markers_gt)

class SmplhOptmize10(nn.Module):
    def __init__(self, gender, batch_size, frame_times):
        device=torch.device('cuda:0')
        super(SmplhOptmize10, self).__init__()
        self.smpl_model = smplh10[gender]
        self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
        self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)

        #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
        self.pred_pose.requires_grad=True

        self.pred_betas = Variable(torch.tensor(np.zeros((batch_size, 10))).float().to(device),requires_grad=True)
        self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='./assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            # print(verts.shape,markers.shape)
            H=(torch.sum((verts[:,[1861,5322,1058,4544]]-markers[:,[28,60,19,53]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
        # print(self.pred_trans.shape)


            
        # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
        #                                     lr=1e-2, line_search_fn='strong_wolfe')
        #self.optimizer= torch.optim.Adam([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], lr=0.01)
            

    def ankle_loss(self):
        return torch.sum(torch.exp(self.pred_pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=self.pred_pose.device)) ** 2)
    def smooth(self):
        return torch.sum((self.pred_pose[1:]-self.pred_pose[:-1])**2)+torch.sum((self.left_hand_pose[1:]-self.left_hand_pose[:-1])**2)+\
        torch.sum((self.right_hand_pose[1:]-self.right_hand_pose[:-1])**2)+\
        torch.sum((self.pred_trans[1:]-self.pred_trans[:-1])**2)+torch.sum((self.glo_pose[1:]-self.glo_pose[:-1])**2)
    def forward_human(self):
        smpl_output = self.smpl_model(body_pose=self.pred_pose[:, :],
            global_orient=self.glo_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10),
            transl=self.pred_trans,)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    # def gmof(self,x, sigma):
    
    #     x_squared = x ** 2
    #     sigma_squared = sigma ** 2
    #     return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                pred_markers = verts[:,markerset_smplh]
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                loss=loss1#+loss2
                loss.backward()
                return loss
            cam_t_optimizer.step(closure)
            
    def beta_restrict(self):
        return torch.sum(self.pred_betas**2)
    def optimize_whole(self,markers_gt):
        body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        
        for i in tqdm(range(100)):
            def closure():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                pred_markers = verts[:,markerset_smplh]
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                # loss3=5*self.ankle_loss()
                # # loss4=5*self.beta_restrict()
                # loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
                #         self.prior.forward(self.pred_pose)
                
                loss=loss1#+loss2+loss3+loss5
                loss.backward()
                #print(loss.shape)
                return loss

            body_optimizer.step(closure)
        with torch.no_grad():
            verts,joints=self.forward_human()
            return verts.detach(), self.smpl_model.faces.astype(np.int32)
            
        
        

    def forward(self,markers_gt):
        self.init_guess(markers_gt)
        self.optimize_cam(markers_gt)
        return self.optimize_whole(markers_gt)
        
    


# path='./zmarkers.npy'
# M=np.load(path)
# print(M.shape)
# M=M[:50]
# markers_gt=torch.from_numpy(M).float().reshape(-1,77,3).to(device)
# ft=markers_gt.shape[0]
# MODEL=SmplhOptmize10('male',ft)
# MODEL.forward(markers_gt)


        

    

# class SmplxOptmize10(nn.Module):
#     def __init__(self, gender, frame_times):
#         super(SmplxOptmize10, self).__init__()
#         self.smpl_model = smplx10[gender]
#         self.pred_pose = nn.Parameter(torch.zeros(frame_times, 66))
#         self.pred_betas = nn.Parameter(torch.zeros(frame_times, 10))
#         self.pred_trans = nn.Parameter(torch.zeros(frame_times, 3))
#         self.left_hand_pose = nn.Parameter(torch.zeros(frame_times, 45))
#         self.right_hand_pose = nn.Parameter(torch.zeros(frame_times, 45))
#         self.frame_times = frame_times
    
#     def forward(self):
#         smpl_output = self.smpl_model(body_pose=self.pred_pose[:, 3:66],
#                                 global_orient=self.pred_pose[:, :3],
#                                 left_hand_pose=self.left_hand_pose,
#                                 right_hand_pose=self.right_hand_pose,
#                                 jaw_pose = torch.zeros([self.frame_times,3]).cuda().float(),
#                                 reye_pose = torch.zeros([self.frame_times,3]).cuda().float(),
#                                 leye_pose = torch.zeros([self.frame_times,3]).cuda().float(),
#                                 expression = torch.zeros([self.frame_times,10]).cuda().float(),
#                                 betas=self.pred_betas,
#                                 transl=self.pred_trans,)
#         verts = smpl_output.vertices
#         joints = smpl_output.joints
#         pred_markers = verts[:,markerset_smplx]
#         return pred_markers, verts, joints

# class SmplxOptmize12(nn.Module):
#     def __init__(self, gender, frame_times):
#         super(SmplxOptmize12, self).__init__()
#         self.smpl_model = smplx12[gender]
#         self.pred_pose = nn.Parameter(torch.zeros(frame_times, 66))
#         self.pred_betas = nn.Parameter(torch.zeros(frame_times, 10))
#         self.pred_trans = nn.Parameter(torch.zeros(frame_times, 3))
#         self.left_hand_pose = nn.Parameter(torch.zeros(frame_times, 12))
#         self.right_hand_pose = nn.Parameter(torch.zeros(frame_times, 12))
#         self.frame_times = frame_times

#     def forward(self):
#         smpl_output = self.smpl_model(body_pose=self.pred_pose[:, 3:66],
#             global_orient=self.pred_pose[:, :3],
#             left_hand_pose=self.left_hand_pose,
#             right_hand_pose=self.right_hand_pose,
#             jaw_pose=torch.zeros(self.frame_times, 3).float(),
#             leye_pose=torch.zeros(self.frame_times, 3).float(),
#             reye_pose=torch.zeros(self.frame_times, 3).float(),
#             expression=torch.zeros(self.frame_times, 10).float(),
#             betas=self.pred_betas,
#             transl=self.pred_trans,)
#         verts = smpl_output.vertices
#         joints = smpl_output.joints
#         pred_markers = verts[:,markerset_smplx]
#         return pred_markers, verts, joints
    
    


# def markers2smpl(markers):
#     crieterion = nn.MSELoss()

#     gender =  'male'

#     frame_times = len(markers)


#     smpl_op_model = SmplhOptmize10(gender, frame_times).cuda()


#     markers = torch.from_numpy(markers).cuda()
#     optimizer = torch.optim.Adam(smpl_op_model.parameters(), lr=0.1)
#     smpl_op_model.train()
#     for i in range(1000):
#         optimizer.zero_grad()
#         pred_markers, verts, joints = smpl_op_model()
#         loss = crieterion(pred_markers, markers)
#         loss.backward()
#         optimizer.step()
#     faces = smpl_op_model.smpl_model.faces

class SmplhOptmize10_handjoints(nn.Module):
    def __init__(self, gender, batch_size, frame_times,extra=[]):
        device=torch.device('cuda:0')
        super(SmplhOptmize10_handjoints, self).__init__()
        self.extra=[]
        self.smpl_model = smplh10[gender]
        self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
        self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)

        #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
        self.pred_pose.requires_grad=True

        self.pred_betas = Variable(torch.tensor(np.zeros((batch_size, 10))).float().to(device),requires_grad=True)
        self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='./assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            # print(verts.shape,markers.shape)
            H=(torch.sum((verts[:,[1861,5322,1058,4544]]-markers[:,[28,60,19,53]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
        # print(self.pred_trans.shape)


            
        # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
        #                                     lr=1e-2, line_search_fn='strong_wolfe')
        #self.optimizer= torch.optim.Adam([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], lr=0.01)
            

    def ankle_loss(self):
        return torch.sum(torch.exp(self.pred_pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=self.pred_pose.device)) ** 2)
    def smooth(self):
        return torch.sum((self.pred_pose[1:]-self.pred_pose[:-1])**2)+torch.sum((self.left_hand_pose[1:]-self.left_hand_pose[:-1])**2)+\
        torch.sum((self.right_hand_pose[1:]-self.right_hand_pose[:-1])**2)+\
        torch.sum((self.pred_trans[1:]-self.pred_trans[:-1])**2)+torch.sum((self.glo_pose[1:]-self.glo_pose[:-1])**2)
    def forward_human(self):
        smpl_output = self.smpl_model(body_pose=self.pred_pose[:, :],
            global_orient=self.glo_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10),
            transl=self.pred_trans,)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    # def gmof(self,x, sigma):
    
    #     x_squared = x ** 2
    #     sigma_squared = sigma ** 2
    #     return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                pred_markers = verts[:,markerset_smplh_hand_new+self.extra]
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                loss=loss1#+loss2
                loss.backward()
                return loss
            cam_t_optimizer.step(closure)
            
    def beta_restrict(self):
        return torch.sum(self.pred_betas**2)
    def optimize_whole(self,markers_gt):
        body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        
        for i in tqdm(range(100)):
            def closure():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                pred_markers = verts[:,markerset_smplh_hand_new+self.extra]
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                # loss3=5*self.ankle_loss()
                # # loss4=5*self.beta_restrict()
                # loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
                #         self.prior.forward(self.pred_pose)
                
                loss=loss1#+loss2+loss3+loss5
                loss.backward()
                #print(loss.shape)
                return loss

            body_optimizer.step(closure)
        with torch.no_grad():
            verts,joints=self.forward_human()
            return verts.detach(), self.smpl_model.faces.astype(np.int32)
            
        
        

    def forward(self,markers_gt):
        self.init_guess(markers_gt)
        self.optimize_cam(markers_gt)
        return self.optimize_whole(markers_gt)


class SmplhOptmize10_fulljoints(nn.Module):
    def __init__(self, gender, batch_size, frame_times,extra=[],joint_nums=52):
        device=torch.device('cuda:0')
        super(SmplhOptmize10_fulljoints, self).__init__()
        self.extra=[]
        self.joint_nums = joint_nums
        self.smpl_model = smplh10[gender]
        self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
        self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)

        #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
        self.pred_pose.requires_grad=True
        self.djoints_index =list(range(22))+list(range(25,55)) 


        self.pred_betas = Variable(torch.tensor(np.zeros((batch_size, 10))).float().to(device),requires_grad=True)
        self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='./assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            # print(verts.shape,markers.shape)
            H=(torch.sum((joints[:,[1,2,16,17]]-markers[:,[1,2,16,17]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
        # print(self.pred_trans.shape)


            
        # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
        #                                     lr=1e-2, line_search_fn='strong_wolfe')
        #self.optimizer= torch.optim.Adam([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], lr=0.01)
            

    def ankle_loss(self):
        return torch.sum(torch.exp(self.pred_pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=self.pred_pose.device)) ** 2)
    def smooth(self):
        return torch.sum((self.pred_pose[1:]-self.pred_pose[:-1])**2)+torch.sum((self.left_hand_pose[1:]-self.left_hand_pose[:-1])**2)+\
        torch.sum((self.right_hand_pose[1:]-self.right_hand_pose[:-1])**2)+\
        torch.sum((self.pred_trans[1:]-self.pred_trans[:-1])**2)+torch.sum((self.glo_pose[1:]-self.glo_pose[:-1])**2)
    def forward_human(self):
        smpl_output = self.smpl_model(body_pose=self.pred_pose[:, :],
            global_orient=self.glo_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10),
            transl=self.pred_trans,)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    # def gmof(self,x, sigma):
    
    #     x_squared = x ** 2
    #     sigma_squared = sigma ** 2
    #     return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # insert
                # if self.joint_nums ==52:
                #     pred_markers = joints[:,self.djoints_index] 
                # else:
                pred_markers = joints[:,:self.joint_nums] 
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                loss=loss1#+loss2
                loss.backward()
                return loss
            cam_t_optimizer.step(closure)
            
    def beta_restrict(self):
        return torch.sum(self.pred_betas**2)
    def optimize_whole(self,markers_gt):
        body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        
        for i in tqdm(range(100)):
            def closure():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # if self.joint_nums == 52:
                    
                #     pred_markers = joints[:,self.djoints_index]
                # else:
                pred_markers = joints[:,:self.joint_nums]
                    
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=2*self.smooth()
                # loss3=5*self.ankle_loss()
                loss4=5*self.beta_restrict()
                loss5 =torch.sum(self.left_hand_pose**2+self.right_hand_pose**2)+torch.sum(self.pred_pose**2)+torch.sum(self.glo_pose**2)
                # loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.right_hand_pose,left_or_right=1)**2)+\
                #         self.prior.forward(self.pred_pose)+torch.sum(self.left_hand_pose**2+self.right_hand_pose**2)+torch.sum(self.pred_pose**2)+torch.sum(self.glo_pose**2)
                # +loss5
                loss=loss1+loss4+loss5
                loss.backward()
                #print(loss.shape)
                return loss

            body_optimizer.step(closure)
        with torch.no_grad():
            verts,joints=self.forward_human()
            return verts.detach(), self.smpl_model.faces.astype(np.int32),torch.cat([self.glo_pose,self.pred_pose,self.left_hand_pose,self.right_hand_pose],-1).detach().cpu().numpy(),self.pred_betas.detach().cpu().numpy(),self.pred_trans.detach().cpu().numpy()
            
        
        

    def forward(self,markers_gt):
        self.init_guess(markers_gt)
        self.optimize_cam(markers_gt)
        return self.optimize_whole(markers_gt)





class SmplhOptmize10_fulljoints_mixamo(nn.Module):
    def __init__(self, gender, batch_size, frame_times,extra=[],joint_nums=52,betas=np.zeros((10)),init_pose=0,init_trans=0):
        device=torch.device('cuda:0')
        super(SmplhOptmize10_fulljoints_mixamo, self).__init__()
        self.extra=[]
        self.joint_nums = joint_nums
        self.smpl_model = smplh10[gender]
        # self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
        # self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        
        self.pred_pose = Variable((init_pose[:,1:22]).reshape(frame_times,-1).float().to(device),requires_grad=True)
        self.glo_pose = Variable((init_pose[:,:1]).reshape(frame_times,-1).float().to(device),requires_grad=True)

        #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
        self.pred_pose.requires_grad=True
        self.djoints_index =list(range(22))+list(range(25,55)) 

        betas = np.repeat(betas.reshape(1,-1), batch_size, axis=0)
        # print(betas.shape,'BETAS')
        self.pred_betas = Variable(torch.from_numpy(betas).float().to(device),requires_grad=True)
        # self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        self.pred_trans = Variable(init_trans.reshape(frame_times,-1).float().to(device),requires_grad=True)
        
        self.trans_rec =init_trans.detach().clone()
        self.pose_rec =init_pose[:,1:22].reshape(frame_times,-1).detach().clone()
        self.glopose_rec =init_pose[:,:1].reshape(frame_times,-1).detach().clone()
        # root_path='./assets'
        # lhand_path = os.path.join(root_path, 'priors', 'lh_prior.pkl')
        # rhand_path = os.path.join(root_path, 'priors', 'rh_prior.pkl')
        # lhand_data = pickle.load(open(lhand_path, 'rb'))['mean'].reshape(1,-1)
        # rhand_data = pickle.load(open(rhand_path, 'rb'))['mean'].reshape(1,-1)
        # print(lhand_data.shape,rhand_data.shape)
        # print(lhand_data,rhand_data)
        # lhand_data=np.repeat(lhand_data, batch_size*frame_times, axis=0)
        # rhand_data=np.repeat(rhand_data, batch_size*frame_times, axis=0)
        lhand_data = init_pose[:,22:37].reshape(frame_times,-1)
        rhand_data = init_pose[:,37:52].reshape(frame_times,-1)
        
        
        self.left_hand_pose = Variable((lhand_data).float().to(device),requires_grad=False)
        
        self.right_hand_pose = Variable((rhand_data).float().to(device),requires_grad=True)
        
        self.lhand_rec = lhand_data.detach().clone()
        self.rhand_rec = rhand_data.detach().clone()
        # self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        
        # self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='./assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            # print(verts.shape,markers.shape)
            H=(torch.sum((joints[:,[1,2,16,17]]-markers[:,[1,2,16,17]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
        # print(self.pred_trans.shape)


            
        # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
        #                                     lr=1e-2, line_search_fn='strong_wolfe')
        #self.optimizer= torch.optim.Adam([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], lr=0.01)
            

    def ankle_loss(self):
        return torch.sum(torch.exp(self.pred_pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=self.pred_pose.device)) ** 2)
    def smooth(self):
        return torch.sum((self.pred_pose[1:]-self.pred_pose[:-1])**2)+torch.sum((self.left_hand_pose[1:]-self.left_hand_pose[:-1])**2)+\
        torch.sum((self.right_hand_pose[1:]-self.right_hand_pose[:-1])**2)+\
        torch.sum((self.pred_trans[1:]-self.pred_trans[:-1])**2)+torch.sum((self.glo_pose[1:]-self.glo_pose[:-1])**2)
    def forward_human(self):
        smpl_output = self.smpl_model(body_pose=self.pred_pose[:, :],
            global_orient=self.glo_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10),
            transl=self.pred_trans,)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    
    def forward_human_hand(self):
        pred_poses = torch.cat([self.pred_pose[:, :19*3].detach(),self.pred_pose[:, 19*3:]],-1)
        smpl_output = self.smpl_model(body_pose=pred_poses,
            global_orient=self.glo_pose.detach(),
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10).detach(),
            transl=self.pred_trans.detach(),)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    # def gmof(self,x, sigma):
    
    #     x_squared = x ** 2
    #     sigma_squared = sigma ** 2
    #     return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # insert
                # if self.joint_nums ==52:
                #     pred_markers = joints[:,self.djoints_index] 
                # else:
                pred_markers = joints[:,:self.joint_nums] 
                loss1=100*(torch.sum((pred_markers[:,:22]-markers_gt[:,:22])**2))+ torch.sum((self.pred_trans-self.trans_rec)**2)
                # loss2=5*self.smooth()
                loss=loss1#+loss2
                loss.backward()
                return loss
            cam_t_optimizer.step(closure)
            
    def beta_restrict(self):
        return torch.sum(self.pred_betas**2)
    def optimize_whole(self,markers_gt):
        
        body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.left_hand_pose,self.right_hand_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        hand_optimizer = torch.optim.LBFGS([self.pred_pose,self.left_hand_pose,self.right_hand_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        # self.pred_betas
        # self.left_hand_pose,self.right_hand_pose
        
        for i in tqdm(range(100)):
            def closure_m():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                verts2,joints2=self.forward_human_hand()
                # if self.joint_nums == 52:
                    
                #     pred_markers = joints[:,self.djoints_index]
                # else:
                pred_markers = joints[:,:self.joint_nums]
                # print(joints.shape,joints2.shape)
                pred_markers2 = joints2[:,:self.joint_nums]
                    
                loss1=100*(torch.sum((pred_markers[:,:22]-markers_gt[:,:22])**2)) + 100*(torch.sum((pred_markers2[:,22:52]-markers_gt[:,22:52])**2))
                # loss2=2*self.smooth()
                # loss3=5*self.ankle_loss()
                loss4=5*self.beta_restrict()
                loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.right_hand_pose,left_or_right=1)**2)+\
                        self.prior.forward(self.pred_pose)
                
                loss=loss1+loss5+loss4
                loss.backward()
                #print(loss.shape)
                return loss
            def closure0():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # if self.joint_nums == 52:
                    
                #     pred_markers = joints[:,self.djoints_index]
                jtr =joints
                left_static = self.left_static
                right_static = self.right_static
                left_foot = jtr[:, 10]
                right_foot = jtr[:, 11]
                if left_static.any():
                    loss_left = torch.mean((((left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]])[left_static]) ** 2))
                else:
                    loss_left = 0
                if right_static.any():
                    loss_right = torch.mean((((right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]])[right_static]) ** 2))
                else:
                    loss_right = 0
                # else:
                pred_markers = joints[:,:self.joint_nums]
                    
                loss1=100*(torch.sum((pred_markers[:,:]-markers_gt[:,:])**2)) 
                # + 500*(torch.sum((pred_markers[:,22:52]-markers_gt[:,22:52])**2))
                # loss2=2*self.smooth()
                # loss3=5*self.ankle_loss()
                loss4=5*self.beta_restrict()
                loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.right_hand_pose,left_or_right=1)**2)+\
                        self.prior.forward(self.pred_pose)
                loss6 = torch.sum((self.left_hand_pose-self.lhand_rec)**2) + torch.sum((self.right_hand_pose-self.rhand_rec)**2) + torch.sum((self.pred_pose-self.pose_rec)**2) + torch.sum((self.glo_pose-self.glopose_rec)**2) + torch.sum((self.pred_trans-self.trans_rec)**2)
                
                loss=loss1+loss5*0.1+loss4+loss_left+loss_right+loss6*0.2
                loss.backward()
                #print(loss.shape)
                return loss
            def closure():
                
            
                body_optimizer.zero_grad()
                hand_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # if self.joint_nums == 52:
                    
                #     pred_markers = joints[:,self.djoints_index]
                # else:
                pred_markers = joints[:,:self.joint_nums]
                    
                loss1=100*(torch.sum((pred_markers[:,:-30]-markers_gt[:,:-30])**2))
                # loss2=2*self.smooth()
                # loss3=5*self.ankle_loss()
                loss4=5*self.beta_restrict()
                loss5=self.prior.forward(self.pred_pose)
                
                loss=loss1+loss5+loss4
                loss.backward()
                
                
                return loss
                
                
                #print(loss.shape)
                
                
            
            def closure2():
                
            
                
                hand_optimizer.zero_grad()
                
                # if self.joint_nums == 52:
                    
                #     pred_markers = joints[:,self.djoints_index]
                # else:
                
                verts2,joints2 = self.forward_human_hand()
                
                pred_markers2 = joints2[:,:self.joint_nums]
                
                loss_1=100*(torch.sum((pred_markers2[:,-30:]-markers_gt[:,-30:])**2))
                # loss2=2*self.smooth()
                # loss3=5*self.ankle_loss()
                # loss_4=5*self.beta_restrict()
                loss_5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)
                
                loss_h=loss_1+loss_5
                loss_h.backward()
                return loss_h
                
                
                #print(loss.shape)
            body_optimizer.step(closure0)
            # if i<60:
                
            #     body_optimizer.step(closure0)
            # else:
            #     body_optimizer.step(closure0)
            #     hand_optimizer.step(closure2)
            # torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
        with torch.no_grad():
            verts,joints=self.forward_human()
            return verts.detach(), self.smpl_model.faces.astype(np.int32),torch.cat([self.glo_pose,self.pred_pose,self.left_hand_pose,self.right_hand_pose],-1).detach().cpu().numpy(),self.pred_betas.detach().cpu().numpy(),self.pred_trans.detach().cpu().numpy()
            
        
        

    def forward(self,markers_gt):
        jtr_gt = markers_gt
        left_foot = jtr_gt[:, 10]
        right_foot = jtr_gt[:, 11]
        delta_left = torch.norm(left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]], dim=1) + 1e-6
        delta_right = torch.norm(right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]], dim=1) + 1e-6
        self.left_static = (delta_left < 0.008)
        self.right_static = (delta_right < 0.008)
        self.init_guess(markers_gt)
        self.optimize_cam(markers_gt)
        return self.optimize_whole(markers_gt)

    
    



class SmplhOptmize10_fulljoints2(nn.Module):
    def __init__(self, gender, batch_size, frame_times,extra=[],joint_nums=52):
        device=torch.device('cuda:0')
        super(SmplhOptmize10_fulljoints2, self).__init__()
        self.extra=[]
        self.joint_nums = joint_nums
        self.smpl_model = smplh10[gender]
        self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
        self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)

        #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
        self.pred_pose.requires_grad=True
        self.djoints_index =list(range(22))+list(range(25,55)) 


        self.pred_betas = Variable(torch.tensor(np.zeros((batch_size, 10))).float().to(device),requires_grad=True)
        self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='./assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            # print(verts.shape,markers.shape)
            H=(torch.sum((joints[:,[1,2,16,17]]-markers[:,[1,2,16,17]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
        # print(self.pred_trans.shape)


            
        # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
        #                                     lr=1e-2, line_search_fn='strong_wolfe')
        #self.optimizer= torch.optim.Adam([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], lr=0.01)
            

    def ankle_loss(self):
        return torch.sum(torch.exp(self.pred_pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=self.pred_pose.device)) ** 2)
    def vel_consistency_loss(self,x):
    # x shape: [T, ...]
        v = x[1:] - x[:-1]       # velocity
        dv = v[1:] - v[:-1]      # velocity change (should be small)
        return (dv ** 2).mean()
    def smooth(self):
        return torch.mean((self.pred_pose[1:]-self.pred_pose[:-1])**2)+torch.mean((self.left_hand_pose[1:]-self.left_hand_pose[:-1])**2)+\
        torch.mean((self.right_hand_pose[1:]-self.right_hand_pose[:-1])**2)+\
        torch.mean((self.pred_trans[1:]-self.pred_trans[:-1])**2)+torch.mean((self.glo_pose[1:]-self.glo_pose[:-1])**2)
    def smooth_acc(self):
        return self.vel_consistency_loss(self.pred_pose)+ self.vel_consistency_loss(self.left_hand_pose)+self.vel_consistency_loss(self.right_hand_pose)+self.vel_consistency_loss(self.pred_trans)+self.vel_consistency_loss(self.glo_pose)
    def forward_human(self):
        smpl_output = self.smpl_model(body_pose=self.pred_pose[:, :],
            global_orient=self.glo_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10),
            transl=self.pred_trans,)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    # def gmof(self,x, sigma):
    
    #     x_squared = x ** 2
    #     sigma_squared = sigma ** 2
    #     return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # insert
                # if self.joint_nums ==52:
                #     pred_markers = joints[:,self.djoints_index] 
                # else:
                pred_markers = joints[:,:self.joint_nums] 
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                loss=loss1#+loss2
                loss.backward()
                return loss
            cam_t_optimizer.step(closure)
    def foot_skate(self,jtr,left_static,right_static):
        left_foot = jtr[:, 10]
        right_foot = jtr[:, 11]
        if left_static.any():
            loss_left = torch.sum((((left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]])[left_static]) ** 2))
        else:
            loss_left = 0
        if right_static.any():
            loss_right = torch.sum((((right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]])[right_static]) ** 2))
        else:
            loss_right = 0
        return loss_left+loss_right
            
    def beta_restrict(self):
        return torch.sum(self.pred_betas**2) + torch.sum((self.pred_betas[1:]-self.pred_betas[:-1])**2)
    def optimize_whole(self,markers_gt):
        body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        left_foot = markers_gt[:, 10]
        right_foot = markers_gt[:, 11]
        delta_left = torch.norm(left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]], dim=1) + 1e-6
        delta_right = torch.norm(right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]], dim=1) + 1e-6
        left_static = (delta_left < 0.01)
        right_static = (delta_right < 0.01)
        
        for i in tqdm(range(100)):
            def closure():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # if self.joint_nums == 52:
                    
                #     pred_markers = joints[:,self.djoints_index]
                # else:
                pred_markers = joints[:,:self.joint_nums]
                    
                loss1=600*(torch.sum((pred_markers-markers_gt)**2))
                loss2=0.001*self.smooth()+0.01*self.smooth_acc()
                loss3=5*self.foot_skate(pred_markers,left_static,right_static)
                loss4=5*self.beta_restrict()
                loss5=1*torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+self.prior.forward(self.pred_pose) #torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
                        # self.prior.forward(self.pred_pose)
                
                loss=loss1+loss5+loss4+loss2+loss3#+loss2+loss3+loss5
                loss.backward()
                #print(loss.shape)
                return loss

            body_optimizer.step(closure)
        with torch.no_grad():
            verts,joints=self.forward_human()
        #     return verts.detach(), self.smpl_model.faces.astype(np.int32)
            
        return verts.detach(), self.smpl_model.faces.astype(np.int32),torch.cat([self.glo_pose,self.pred_pose,self.left_hand_pose,self.right_hand_pose],-1).detach().cpu().numpy(),self.pred_betas.detach().cpu().numpy(),self.pred_trans.detach().cpu().numpy()
        


     
        # with torch.no_grad():
        #     verts,joints=self.forward_human()
        #     return verts.detach(), self.smpl_model.faces.astype(np.int32)
            
        
        

    def forward(self,markers_gt):
        self.init_guess(markers_gt)
        self.optimize_cam(markers_gt)
        return self.optimize_whole(markers_gt)




class SmplhOptmize10_fulljoints2s(nn.Module):
    def __init__(self, gender, batch_size, frame_times,extra=[],joint_nums=52):
        device=torch.device('cuda:0')
        super(SmplhOptmize10_fulljoints2s, self).__init__()
        self.extra=[]
        self.joint_nums = joint_nums
        self.joint_nums = 52
        self.smpl_model = smplh10[gender]
        frame_times = 1
        self.batch_size = batch_size 
        # self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
        # self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)

        # #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
        # self.pred_pose.requires_grad=True
        # self.djoints_index =list(range(22))+list(range(25,55)) 


        # self.pred_betas = Variable(torch.tensor(np.zeros((batch_size, 10))).float().to(device),requires_grad=True)
        # self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        # self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        # self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        # self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='./assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            # print(verts.shape,markers.shape)
            H=(torch.sum((joints[:,[1,2,16,17]]-markers[:,[1,2,16,17]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
        # print(self.pred_trans.shape)


            
        # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
        #                                     lr=1e-2, line_search_fn='strong_wolfe')
        #self.optimizer= torch.optim.Adam([self.pred_trans,self.pred_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], lr=0.01)
            

    def ankle_loss(self):
        return torch.sum(torch.exp(self.pred_pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=self.pred_pose.device)) ** 2)
    def smooth(self):
        return torch.sum((self.pred_pose[1:]-self.pred_pose[:-1])**2)+torch.sum((self.left_hand_pose[1:]-self.left_hand_pose[:-1])**2)+\
        torch.sum((self.right_hand_pose[1:]-self.right_hand_pose[:-1])**2)+\
        torch.sum((self.pred_trans[1:]-self.pred_trans[:-1])**2)+torch.sum((self.glo_pose[1:]-self.glo_pose[:-1])**2)
    def forward_human(self):
        smpl_output = self.smpl_model(body_pose=self.pred_pose[:, :],
            global_orient=self.glo_pose,
            left_hand_pose=self.left_hand_pose,
            right_hand_pose=self.right_hand_pose,
            betas=self.pred_betas[:,None].repeat(1,self.frame_times,1).reshape(-1,10),
            transl=self.pred_trans,)
        verts = smpl_output.vertices
        joints = smpl_output.joints
        # print(joints.shape,'JOINTS')
        return verts,joints
    # def gmof(self,x, sigma):
    
    #     x_squared = x ** 2
    #     sigma_squared = sigma ** 2
    #     return (sigma_squared * x_squared) / (sigma_squared + x_squared)

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # insert
                # if self.joint_nums ==52:
                #     pred_markers = joints[:,self.djoints_index] 
                # else:
                pred_markers = joints[:,:self.joint_nums] 
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                loss=loss1#+loss2
                loss.backward()
                return loss
            cam_t_optimizer.step(closure)
            
    def beta_restrict(self):
        return torch.sum(self.pred_betas**2)
    def optimize_whole(self,markers_gt,i):
        if i==0:
            # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose], max_iter=100,
            #                                     lr=1e-2, line_search_fn='strong_wolfe')
            body_optimizer = torch.optim.Adam([self.pred_trans,self.pred_pose,self.glo_pose,self.pred_betas,self.left_hand_pose,self.right_hand_pose],
                                                lr=1e-2)       
        else:
            
            # body_optimizer = torch.optim.LBFGS([self.pred_trans,self.pred_pose,self.glo_pose,self.left_hand_pose,self.right_hand_pose], max_iter=100,
            #                                     lr=1e-2, line_search_fn='strong_wolfe')
            body_optimizer = torch.optim.Adam([self.pred_trans,self.pred_pose,self.glo_pose,self.left_hand_pose,self.right_hand_pose],
                                                lr=1e-2)
        
        for i in tqdm(range(100)):
            def closure():
                body_optimizer.zero_grad()
                verts,joints=self.forward_human()
                # if self.joint_nums == 52:
                    
                #     pred_markers = joints[:,self.djoints_index]
                # else:
                pred_markers = joints[:,:self.joint_nums]
                    
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                # loss2=5*self.smooth()
                # loss3=5*self.ankle_loss()
                loss4=3*self.beta_restrict()
                loss5=1*torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+1*self.prior.forward(self.pred_pose) #torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
                        # self.prior.forward(self.pred_pose)
                
                loss=loss1+loss5+loss4#+loss2+loss3+loss5
                loss.backward()
                #print(loss.shape)
                return loss

            body_optimizer.step(closure)
        with torch.no_grad():
            verts,joints=self.forward_human()
        #     return verts.detach(), self.smpl_model.faces.astype(np.int32)
            
        return verts.detach(), self.smpl_model.faces.astype(np.int32),torch.cat([self.glo_pose,self.pred_pose,self.left_hand_pose,self.right_hand_pose],-1),self.pred_betas,self.pred_trans
        


     
        # with torch.no_grad():
        #     verts,joints=self.forward_human()
        #     return verts.detach(), self.smpl_model.faces.astype(np.int32)
            
        
        

    def forward(self,markers_gt):
        # B,T,N,3
        batch_size,T,K,_ = markers_gt.shape
        Poses_all = torch.tensor(np.zeros((batch_size,T, 52*3))).float().to(device)
        betas_all = torch.tensor(np.zeros((batch_size,10))).float().to(device)
        trans_all = torch.tensor(np.zeros((batch_size,T,3))).float().to(device)
        
        for i in tqdm(range(T)):
            frame_times =1 
            # break
            if i!=0:
                Poses_all_u = Poses_all.reshape(batch_size,T,52,3)
                self.pred_pose = Variable(torch.tensor(Poses_all_u[:,i,1:22].reshape(-1,63),requires_grad=True))
                self.glo_pose = Variable(torch.tensor(Poses_all_u[:,i,:1].reshape(-1,3),requires_grad=True))

                #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
                self.pred_pose.requires_grad=True
                self.djoints_index =list(range(22))+list(range(25,55)) 


                self.pred_betas = Variable(betas_all,requires_grad=True)
                self.pred_trans = Variable(trans_all[:,i,],requires_grad=True)
                self.left_hand_pose = Variable(Poses_all_u[:,i,-30:-15].reshape(-1,45),requires_grad=True)
                self.right_hand_pose = Variable(Poses_all_u[:,i,-15:].reshape(-1,45),requires_grad=True)
                self.frame_times = frame_times
            else:
                self.pred_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 63))).float().to(device),requires_grad=True)
                self.glo_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)

                #self.pred_pose =torch.tensor(np.zeros((frame_times, 63))).float().to(device)
                self.pred_pose.requires_grad=True
                self.djoints_index =list(range(22))+list(range(25,55)) 


                self.pred_betas = Variable(torch.tensor(np.zeros((batch_size, 10))).float().to(device),requires_grad=True)
                self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
                self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
                self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
                self.frame_times = frame_times
                
                
            self.init_guess(markers_gt[:,i])
            self.optimize_cam(markers_gt[:,i])
            _,_,poses,betas,trans = self.optimize_whole(markers_gt[:,i],i)
            Poses_all[:,i] = poses.detach()
            if i==0:
                betas_all[:,:] = betas.detach()
            trans_all[:,i] = trans.detach()
            
        return Poses_all.detach().cpu().numpy(),betas_all.detach().cpu().numpy(),trans_all.detach().cpu().numpy()


