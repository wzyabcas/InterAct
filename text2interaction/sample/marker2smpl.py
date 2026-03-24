import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import smplx
from torch.autograd import Variable
import copy

from sample.prior import *
from utils.markerset import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


MODEL_PATH = '../models'


######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl').to(device)


smplh10 = {'male': smplh_model_male}

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
        self.hand_prior=HandPrior(prior_path='../assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            H=(torch.sum((verts[:,[1861,5322,1058,4544]]-markers[:,[28,60,19,53]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)#torch.tensor(H)
            

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
        return verts,joints

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
                loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
                        self.prior.forward(self.pred_pose)
                
                loss=loss1+loss2+loss3+loss5
                loss.backward()
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

        self.pred_pose.requires_grad=True

        self.pred_betas = Variable(torch.tensor(np.zeros((batch_size, 10))).float().to(device),requires_grad=True)
        self.pred_trans = Variable(torch.tensor(np.zeros((batch_size*frame_times, 3))).float().to(device),requires_grad=True)
        self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path='../assets',device=device)
        self.prior=Prior()

    def init_guess(self, markers):
        with torch.no_grad():
        
            verts,joints=self.forward_human()
            H=(torch.sum((verts[:,[1861,5322,1058,4544]]-markers[:,[28,60,19,53]]),dim=1)/4).float()
        self.pred_trans=Variable(copy.deepcopy(H),requires_grad=True)
            

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
        return verts,joints

    def optimize_cam(self,markers_gt):
        cam_t_optimizer = torch.optim.LBFGS([self.pred_trans,self.glo_pose], max_iter=100,
                                            lr=1e-2, line_search_fn='strong_wolfe')
        for i in tqdm(range(10)):
            def closure():
                cam_t_optimizer.zero_grad()
                verts,joints=self.forward_human()
                pred_markers = verts[:,markerset_smplh]
                loss1=100*(torch.sum((pred_markers-markers_gt)**2))
                loss=loss1
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
                loss4=5*self.beta_restrict()
                loss5=torch.sum(self.hand_prior(self.left_hand_pose,left_or_right=0)**2+self.hand_prior(self.left_hand_pose,left_or_right=1)**2)+\
                        self.prior.forward(self.pred_pose)
                
                loss=loss1+loss2+loss3+loss4+loss5
                loss.backward()
                return loss

            body_optimizer.step(closure)
        with torch.no_grad():
            verts,joints=self.forward_human()
            return verts.detach(), self.smpl_model.faces
            
        
        

    def forward(self,markers_gt):
        self.init_guess(markers_gt)
        self.optimize_cam(markers_gt)
        return self.optimize_whole(markers_gt)
        
    


   