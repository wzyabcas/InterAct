import numpy as np
import torch
import torch.nn as nn
import os
import smplx
import copy

from tqdm import tqdm
from torch.autograd import Variable

from text2interaction.utils.markerset import markerset_smplh
from text2interaction.sample.prior import HandPrior, Prior


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH = os.path.join(PROJECT_PATH, "models")


######################################## smplh 10 ########################################
smplh_model_male = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="male",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=True).to(device)

smplh_model_female = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="female",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=True).to(device)

smplh_model_neutral = smplx.create(MODEL_PATH, model_type='smplh',
                        gender="neutral",
                        use_pca=False,
                        ext='pkl',flat_hand_mean=True).to(device)

smplh10 = {'male': smplh_model_male,'female':smplh_model_female,'neutral':smplh_model_neutral}


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
        self.hand_prior=HandPrior(prior_path=os.path.join(PROJECT_PATH,'assets'), device=device)
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
    """
    parameters:
        init_pose: initial pose, like rotA
        init_trans: initial translation, like transA
    """
    
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
        self.djoints_index =list(range(22))+list(range(25,55)) # skip 23,24,55,56,57,58

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
        
        
        self.left_hand_pose = Variable((lhand_data).float().to(device),requires_grad=True)     
        self.right_hand_pose = Variable((rhand_data).float().to(device),requires_grad=True)
        
        self.lhand_rec = lhand_data.detach().clone()
        self.rhand_rec = rhand_data.detach().clone()
        # self.left_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        
        # self.right_hand_pose = Variable(torch.tensor(np.zeros((batch_size*frame_times, 45))).float().to(device),requires_grad=True)
        self.frame_times = frame_times
        self.hand_prior=HandPrior(prior_path=os.path.join(PROJECT_PATH,'assets'),device=device)
        self.prior=Prior()

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

    def init_guess(self, markers):
        with torch.no_grad():
            _, joints = self.forward_human()
            idx = [1, 2, 16, 17]
            delta = (markers[:, idx] - joints[:, idx]).mean(dim=1)
            new_trans = self.pred_trans + delta

        self.pred_trans = Variable(copy.deepcopy(new_trans), requires_grad=True)

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
            transl=self.pred_trans,
        )
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
        
    


   