import torch
import torch.nn as nn
import torch as th
from utils.utils import *
# import chamfer_pytorch.dist_chamfer as ext
from utils.rotation_conversions import axis_angle_to_matrix
# distChamfer = ext.chamferDist()

from data_loaders.behave.utils.plot_script import plot_3d_motion
import data_loaders.behave.utils.paramUtil as paramUtil

# import ChamferDistancePytorch.dist_chamfer_idx  as ext
# distChamfer = ext.chamferDist()

kinematic_chain = [[0, 2, 5, 8, 11],
                 [0, 1, 4, 7, 10],
                 [0, 3, 6, 9, 12, 15],
                 [9, 14, 17, 19, 21],
                 [9, 13, 16, 18, 20]]


l_idx1, l_idx2 = 17, 18
# Right/Left foot
fid_r, fid_l = [14, 15], [19, 20]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [11, 16, 5, 8]
# l_hip, r_hip
r_hip, l_hip = 11, 16

# humanml3d_kinematic_tree = [
#     [0, 3, 6, 9, 12, 15],  # body
#     [9, 14, 17, 19, 21],  # right arm
#     [9, 13, 16, 18, 20],  # left arm
#     [0, 2, 5, 8, 11],  # right leg
#     [0, 1, 4, 7, 10],
# ]  # left leg

class InterLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(InterLoss, self).__init__()
        self.nb_joints = nb_joints
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.normalizer = MotionNormalizerTorch()

        self.weights = {}
        self.weights["RO"] = 0.1
        self.weights["JA"] = 3
        self.weights["DM"] = 3
        self.weights["CON"] = 10 

        self.skeleton = paramUtil.t2m_kinematic_chain

        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        else:
    
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]

        if contact_mask is not None:
            loss = (self.Loss(prediction, target) * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2, -3)) / (mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, motion_pred, motion_gt, mask, timestep_mask, y):

        #  [B,T,1,D]
        B, T = motion_pred.shape[:2]
        self.losses["simple"] = self.seq_masked_mse(motion_pred[...,:816], motion_gt[...,:816], mask)

        self.losses["simple_obj"] = self.seq_masked_mse(motion_pred[...,816:822], motion_gt[...,816:822], mask) * 0.1

        # self.contact_labels = y["contact_labels"]
        # self.obj_points = y["obj_points"]
        # # self.contact_labels = contact_labels.argmax(dim=-1)




        target = self.normalizer.backward(motion_gt, global_rt=True, dim=822)
        prediction = self.normalizer.backward(motion_pred, global_rt=True, dim=822)


        self.pred_h_joints = prediction[..., :self.nb_joints * 3].reshape(B, T, -1, self.nb_joints, 3)
        self.tgt_h_joints = target[..., :self.nb_joints * 3].reshape(B, T, -1, self.nb_joints, 3)

        self.pred_o_pos = prediction[..., 816 + 3 : 816 + 6 ].reshape(B, T, -1, 1, 3)
        self.tgt_o_pos = target[..., 816 + 3 : 816 + 6].reshape(B, T, -1, 1, 3)




        self.pred_o_rot = prediction[..., 816: 816 + 3 ].reshape(B, T, -1, 1, 3).squeeze(2)
        self.tgt_o_rot = target[..., 816 : 816 + 3].reshape(B, T, -1, 1, 3).squeeze(2)


        self.mask = mask
        self.timestep_mask = timestep_mask


        # self.forward_distance_map(thresh=1.0)
        # self.forward_contact()
        # self.forward_joint_affinity(thresh=0.1)
        self.forward_relatvie_rot()
        self.accum_loss()


#     def forward_contact(self):

#         B, T, _, _, _ = self.pred_h_joints.shape
#         device = self.pred_h_joints.device

#         pred_joints_output = self.pred_h_joints.reshape(B, T,  self.nb_joints, 3)
#         pred_obj_output = torch.cat([self.pred_o_rot.reshape(B, T, 3), self.pred_o_pos.reshape(B, T, 3)], dim=-1)


#         tgt_joints_output = self.tgt_h_joints.reshape(B, T,  self.nb_joints, 3)
#         tgt_obj_output = torch.cat([self.tgt_o_rot.reshape(B, T, 3), self.tgt_o_pos.reshape(B, T, 3)], dim=-1)



#         h_contact_labels = self.contact_labels[:,:,:22]
#         o_contact_labels = self.contact_labels[:,:,22:]
        
        
#         batch_size = pred_joints_output.size(0)
#         all_loss_joints_contact = 0
#         all_loss_object_contact = 0
#         all_loss_h_collision = 0
#         all_loss_o_collision = 0

#         # # contacts_joints = (sample[:, 0,:, 269:269+22]> 0.5).int()
#         # # contacts_object = (sample[:, 0,:, 269+22:]> 0.5).int()
#         # print(f"======= {h_contact_labels}")
#         target_mask1 = torch.zeros_like(pred_joints_output, dtype=torch.bool)
#         target_mask1[torch.where(h_contact_labels == 1.0)] = True 
#         target_mask1 = target_mask1[...,0]

#         # target_mask2 = torch.zeros([B, T , 512], dtype=torch.bool).to(device)
#         # target_mask2[torch.where(o_contact_labels == 1.0)] = True 
#         # target_mask2 = target_mask2[...,0]


#         h_contact_dist = torch.zeros(0).to(device)
#         h_contact_dist2 = torch.zeros(0).to(device)
#         # o_contact_dist = torch.zeros(0).to(device)

#         # all_pred_points = torch.zeros(0).to(device)
#         # all_loss_contact  = torch.zeros(0).to(device)

#         batch_size = pred_joints_output.shape[0]
#         for i in range(batch_size):
#             # transform
#             vertices = self.obj_points[i]
#             center = torch.mean(vertices, 0)
#             vertices -= center
#             pred_angle, pred_trans = pred_obj_output[i, :, :3].transpose(1,0), pred_obj_output[i, :, 3:].transpose(1,0)

#             pred_rot = axis_angle_to_matrix(pred_angle.transpose(1,0))
#             pred_points = torch.matmul(vertices.unsqueeze(0), pred_rot.permute(0, 2, 1)) + pred_trans.transpose(1, 0).unsqueeze(1)
#             # pred_points.requires_grad_(True)


#             tgt_angle, tgt_trans = tgt_obj_output[i, :, :3].transpose(1,0), tgt_obj_output[i, :, 3:].transpose(1,0)

#             tgt_rot = axis_angle_to_matrix(tgt_angle.transpose(1,0))
#             tgt_points = torch.matmul(vertices.unsqueeze(0),tgt_rot.permute(0, 2, 1)) + tgt_trans.transpose(1, 0).unsqueeze(1)

            
#             # all_pred_points = torch.cat([all_pred_points, pred_points.unsqueeze(0)]) 
#             # print(f" gt: {tgt_joints_output.squeeze()[i].shape}, {tgt_points.shape}")
#             # plot_3d_motion('./test{}.mp4'.format(i), self.skeleton, tgt_joints_output.squeeze()[i].detach().cpu().numpy(), tgt_points.detach().cpu().numpy(),  None, None, title="test", fps=20)


#             ##########################################################


#             dist_chamfer_contact = ext.chamferDist()

#             o_contact_idx = torch.where(o_contact_labels[i] == 1.0)
#             h_contact_idx = torch.where(h_contact_labels[i] == 1.0)

#             if len(o_contact_idx[1]>0):
#                 contact_dist1, _ = dist_chamfer_contact(pred_joints_output[i].float().contiguous(),
#                                                                 pred_points.float().contiguous())
#             else:
#                 contact_dist1, _ = dist_chamfer_contact(pred_joints_output[i].float().contiguous(),
#                                                                 pred_points.float().contiguous())

#             if len(o_contact_idx[1]>0):
#                 contact_dist2, _ = dist_chamfer_contact(tgt_joints_output[i].float().contiguous(),
#                                                                 tgt_points.float().contiguous())
#             else:
#                 contact_dist2, _ = dist_chamfer_contact(tgt_joints_output[i].float().contiguous(),
#                                                                 tgt_points.float().contiguous())

#             contact_dist1 = torch.mean(torch.sqrt(contact_dist1 + 1e-4) / (torch.sqrt(contact_dist1 + 1e-4) + 1.0)) 

#             contact_dist2 = torch.mean(torch.sqrt(contact_dist2 + 1e-4) / (torch.sqrt(contact_dist2 + 1e-4) + 1.0)) 



#             h_contact_dist = torch.cat([h_contact_dist, contact_dist1.unsqueeze(0)], dim=0)
#             h_contact_dist2 = torch.cat([h_contact_dist2, contact_dist2.unsqueeze(0)], dim=0)

#         # loss_sum = F.mse_loss(h_contact_dist , torch.zeros_like(h_contact_dist),
#         #                             reduction='none') * target_mask1 

#         # loss_sum2 = F.mse_loss(h_contact_dist2 , torch.zeros_like(h_contact_dist),
#         #                         reduction='none') * target_mask1 

#         loss = h_contact_dist.sum() / batch_size

#         loss = (loss * self.timestep_mask).sum(dim=0) / (self.timestep_mask.sum(dim=0) + 1.e-7)

#         loss2 = h_contact_dist2.sum() / batch_size
#         loss2 = (loss2 * self.timestep_mask).sum(dim=0) / (self.timestep_mask.sum(dim=0) + 1.e-7)
 
#         # loss = loss_sum.mean(dim=-1, keepdim=True).unsqueeze(-1)

#         # loss2 = loss_sum2.mean(dim=-1, keepdim=True).unsqueeze(-1)



#         # loss = (loss * self.mask).sum(dim=(-1, -2, -3)) / (self.mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
#         # loss = (loss * self.timestep_mask).sum(dim=0) / (self.timestep_mask.sum(dim=0) + 1.e-7)




#         # loss2 = (loss2 * self.mask).sum(dim=(-1, -2, -3)) / (self.mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
#         # loss2 = (loss2 * self.timestep_mask).sum(dim=0) / (self.timestep_mask.sum(dim=0) + 1.e-7)

# # 
#         print(f" loss_pred : {loss}   tgt   {loss2} ")

#         # self.losses['CON'] = loss * self.weights["CON"]    

#         # multi_joints = False
#         # if not multi_joints:
#         # loss_sum = loss_sum.sum() / target_mask1.sum() * batch_size


    def forward_relatvie_rot(self):
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
        across = self.pred_h_joints[..., r_hip, :] - self.pred_h_joints[..., l_hip, :]
        across = across / across.norm(dim=-1, keepdim=True)
        
        across_gt = self.tgt_h_joints[..., r_hip, :] - self.tgt_h_joints[..., l_hip, :]
        across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

        y_axis = torch.zeros_like(across)
        y_axis[..., 1] = 1

        forward = torch.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward_gt = torch.cross(y_axis, across_gt, axis=-1)
        forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)


        pred_relative_rot = qbetween(forward[..., 0, :],  self.pred_o_rot[..., 0, :])
        tgt_relative_rot = qbetween(forward_gt[..., 0, :],  self.tgt_o_rot[..., 0, :])

        # self.losses["RO"] = self.mix_masked_mse(pred_relative_rot[..., [0, 2]],
        #                                                     tgt_relative_rot[..., [0, 2]],
        #                                                     self.mask[..., 0, :], self.timestep_mask) * self.weights["RO"]

        
        self.losses["RO"] = self.mix_masked_mse(pred_relative_rot[..., [0, 1, 2]],
                                                            tgt_relative_rot[..., [0, 1, 2]],
                                                            self.mask[..., 0, :], self.timestep_mask) * self.weights["RO"]    
                                          


    def forward_distance_map(self, thresh):

        pred_h_joints = self.pred_h_joints.reshape(self.mask.shape[:-1] + (-1,))
        tgt_h_joints = self.tgt_h_joints.reshape(self.mask.shape[:-1] + (-1,))

        pred_o_pos = self.pred_o_pos.reshape(self.mask.shape[:-1] + (-1,))
        tgt_o_pos = self.tgt_o_pos.reshape(self.mask.shape[:-1] + (-1,))


    
        pred_h_joints = pred_h_joints.reshape(-1, self.nb_joints, 3)   # [6272, 22, 3]
        pred_o_pos = pred_o_pos.reshape(-1, 1, 3)
        tgt_h_joints = tgt_h_joints.reshape(-1, self.nb_joints, 3)
        tgt_o_pos = tgt_o_pos.reshape(-1, 1, 3)

        pred_distance_matrix = torch.cdist(pred_h_joints.contiguous(), pred_o_pos.contiguous()).reshape(
            self.mask.shape[:-2] + (1, -1,))

        tgt_distance_matrix = torch.cdist(tgt_h_joints.contiguous(), tgt_o_pos.contiguous()).reshape(
            self.mask.shape[:-2] + (1, -1,))   # [32, 196, 1, 484]

        # distance_matrix_mask = (pred_distance_matrix < thresh).float()


        self.losses["DM"] = self.mix_masked_mse(pred_distance_matrix, tgt_distance_matrix,
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask) * self.weights["DM"]

    def forward_joint_affinity(self, thresh):
        pred_h_joints = self.pred_h_joints.reshape(self.mask.shape[:-1] + (-1,))
        tgt_h_joints = self.tgt_h_joints.reshape(self.mask.shape[:-1] + (-1,))

        pred_o_pos = self.pred_o_pos.reshape(self.mask.shape[:-1] + (-1,))
        tgt_o_pos = self.tgt_o_pos.reshape(self.mask.shape[:-1] + (-1,))

        pred_h_joints = pred_h_joints.reshape(-1, self.nb_joints, 3)   # [6272, 22, 3]
        pred_o_pos = pred_o_pos.reshape(-1, 1, 3)
        tgt_h_joints = tgt_h_joints.reshape(-1, self.nb_joints, 3)
        tgt_o_pos = tgt_o_pos.reshape(-1, 1, 3)

        pred_distance_matrix = torch.cdist(pred_h_joints.contiguous(), pred_o_pos.contiguous()).reshape(
            self.mask.shape[:-2] + (1, -1,))

        tgt_distance_matrix = torch.cdist(tgt_h_joints.contiguous(), tgt_o_pos.contiguous()).reshape(
            self.mask.shape[:-2] + (1, -1,))   # [32, 196, 1, 484]

        distance_matrix_mask = (tgt_distance_matrix < thresh).float()

        self.losses["JA"] = self.mix_masked_mse(pred_distance_matrix, torch.zeros_like(tgt_distance_matrix),
                                                                self.mask[..., 0:1, :],
                                                                self.timestep_mask, dm_mask=distance_matrix_mask) * self.weights["JA"]

    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses["total"] = loss
        return self.losses


class SingleLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints):
        super(SingleLoss, self).__init__()
        self.nb_joints = nb_joints
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.normalizer = MotionNormalizerTorch()


        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        else:
    
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (self.Loss(prediction, target) * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2, -3)) / (mask.sum(dim=(-1, -2, -3)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, motion_pred, motion_gt, mask, timestep_mask):

        #  [B,T,1,D]
        B, T = motion_pred.shape[:2]
        self.losses["simple"] = self.seq_masked_mse(motion_pred[...,:816], motion_gt[...,:816], mask)

        self.mask = mask
        self.timestep_mask = timestep_mask
        self.accum_loss()


    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses["total"] = loss
        return self.losses



class GeometricLoss(nn.Module):
    def __init__(self, recons_loss, nb_joints, name):
        super(GeometricLoss, self).__init__()
        self.name = name
        self.nb_joints = nb_joints
        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss(reduction='none')
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss(reduction='none')
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss(reduction='none')

        self.normalizer = MotionNormalizerTorch()
        self.fids = [7, 10, 8, 11]

        self.weights = {}
        self.weights["VEL"] = 30
        self.weights["BL"] = 10
        self.weights["FC"] = 30
        self.weights["POSE"] = 1
        self.weights["TR"] = 100

        self.losses = {}

    def seq_masked_mse(self, prediction, target, mask):
        loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)
        loss = (loss * mask).sum() / (mask.sum() + 1.e-7)
        return loss

    def mix_masked_mse(self, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
        if dm_mask is not None:
            loss = (self.Loss(prediction, target) * dm_mask).sum(dim=-1, keepdim=True)/ (dm_mask.sum(dim=-1, keepdim=True) + 1.e-7)  # [b,t,p,4,1]
        else:
            loss = self.Loss(prediction, target).mean(dim=-1, keepdim=True)  # [b,t,p,4,1]
        if contact_mask is not None:
            loss = (loss[..., 0] * contact_mask).sum(dim=-1, keepdim=True) / (contact_mask.sum(dim=-1, keepdim=True) + 1.e-7)
        loss = (loss * mask).sum(dim=(-1, -2)) / (mask.sum(dim=(-1, -2)) + 1.e-7)  # [b]
        loss = (loss * batch_mask).sum(dim=0) / (batch_mask.sum(dim=0) + 1.e-7)

        return loss

    def forward(self, motion_pred, motion_gt, mask, timestep_mask):
        B, T = motion_pred.shape[:2]
        # self.losses["simple"] = self.seq_masked_mse(motion_pred, motion_gt, mask)  # * 0.01
        target = self.normalizer.backward(motion_gt, global_rt=True, dim=816)
        prediction = self.normalizer.backward(motion_pred, global_rt=True, dim=816)

        self.first_motion_pred =motion_pred[:,0:1]
        self.first_motion_gt =motion_gt[:,0:1]

        self.pred_g_joints = prediction[..., :self.nb_joints * 3].reshape(B, T, self.nb_joints, 3)
        self.tgt_g_joints = target[..., :self.nb_joints * 3].reshape(B, T, self.nb_joints, 3)
        self.mask = mask
        self.timestep_mask = timestep_mask

        self.forward_vel()
        self.forward_bone_length()
        self.forward_contact()
        self.accum_loss()
        # return self.losses["simple"]

    def get_local_positions(self, positions, r_rot):
        '''Local pose'''
        positions[..., 0] -= positions[..., 0:1, 0]
        positions[..., 2] -= positions[..., 0:1, 2]
        '''All pose face Z+'''
        positions = qrot(r_rot[..., None, :].repeat(1, 1, positions.shape[-2], 1), positions)
        return positions

    def forward_local_pose(self):
        r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

        pred_g_joints = self.pred_g_joints.clone()
        tgt_g_joints = self.tgt_g_joints.clone()

        across = pred_g_joints[..., r_hip, :] - pred_g_joints[..., l_hip, :]
        across = across / across.norm(dim=-1, keepdim=True)
        across_gt = tgt_g_joints[..., r_hip, :] - tgt_g_joints[..., l_hip, :]
        across_gt = across_gt / across_gt.norm(dim=-1, keepdim=True)

        y_axis = torch.zeros_like(across)
        y_axis[..., 1] = 1

        forward = torch.cross(y_axis, across, axis=-1)
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward_gt = torch.cross(y_axis, across_gt, axis=-1)
        forward_gt = forward_gt / forward_gt.norm(dim=-1, keepdim=True)

        z_axis = torch.zeros_like(forward)
        z_axis[..., 2] = 1
        noise = torch.randn_like(z_axis) *0.0001
        z_axis = z_axis+noise
        z_axis = z_axis / z_axis.norm(dim=-1, keepdim=True)


        pred_rot = qbetween(forward, z_axis)
        tgt_rot = qbetween(forward_gt, z_axis)

        B, T, J, D = self.pred_g_joints.shape
        pred_joints = self.get_local_positions(pred_g_joints, pred_rot).reshape(B, T, -1)
        tgt_joints = self.get_local_positions(tgt_g_joints, tgt_rot).reshape(B, T, -1)

        self.losses["POSE_"+self.name] = self.mix_masked_mse(pred_joints, tgt_joints, self.mask, self.timestep_mask) * self.weights["POSE"]

    def forward_vel(self):
        B, T = self.pred_g_joints.shape[:2]

        pred_vel = self.pred_g_joints[:, 1:] - self.pred_g_joints[:, :-1]
        tgt_vel = self.tgt_g_joints[:, 1:] - self.tgt_g_joints[:, :-1]

        pred_vel = pred_vel.reshape(pred_vel.shape[:-2] + (-1,))
        tgt_vel = tgt_vel.reshape(tgt_vel.shape[:-2] + (-1,))

        # print(f"VEL_:{pred_vel.shape}  {tgt_vel.shape}   {self.mask.shape}")


        self.losses["VEL_"+self.name] = self.mix_masked_mse(pred_vel, tgt_vel, self.mask[:, :-1], self.timestep_mask) * self.weights["VEL"]


    def forward_contact(self):

        feet_vel = self.pred_g_joints[:, 1:, self.fids, :] - self.pred_g_joints[:, :-1, self.fids,:]
        feet_h = self.pred_g_joints[:, :-1, self.fids, 1]
        # contact = target[:,:-1,:,-8:-4] # [b,t,p,4]

        contact = self.foot_detect(feet_vel, feet_h, 0.001)

        # print(f"FC_:{self.mix_masked_mse(feet_vel, torch.zeros_like(feet_vel), self.mask[:, :-1],self.timestep_mask,contact)}")

        self.losses["FC_"+self.name] = self.mix_masked_mse(feet_vel, torch.zeros_like(feet_vel), self.mask[:, :-1],
                                                          self.timestep_mask,
                                                          contact) * self.weights["FC"]



    def forward_bone_length(self):
        pred_g_joints = self.pred_g_joints
        tgt_g_joints = self.tgt_g_joints
        pred_bones = []
        tgt_bones = []
        for chain in kinematic_chain:
            for i, joint in enumerate(chain[:-1]):
                pred_bone = (pred_g_joints[..., chain[i], :] - pred_g_joints[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                         keepdim=True)  # [B,T,P,1]
                tgt_bone = (tgt_g_joints[..., chain[i], :] - tgt_g_joints[..., chain[i + 1], :]).norm(dim=-1,
                                                                                                      keepdim=True)
                pred_bones.append(pred_bone)
                tgt_bones.append(tgt_bone)

        pred_bones = torch.cat(pred_bones, dim=-1)
        tgt_bones = torch.cat(tgt_bones, dim=-1)

        # print(f"BL_:{self.mix_masked_mse(pred_bones, tgt_bones, self.mask, self.timestep_mask)}")

        self.losses["BL_"+self.name] = self.mix_masked_mse(pred_bones, tgt_bones, self.mask, self.timestep_mask) * self.weights[
            "BL"]


    def forward_traj(self):
        B, T = self.pred_g_joints.shape[:2]

        pred_traj = self.pred_g_joints[..., 0, [0, 2]]
        tgt_g_traj = self.tgt_g_joints[..., 0, [0, 2]]

        self.losses["TR_"+self.name] = self.mix_masked_mse(pred_traj, tgt_g_traj, self.mask, self.timestep_mask) * self.weights["TR"]


    def accum_loss(self):
        loss = 0
        for term in self.losses.keys():
            loss += self.losses[term]
        self.losses[self.name] = loss

    def foot_detect(self, feet_vel, feet_h, thres):
        velfactor, heightfactor = torch.Tensor([thres, thres, thres, thres]).to(feet_vel.device), torch.Tensor(
            [0.12, 0.05, 0.12, 0.05]).to(feet_vel.device)

        feet_x = (feet_vel[..., 0]) ** 2
        feet_y = (feet_vel[..., 1]) ** 2
        feet_z = (feet_vel[..., 2]) ** 2

        contact = (((feet_x + feet_y + feet_z) < velfactor) & (feet_h < heightfactor)).float()
        return contact



def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
        return_vector=False,
):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    # ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = distChamfer(x,y)


    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near  # y point to x
    y2x = y - y_near  # x point to y

    if x_normals is not None:
        print(f"======== {yidx_near_expanded.shape}  ")
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = th.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = th.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    if not return_vector:
        return y2x_signed, x2y_signed, yidx_near, xidx_near
    else:
        return y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y