import torch
import torch.nn.functional as F

from pytorch3d.transforms import *
from utils.markerset import *
from utils.normals import find_closest_indices_vectorized



def cont6d_to_matrix_torch(cont6d):
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)

    y = torch.cross(z, x, dim=-1)

    x = x[..., None]
    y = y[..., None]
    z = z[..., None]

    mat = torch.cat([x, y, z], dim=-1)
    return mat

def contact_detect_torch(verts, obj_points):
    # verts: B x T x M X 3 
    # obj_points: B x T x N x 3
    # return B x T x M
    contact = verts[:, :, :, None, :] - obj_points[:, :, None, :, :]
    contact = torch.norm(contact, dim=-1)
    contact, _ = torch.min(contact, dim=-1)
    return contact

def normal_to_rotation_matrix(marker_normals):
    # Input: marker_normals: BxTx77x3
    # return: rotation_matrix: BxTx77x3x3
    # compute the normal vector

    # compute the rotation matrix
    origin = torch.cat([torch.zeros_like(marker_normals[...,:1]), torch.ones_like(marker_normals[...,:1]),torch.zeros_like(marker_normals[...,:1])], dim=-1)
    axis = torch.cross(origin, marker_normals)
    non_zero = torch.norm(axis, dim=-1) > 1e-6
    axis[non_zero] = axis[non_zero] / torch.norm(axis[non_zero], dim=-1, keepdim=True)
    angle = torch.sum(marker_normals * origin, dim=-1,keepdim=True) / torch.norm(marker_normals, dim=-1, keepdim=True) / torch.norm(origin, dim=-1, keepdim=True)
    angle = torch.arccos(angle)
    rotvec = axis * angle
    rotation_matrix = axis_angle_to_matrix(rotvec)
    
    return rotation_matrix

def relative_representation(markers, rotation_matrix, obj_rot6D, obj_trans):
    # Input: ori_normal: BxTx77x3, rotation_matrix: BxTx77x3x3, obj_rot6D: BxTx6
    # return: obj_rot6D: BxTx77x6, obj_trans: BxTx77x3

    # apply the rotation matrix to the normal
    trans_delta = obj_trans[...,None,:] - markers
    rel_obj_trans = torch.matmul(rotation_matrix, trans_delta[...,None]).squeeze(-1)
    obj_matrix = rotation_6d_to_matrix(obj_rot6D)
    rel_obj_matrix = torch.matmul(rotation_matrix, obj_matrix[...,None,:,:])
    rel_obj_rot6D = matrix_to_rotation_6d(rel_obj_matrix)


    return rel_obj_rot6D, rel_obj_trans

def global_representation(marker, rotation_matrix, obj_rot6D, obj_trans):
    # Input: marker: BxTx77x3, rotation_matrix: BxTx77x3x3, obj_rot6D: BxTx77x6
    # return: obj_rot6D: BxTx77x6, obj_trans: BxTx77x3

    glo_obj_trans = torch.matmul(rotation_matrix, obj_trans[...,None]).squeeze(-1) + marker
    rel_obj_matrix = rotation_6d_to_matrix(obj_rot6D)
    glo_obj_matrix = torch.matmul(rotation_matrix, rel_obj_matrix)
    glo_obj_rot6D = matrix_to_rotation_6d(glo_obj_matrix)

    return glo_obj_rot6D, glo_obj_trans

def weight_obj_1(C, O):
    # C: B x T x 78
    # O: B x T x 78 x 9
    B, T, N, _ = O.shape

    # Compute max(|C_i|) for each B, T over 78 (features)
    max_abs_C = torch.max(torch.abs(C), dim=-1, keepdim=True)[0]  # Shape: [B, T, 1]

    # Calculate the difference between max(|C_i|) and |C_j|
    diff = max_abs_C - torch.abs(C)  # Shape: [B, T, 78]

    # Compute the softmax of diff along the 78 dimension (axis=2)
    weights = torch.softmax(diff, dim=2)  # Shape: [B, T, 78]

    # Compute the weighted sum
    weighted_sum = torch.sum(weights.unsqueeze(-1) * O, dim=2)  # Shape: [B, T, 9]

    return weighted_sum


def weight_obj(C, O):
    # C: B x T x 78
    # O: B x T x 78 x 9
    B, T, N, _ = O.shape

    # Compute max(|C_i|) for each B, T over 78 (features)
    max_abs_C = torch.max(torch.abs(C), dim=-1, keepdim=True)[0]  # Shape: [B, T, 1]

    # Calculate the difference between max(|C_i|) and |C_j|
    diff = max_abs_C - torch.abs(C)  # Shape: [B, T, 78]

    # Compute the softmax of diff along the 78 dimension (axis=2)
    weights = torch.softmax(diff**4, dim=2)  # Shape: [B, T, 78]

    # Compute the weighted sum
    weighted_sum = torch.sum(weights.unsqueeze(-1) * O, dim=2)  # Shape: [B, T, 9]

    return weights, weighted_sum



class Guide_Contact:
    def __init__(self,
                 classifiler_scale=10.0,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 ):

        self.classifiler_scale = classifiler_scale
        self.n_joints = 77
        self.sigmoid = torch.nn.Sigmoid()


        self.loss_all = []


    def __call__(self, x, t, y=None, human_mean=None): # *args, **kwds):

        loss, grad, loss_list, new_x = self.gradients(x, t, y['obj_points'], y['obj_normals'])

            
        return loss, grad, loss_list, new_x

    def gradients(self, x, t, obj_points, obj_normals):
        with torch.enable_grad():
            n_joints = 77 
            sample = x.permute(0, 2, 3, 1).squeeze()

            B, T , _ = sample.shape

            pred_markers = sample[...,:231].reshape(B, T, n_joints, 3)
            pred_obj_data = sample[...,476:485]
            pred_obj_relative = sample[...,494:962].reshape(B, T, 78, 6)
            pred_rel_obj_marker = pred_obj_relative[...,:3]
            pred_rel_obj_points_trans = pred_obj_relative[...,3:6]
            

            pred_obj_rot6D = pred_obj_data[...,:6]
            pred_obj_trans = pred_obj_data[...,None,6:9] # B x T x 1 x 3
            pred_obj_trans[...,1] = 0

            all_markers = torch.cat((pred_markers, pred_obj_trans), dim=2) # B x T x 78 x 3

            # find canonical points indices
            pred_rel_obj_points_indices = find_closest_indices_vectorized(pred_rel_obj_points_trans, obj_points) # B x T x 78 
            pred_rel_obj_points_indices = pred_rel_obj_points_indices.unsqueeze(-1).expand(-1, -1, -1, 3) # B x T x 78 x 3

            # get points coordinate based on translation
            pred_rot_matrix = rotation_6d_to_matrix(pred_obj_rot6D) # B x T x 3 x 3
            obj_points = obj_points[:, None, ...].repeat(1,T,1,1).float()
            rot_obj_points = torch.matmul(obj_points, pred_rot_matrix.transpose(2, 3)) # B x T x N x 3
            selected_rel_obj_points = torch.gather(rot_obj_points, 2, pred_rel_obj_points_indices) # B * T * 78 * 3
            
            pred_rel_obj_trans = all_markers + pred_rel_obj_marker - selected_rel_obj_points

            # calculate contact distance
            pred_contact_distance = torch.norm(pred_rel_obj_marker, dim=-1) # B * T * 78

            weights, pred_obj_trans = weight_obj(pred_contact_distance, pred_rel_obj_trans) # B * T * 3

            x[:,482:485] = pred_obj_trans.reshape(B, T, 3, 1).permute(0,2,3,1)

            # guide rotation
            x.requires_grad_(True)
            sample = x.permute(0, 2, 3, 1).squeeze()
            pred_markers = sample[...,:231].reshape(B, T, n_joints, 3)
            pred_obj_trans[...,1] = 0
            all_markers = torch.cat((pred_markers, pred_obj_trans[...,None,:]), dim=2) # B x T x 78 x 3
            pred_obj_rot6D = sample[...,476:482]
            pred_obj_data = sample[...,476:485]

            # rotate object
            pred_rot_matrix = rotation_6d_to_matrix(pred_obj_rot6D) # B x T x 3 x 3
            rot_obj_points = torch.matmul(obj_points, pred_rot_matrix.transpose(2, 3)) + pred_obj_trans[..., None, :]  # B x T x N x 3
            selected_obj_points = torch.gather(rot_obj_points, 2, pred_rel_obj_points_indices) # B * T * 78 * 3
           
            loss_contact = torch.norm((all_markers + pred_rel_obj_marker - selected_obj_points), dim=-1) * weights
        
            loss_sum = loss_contact.sum() 

            
            loss_smooth_obj = F.mse_loss(pred_obj_data[:, 1:, :], pred_obj_data[:, :-1, :]) * 500.0
            
            loss_sum += loss_smooth_obj

            self.loss_all.append(loss_sum)
            grad = torch.autograd.grad([loss_sum], [x])[0]

        return loss_sum, grad, self.loss_all, x.detach()

class Guide_Contactv2:
    def __init__(self,
                 classifiler_scale=10.0,
                 guidance_style='xstart',
                 stop_cond_from=0,
                 ):

        self.classifiler_scale = classifiler_scale
        self.n_joints = 77
        self.sigmoid = torch.nn.Sigmoid()


        self.loss_all = []


    def __call__(self, x, t, y=None, human_mean=None): # *args, **kwds):

        loss, grad, loss_list, new_x = self.gradients(x, t, y['obj_points'], y['obj_normals'])

            
        return loss, grad, loss_list, new_x

    def gradients(self, x, t, obj_points, obj_normals):
        with torch.enable_grad():
            n_joints = 77 
            x.requires_grad_(True)

            sample = x.permute(0, 2, 3, 1).squeeze()

            B, T , _ = sample.shape

            pred_markers = sample[...,:231].reshape(B, T, n_joints, 3)
            pred_obj_data = sample[...,476:485]
            pred_obj_relative = sample[...,494:962].reshape(B, T, 78, 6).detach()
            pred_rel_obj_marker = pred_obj_relative[...,:3]
            pred_rel_obj_points_trans = pred_obj_relative[...,3:6]
            

            pred_obj_rot6D = pred_obj_data[...,:6]
            pred_obj_trans = pred_obj_data[...,6:9]
            pred_ground_marker = pred_obj_data[...,None,6:9].detach().clone() # B x T x 1 x 3
            pred_ground_marker[...,1] = 0

            pred_all_markers = torch.cat((pred_markers, pred_ground_marker), dim=2) # B x T x 78 x 3

            # find canonical points indices
            pred_rel_obj_points_indices = find_closest_indices_vectorized(pred_rel_obj_points_trans, obj_points) # B x T x 78 
            pred_rel_obj_points_indices = pred_rel_obj_points_indices.unsqueeze(-1).expand(-1, -1, -1, 3) # B x T x 78 x 3

            # get points coordinate based on translation
            pred_rot_matrix = rotation_6d_to_matrix(pred_obj_rot6D) # B x T x 3 x 3
            obj_points = obj_points[:, None, ...].repeat(1,T,1,1).float()
            rot_obj_points = torch.matmul(obj_points, pred_rot_matrix.transpose(2, 3)) + pred_obj_trans[..., None, :]  # B x T x N x 3
            selected_rel_obj_points = torch.gather(rot_obj_points, 2, pred_rel_obj_points_indices) # B * T * 78 * 3
            
            # calculate contact distance
            pred_contact_distance = torch.norm(pred_rel_obj_marker, dim=-1) # B * T * 78

            # x[:,482:485] = pred_obj_trans.reshape(B, T, 3, 1).permute(0,2,3,1)
           
            loss_contact = torch.norm((pred_all_markers + pred_rel_obj_marker - selected_rel_obj_points), dim=-1, keepdim=True) # B * T * 78 * 1

            _, loss_contact = weight_obj(pred_contact_distance, loss_contact) # B * T * 1

            loss_sum = loss_contact.sum() 

            
            # loss_smooth_obj = F.mse_loss(pred_obj_data[:, 1:, :], pred_obj_data[:, :-1, :]) * 500.0
            
            # loss_sum += loss_smooth_obj

            self.loss_all.append(loss_sum)
            grad = torch.autograd.grad([loss_sum], [x])[0]

        return loss_sum, grad, self.loss_all, x.detach()
