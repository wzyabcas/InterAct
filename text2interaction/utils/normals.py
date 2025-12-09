import pytorch3d.loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops import cot_laplacian
from pytorch3d.structures import Meshes
from human_body_prior.tools import tgm_conversion as tgm
import chamfer_distance as chd

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(0, faces[:, 1].long(),
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                   vertices_faces[:, 0] - vertices_faces[:, 1]))
    normals.index_add_(0, faces[:, 2].long(),
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                   vertices_faces[:, 1] - vertices_faces[:, 2]))
    normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                   vertices_faces[:, 2] - vertices_faces[:, 0]))

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals

# def find_closest_indices_vectorized(pred_rel_obj_trans, obj_points):
#     # Get the shape of the inputs
#     B, T, num_points, _ = pred_rel_obj_trans.shape  # B*T*78*3
#     N = obj_points.shape[1]  # B*N, the number of points in the point cloud
    
#     # Compute the Euclidean distance matrix (B, T, 78, N)
#     # pred_rel_obj_trans has shape (B, T, 78, 3)
#     # obj_points has shape (B, N, 3)
    
#     # Broadcasting to compute pairwise differences between pred_rel_obj_trans points and obj_points
#     # Adding extra dimensions to enable broadcasting:
#     # pred_rel_obj_trans[:, :, :, None, :] expands to (B, T, 78, 1, 3)
#     # obj_points[:, None, None, :, :] expands to (B, 1, 1, N, 3)
#     # The result of the subtraction is (B, T, 78, N, 3), representing the differences between each point.
#     diff = pred_rel_obj_trans[:, :, :, None, :] - obj_points[:, None, None, :, :]  # (B, T, 78, N, 3)
    
#     # Compute the Euclidean distance by taking the norm of the difference along the last dimension (x, y, z)
#     dist = torch.norm(diff, dim=-1)  # (B, T, 78, N) - distances between each predicted point and each point cloud point
    
#     # Find the index of the minimum distance along the last axis (N dimension)
#     indices = torch.argmin(dist, dim=-1)  # (B, T, 78) - index of the closest point in the point cloud for each predicted point
    
#     return indices  # B*T*78 matrix of indices

def find_closest_indices_vectorized(pred_rel_obj_trans, obj_points):
    pred_rel_obj_trans = pred_rel_obj_trans.float()
    obj_points = obj_points.float()
    # Get shapes
    B, T, num_points, _ = pred_rel_obj_trans.shape
    N = obj_points.shape[1]
    
    # Compute squared Euclidean distances directly, without explicit broadcasting of each tensor
    # Expand dimensions only where necessary and perform the subtraction along the expanded dimensions
    pred_squared = (pred_rel_obj_trans ** 2).sum(dim=-1, keepdim=True)  # (B, T, 78, 1)
    obj_squared = (obj_points ** 2).sum(dim=-1).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
    
    # Compute distances squared and reduce memory usage by avoiding an explicit diff tensor
    # Equivalent to ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a * b
    dist_squared = pred_squared + obj_squared - 2 * torch.einsum("btpc,bnc->btpn", pred_rel_obj_trans, obj_points)
    
    # Find the index of the minimum squared distance
    indices = torch.argmin(dist_squared, dim=-1)  # (B, T, 78)
    
    return indices  # Returns the indices of the closest point in obj_points for each point in pred_rel_obj_trans