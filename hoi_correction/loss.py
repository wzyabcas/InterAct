import pytorch3d.loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch3d.ops import cot_laplacian
from pytorch3d.structures import Meshes
from human_body_prior.tools import tgm_conversion as tgm
import chamfer_distance as chd
#import kaolin


# def intersection( sbj_verts, obj_verts, sbj_faces, obj_faces, do_sbj2obj,do_obj2sbj,full_body=True, adjacency_matrix=None):
#     """
#     Compute intersection penalty between body and object (or obstacle) given vertices and normals of both.

#     :param sbj_verts                (torch.Tensor) on device - (bs, N_sbj, 3)
#     :param obj_verts                (torch.Tensor) on device - (1, N_obj, 3)
#     :param sbj_faces                (torch.Tensor) on device - (F_sbj, 3)
#     :param obj_faces                (torch.Tensor) on device - (F_obj, 3)
#     :param full_body                (bool) -- for full-body if True; else for rhand
#     :param adjacency_matrix         (optional)

#     :return penet_loss_batched_in   (torch.Tensor) - (bs,) - loss values for each batch element - penetration
#     :return penet_loss_batched_out  (torch.Tensor) - (bs,) - loss values for each batch element - outside
#     """
#     device = sbj_verts.device
#     bs = sbj_verts.shape[0]
#     #obj_verts = obj_verts.repeat(bs, 1, 1)                                                                               # (bs, N_obj, 3)
#     num_obj_verts, num_sbj_verts = obj_verts.shape[1], sbj_verts.shape[1]
#     penet_loss_batched_in, penet_loss_batched_out = torch.zeros(bs).to(device), torch.zeros(bs).to(device)
#     thresh = 0.00

#     # (*) Object to subject.
#     if do_obj2sbj or 0:
#         # 1. Use Kaolin to calculate sign (True if inside, False if outside)
#         sign = kaolin.ops.mesh.check_sign(sbj_verts, sbj_faces, obj_verts)                                               # (bs, N_obj)
#         ones = torch.ones_like(sign.long())                                                                              # (bs, N_obj)
#         # 2. Negative for penetration, Positive for outside (to keep consistent with previous format).
#         sign = torch.where(sign, -ones, ones)                                                                            # (bs, N_obj)
#         # 3. Calculate absolute distance of points from mesh, and multiply by sign.
#         face_vertices = kaolin.ops.mesh.index_vertices_by_faces(sbj_verts, sbj_faces)                                    # (bs, F_sbj, 3, 3)
#         dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(obj_verts.contiguous(), face_vertices)           # (bs, N_obj)
#         obj2sbj = dist * sign                                                                                            # (bs, N_obj)
#         # 4. Average across batch for negative and positive values.
#         zeros_o2s, ones_o2s = torch.zeros_like(obj2sbj).to(device), torch.ones_like(obj2sbj).to(device)
#         loss_o2s_in = torch.sum(abs(torch.where(obj2sbj<thresh, obj2sbj-thresh, zeros_o2s)), 1) / num_obj_verts          # (bs,) -- averaged across (bs, N_obj)
#         loss_o2s_out = torch.sum(torch.log(torch.where(obj2sbj>thresh, obj2sbj+ones_o2s, ones_o2s)), 1) / num_obj_verts  # (bs,) -- averaged across (bs, N_obj)
#         # 5. Add to final loss.
#         penet_loss_batched_in += loss_o2s_in
#         penet_loss_batched_out += loss_o2s_out
    

#     # (*) Subject to object.
#     if do_sbj2obj:
#         # 0. Simplify obstacle faces - many have determinant ~0, i.e., it is a degenerate triangle.
#         # NOTE: No need to do for all elements in batch because faces are the same, so just repeat.
        
#         face_vertices = kaolin.ops.mesh.index_vertices_by_faces(obj_verts, obj_faces)
#         #print(obj_faces.shape,face_vertices.shape)                                 # (bs, F_obj, 3, 3)
#         #indices_good_faces = :#(face_vertices[0].det().abs() > 0.001)
#         # print(face_vertices[0].det().abs()[:20])
#         # print(indices_good_faces.shape,'INDICES')
#         # print(torch.sum(indices_good_faces),'GOOD')                                                      # (F_obj)
#         obj_faces = obj_faces[:]
#         #face_vertices = face_vertices[0][:][None].repeat(bs, 1, 1, 1)
#                                            # (bs, F_obj_good, 3, 3)
#         # 1. Use Kaolin to calculate sign (True if inside, False if outside)
#         #print(obj_faces.shape,obj_verts.shape, sbj_verts.shape,'KKKKK')
#         sign = kaolin.ops.mesh.check_sign(obj_verts, obj_faces, sbj_verts)                                               # (bs, N_sbj)
#         ones = torch.ones_like(sign.long())                                                                              # (bs, N_sbj)
#         # 2. Negative for penetration, Positive for outside (to keep consistent with previous format).
#         sign = torch.where(sign, -ones, ones)                                                                          # (bs, N_sbj)
#         # 3. Calculate absolute distance of points from mesh, and multiply by sign.
#         dist, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(sbj_verts.contiguous(), face_vertices)           # (bs, N_sbj)
#         M=torch.sum(dist<0)
#         if torch.sum(M)>0:
#             print(dist[M],'ILL')

#         sbj2obj = torch.sqrt(dist) * sign                                                                                            # (bs, N_sbj)
#         # 4. Average across batch for negative and positive values.
#         #zeros_s2o, ones_s2o = torch.zeros_like(sbj2obj).to(device), torch.ones_like(sbj2obj).to(device)
#         # loss_s2o_out = torch.sum(torch.log(torch.where(sbj2obj>thresh, sbj2obj+ones_s2o, ones_s2o)), 1) / num_sbj_verts  # (bs,)  -- averaged across (bs, N_sbj)
#         # # 4.1. Special case for sbj2obj negative values - check whether to do connected components or not.
#         # loss_s2o_in = torch.sum(abs(torch.where(sbj2obj<thresh, sbj2obj-thresh, zeros_s2o)), 1) / num_sbj_verts          # (bs,)  -- averaged across (bs, N_sbj)
#         # if full_body and self.cfg.obstacle_sbj2obj_extra == 'connected_components' and loss_s2o_in.mean() > 0:
#         #     # Connected components based loss.
#         #     edges = np.stack(np.where(adjacency_matrix))
#         #     num_nodes = adjacency_matrix.shape[0]
#         #     v_to_edges = torch.zeros((num_nodes, edges.shape[1]))
#         #     v_to_edges[edges[0], range(edges.shape[1])] = 1
#         #     v_to_edges[edges[1], range(edges.shape[1])] = 1

#         #     indices_inter = (sbj2obj < thresh)
#         #     v_to_edges = v_to_edges[None].expand(bs, -1, -1).clone()
#         #     v_to_edges[torch.where(indices_inter)] = 0
#         #     edges_indices = v_to_edges.sum(1) == 2
#         #     num_inter_v = indices_inter.sum(-1)

#         #     for i in range(bs):
#         #         if loss_s2o_in[i] > 0:
#         #             edges_i = edges[:, edges_indices[i]]
#         #             adj = pytorch_geometric.to_scipy_sparse_matrix(edges_i, num_nodes=num_nodes)
#         #             n_components, labels = sp.csgraph.connected_components(adj)

#         #             n_components -= num_inter_v[i]  # Inside obstacles are not taken into account
#         #             if n_components > 1:
#         #                 indices_out = torch.ones([num_sbj_verts])
#         #                 indices_out[indices_inter[i]] = 0
#         #                 labels_ = labels[indices_out.bool()]
#         #                 # We penalize only the vertices that are out, but the penalization is wrt the original
#         #                 # edge, not including the threshold.
#         #                 most_common_label = Counter(labels_).most_common()[0][0]
#         #                 penalized_joints = (labels != most_common_label) * indices_out.bool().numpy()
#         #                 loss_s2o_in[i] += sbj2obj[i][penalized_joints].sum() / num_sbj_verts

#         # 5. Add to final loss.
#         # penet_loss_batched_in += loss_s2o_in
#         # penet_loss_batched_out += loss_s2o_out

#     # (*) Return final.
#     return sbj2obj
    #return penet_loss_batched_in, penet_loss_batched_out

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

    # print(y.shape,'Y')
    # print(x.shape,'X')

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = ch_dist(x,y,x_normals=x_normals,y_normals=y_normals)

    xidx_near_expanded = xidx_near.view(N, P1, 1).contiguous().expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).contiguous().expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near  # y point to x
    y2x = y - y_near  # x point to y

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.contiguous().view(-1, 1, 3), y2x.contiguous().view(-1, 3, 1)).contiguous().view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.contiguous().view(-1, 1, 3), x2y.contiguous().view(-1, 3, 1)).contiguous().view(N, -1).contiguous().sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    if not return_vector:
        return y2x_signed, x2y_signed, yidx_near, xidx_near
    else:
        return y2x_signed, x2y_signed, yidx_near_expanded, xidx_near_expanded, y2x, x2y

def aa2matrot(pose):
    '''
    :param Nx3
    :return: pose_matrot: Nx3x3
    '''
    bs = pose.size(0)
    num_joints = pose.size(1)//3
    pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose)[:, :3, :3].contiguous()#.view(bs, num_joints*9)
    return pose_body_matrot

def rotvec_to_rotmat(rotvec):
    rotmat = aa2matrot(rotvec.contiguous().view(-1, 3)).view(-1, 3, 3)
    return rotmat 

def mesh_laplacian_smoothing(meshes, method: str = "uniform"):
    r"""
    Computes the laplacian smoothing objective for a batch of meshes.
    This function supports three variants of Laplacian smoothing,
    namely with uniform weights("uniform"), with cotangent weights ("cot"),
    and cotangent curvature ("cotcurv").For more details read [1, 2].

    Args:
        meshes: Meshes object with a batch of meshes.
        method: str specifying the method for the laplacian.
    Returns:
        loss: Average laplacian smoothing loss across the batch.
        Returns 0 if meshes contains no meshes or all empty meshes.

    Consider a mesh M = (V, F), with verts of shape Nx3 and faces of shape Mx3.
    The Laplacian matrix L is a NxN tensor such that LV gives a tensor of vectors:
    for a uniform Laplacian, LuV[i] points to the centroid of its neighboring
    vertices, a cotangent Laplacian LcV[i] is known to be an approximation of
    the surface normal, while the curvature variant LckV[i] scales the normals
    by the discrete mean curvature. For vertex i, assume S[i] is the set of
    neighboring vertices to i, a_ij and b_ij are the "outside" angles in the
    two triangles connecting vertex v_i and its neighboring vertex v_j
    for j in S[i], as seen in the diagram below.

    .. code-block:: python

               a_ij
                /\
               /  \
              /    \
             /      \
        v_i /________\ v_j
            \        /
             \      /
              \    /
               \  /
                \/
               b_ij

        The definition of the Laplacian is LV[i] = sum_j w_ij (v_j - v_i)
        For the uniform variant,    w_ij = 1 / |S[i]|
        For the cotangent variant,
            w_ij = (cot a_ij + cot b_ij) / (sum_k cot a_ik + cot b_ik)
        For the cotangent curvature, w_ij = (cot a_ij + cot b_ij) / (4 A[i])
        where A[i] is the sum of the areas of all triangles containing vertex v_i.

    There is a nice trigonometry identity to compute cotangents. Consider a triangle
    with side lengths A, B, C and angles a, b, c.

    .. code-block:: python

               c
              /|\
             / | \
            /  |  \
         B /  H|   \ A
          /    |    \
         /     |     \
        /a_____|_____b\
               C

        Then cot a = (B^2 + C^2 - A^2) / 4 * area
        We know that area = CH/2, and by the law of cosines we have

        A^2 = B^2 + C^2 - 2BC cos a => B^2 + C^2 - A^2 = 2BC cos a

        Putting these together, we get:

        B^2 + C^2 - A^2     2BC cos a
        _______________  =  _________ = (B/H) cos a = cos a / sin a = cot a
           4 * area            2CH


    [1] Desbrun et al, "Implicit fairing of irregular meshes using diffusion
    and curvature flow", SIGGRAPH 1999.

    [2] Nealan et al, "Laplacian Mesh Optimization", Graphite 2006.
    """

    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
    num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
    verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
    weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
    weights = 1.0 / weights.float()

    # We don't want to backprop through the computation of the Laplacian;
    # just treat it as a magic constant matrix that is used to transform
    # verts into normals
    with torch.no_grad():
        if method == "uniform":
            L = meshes.laplacian_packed()
        elif method in ["cot", "cotcurv"]:
            L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            if method == "cot":
                norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                idx = norm_w > 0
                norm_w[idx] = 1.0 / norm_w[idx]
            else:
                L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
                norm_w = 0.25 * inv_areas
        else:
            raise ValueError("Method should be one of {uniform, cot, cotcurv}")

    if method == "uniform":
        laplacian = L.mm(verts_packed)
    elif method == "cot":
        laplacian = L.mm(verts_packed) * norm_w - verts_packed
    elif method == "cotcurv":
        # pyre-fixme[61]: `norm_w` may not be initialized here.
        laplacian = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
    curvature = laplacian.norm(p=2, dim=1)

    return curvature

class LaplacianLoss(nn.Module):
    # faces: BxFx3
    def __init__(self, faces):
        super(LaplacianLoss, self).__init__()
        self.faces = faces
        self.criterion = nn.L1Loss(reduction='mean')

    # x,y: BxVx3
    def forward(self, x, y):
        batch_size = x.shape[0]
        mesh_x = Meshes(
            verts=x,
            faces=self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        )

        mesh_y = Meshes(
            verts=y,
            faces=self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        )

        curvature_x = mesh_laplacian_smoothing(mesh_x, method='cotcurv')
        # curvature_y = mesh_laplacian_smoothing(mesh_y, method='cotcurv')

        # loss = self.criterion(curvature_x, curvature_y)
        loss = curvature_x.mean()

        return loss

class NormalConsistencyLoss(nn.Module):
    # faces: BxFx3
    def __init__(self, faces):
        super(NormalConsistencyLoss, self).__init__()
        self.faces = faces

    # x,y: BxVx3
    def forward(self, x):
        batch_size = x.shape[0]
        mesh_x = Meshes(
            verts=x,
            faces=self.faces.unsqueeze(0).expand(batch_size, -1, -1)
        )

        return pytorch3d.loss.mesh_normal_consistency(mesh_x)

# https://github.com/hongsukchoi/Pose2Mesh_RELEASE/blob/master/lib/core/loss.py
class NormalVectorLoss(nn.Module):
    # face: Fx3
    def __init__(self, face):
        super(NormalVectorLoss, self).__init__()
        self.face = face

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        v1_out = coord_out[:, face[:, 1], :] - coord_out[:, face[:, 0], :]
        v1_out = F.normalize(v1_out, p=2, dim=2)  # L2 normalize to make unit vector
        v2_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 0], :]
        v2_out = F.normalize(v2_out, p=2, dim=2)  # L2 normalize to make unit vector
        v3_out = coord_out[:, face[:, 2], :] - coord_out[:, face[:, 1], :]
        v3_out = F.normalize(v3_out, p=2, dim=2)  # L2 nroamlize to make unit vector

        v1_gt = coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 0], :]
        v1_gt = F.normalize(v1_gt, p=2, dim=2)  # L2 normalize to make unit vector
        v2_gt = coord_gt[:, face[:, 2], :] - coord_gt[:, face[:, 0], :]
        v2_gt = F.normalize(v2_gt, p=2, dim=2)  # L2 normalize to make unit vector
        normal_gt = torch.cross(v1_gt, v2_gt, dim=2)
        normal_gt = F.normalize(normal_gt, p=2, dim=2)  # L2 normalize to make unit vector

        cos1 = torch.abs(torch.sum(v1_out * normal_gt, 2, keepdim=True))
        cos2 = torch.abs(torch.sum(v2_out * normal_gt, 2, keepdim=True))
        cos3 = torch.abs(torch.sum(v3_out * normal_gt, 2, keepdim=True))
        loss = torch.cat((cos1, cos2, cos3), 1)
        return loss.mean()

epsilon = 1e-16
class EdgeLengthLoss(nn.Module):
    def __init__(self, face, relative_length=False):
        super(EdgeLengthLoss, self).__init__()
        self.face = face
        self.relative_length = relative_length

    def forward(self, coord_out, coord_gt):
        face = torch.LongTensor(self.face).cuda()

        d1_out = torch.sqrt(epsilon +
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(epsilon +
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(epsilon +
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(epsilon + torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(epsilon + torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(epsilon + torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        if self.relative_length:
            diff1 = diff1 / d1_gt
            diff2 = diff2 / d2_gt
            diff3 = diff3 / d3_gt
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()