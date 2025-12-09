import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG


# https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2/models/pointnet2_msg_sem.py
class PointNet2Encoder(nn.Module):
    """
    c_in: input point feature dimension exculding xyz
    """
    def __init__(self, c_in=6, c_out=128, num_keypoints=256):
        super(PointNet2Encoder, self).__init__()
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024, 
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=True,
            )
        )
        c_out_0 = 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=num_keypoints,  # 256
                radii=[0.1, 0.2],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=True,
            )
        )
        c_out_1 = 128 + 128

        self.num_keypoints = num_keypoints
        self.c_out = c_out
        self.Linear = nn.Linear(c_out_1, c_out - 3)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud):
        r"""
            Forward pass of the network
            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        # B, P, C = pointcloud.shape
        # pointcloud = pointcloud.reshape(B*I, P, C)
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        # print(l_xyz[-1].shape, l_features[-1].shape)
        local_keypoints = torch.cat((l_xyz[-1],
                                     self.Linear(l_features[-1].transpose(1, 2))), dim=-1)  # B*I x Pb x C
        return local_keypoints #.reshape(B, I, self.num_keypoints, self.c_out)
 