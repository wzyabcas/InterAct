import torch
import torch.nn as nn
from model.points_encoder import PointNet2Encoder
from model.mdm import MDM
from model.mdm import *
from pytorch3d.transforms import rotation_6d_to_matrix
from diffusion.fp16_util import convert_module_to_f16
def get_representation(positions, positions_obj, obj_points):
    # positions: B x T x N*3 
    # positions_obj: B x T x 6
    # obj_points: B x M x 3
    # return B x T x N

    # Right/Left foot
    fid_r, fid_l = [61, 52, 53, 40, 34, 49, 40], [29, 30, 18, 19, 7, 2, 15]
    B, T, _ = positions.shape
    _, M, _ = obj_points.shape
    positions = positions.reshape(B, T, -1, 3)
    obj_points = obj_points[:, None, :, :].repeat(1,T,1,1)

    obj_angles = positions_obj[...,:6].reshape(-1,6).float()
    obj_trans = positions_obj[...,6:].reshape(-1,3).float()
    rot = rotation_6d_to_matrix(obj_angles).float()
    obj_points = torch.bmm(obj_points.reshape(-1,M,3), rot.transpose(1, 2)) + obj_trans[:, None, :]
    
    obj_points = obj_points.reshape(B,T,M,3)

    velocity = (positions[:,1: ] - positions[:,:-1])
    velocity_obj = (positions_obj[:,1: ] - positions_obj[:,:-1])
    # pad the last velocity using torch
    velocity = torch.cat([velocity, torch.zeros((B,1,velocity.shape[-2],velocity.shape[-1])).cuda()], axis=1)
    velocity_obj = torch.cat([velocity_obj, torch.zeros((B,1,velocity_obj.shape[-1])).cuda()], axis=1)

    def interaction_aware_torch(distance, omega=5.0):
        return torch.exp(-omega * distance)
    """ Get Foot Contacts """

    feet_l = positions[...,fid_l,1] 

    feet_r = positions[...,fid_r,1] 

    def contact_detect_torch(verts, obj_points):
        # verts: B x T x N X 3 
        # obj_points: B x T x M x 3
        # return B x T x N
        contact = verts[:, :, :, None, :] - obj_points[:, :, None, :, :]
        contact = torch.norm(contact, dim=-1)
        contact, _ = torch.min(contact, dim=-1)
        return contact
    
    contact_aware = contact_detect_torch(positions, obj_points)

    velocity = velocity.reshape(B, T, -1)

    positions = positions.reshape(B, T, -1)

    data = torch.cat([positions, velocity, feet_l, feet_r, contact_aware, positions_obj, velocity_obj], axis=-1)
    return data[:, None, :, :].permute(0, 3, 1, 2).float()

def get_repre(positions_data, positions_obj_data):
    # positions: B x T x N*3 
    # positions_obj: B x T x 9
    # return B x T x N

    # Right/Left foot
    fid_r, fid_l = [61, 52, 53, 40, 34, 49, 40], [29, 30, 18, 19, 7, 2, 15]
    B, T, _ = positions_data.shape

    positions = positions_data[...,:231].reshape(B, T, -1, 3)
    
    positions_obj = positions_obj_data[...,:9]


    feet_l = positions[...,fid_l,1] 

    feet_r = positions[...,fid_r,1] 


    velocity = (positions[:,1: ] - positions[:,:-1])
    velocity_obj = (positions_obj[:,1: ] - positions_obj[:,:-1])
    # pad the last velocity using torch
    velocity = torch.cat([velocity, torch.zeros((B,1,velocity.shape[-2],velocity.shape[-1])).cuda()], axis=1)
    velocity_obj = torch.cat([velocity_obj, torch.zeros((B,1,velocity_obj.shape[-1])).cuda()], axis=1)

    velocity = velocity.reshape(B, T, -1)

    positions = positions.reshape(B, T, -1)

    data = torch.cat([positions, velocity, feet_l, feet_r, positions_obj, velocity_obj, positions_obj_data[...,18:]], axis=-1)
    return data[:, None, :, :].permute(0, 3, 1, 2).float()

class HOIDiff(MDM):
    def __init__(self,modeltype, njoints, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, clip_version=None, args=None, **kargs):
        super(HOIDiff, self).__init__(modeltype, njoints-18-78*6, nfeats, num_actions, translation, pose_rep, glob, glob_rot,
                         latent_dim, ff_size, num_layers, num_heads, dropout,
                         ablation, activation, legacy, data_rep, dataset, clip_dim,
                         arch, emb_trans_dec, clip_version, **kargs)
        
        self.args = args
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=1024*3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=latent_dim),
            )
        self.pointnet_encoder = PointNet2Encoder(c_in=1, c_out=self.latent_dim, num_keypoints=1) 

        if self.arch == 'trans_enc':

            # print(f"  {self.args.multi_backbone_split}  {self.num_layers} ")
            assert 0 < self.args.multi_backbone_split <= self.num_layers
            print(f'CUTTING BACKBONE AT LAYER [{self.args.multi_backbone_split}]')
            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)
            del self.seqTransEncoder
            self.seqTransEncoder_start = nn.TransformerEncoder(seqTransEncoderLayer,
                                                               num_layers=self.args.multi_backbone_split)
            self.seqTransEncoder_end = nn.TransformerEncoder(seqTransEncoderLayer,
                                                             num_layers= self.num_layers - self.args.multi_backbone_split)
        else:
            raise ValueError('Supporting only trans_enc arch.')



        seqTransEncoderLayer_obj_pose = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)



        self.seqTransEncoder_obj_pose_start = nn.TransformerEncoder(seqTransEncoderLayer_obj_pose,
                                                         num_layers=self.args.multi_backbone_split)

        self.seqTransEncoder_obj_pose_end = nn.TransformerEncoder(seqTransEncoderLayer_obj_pose,
                                                         num_layers=self.num_layers - self.args.multi_backbone_split)

        self.mutual_attn = MutualAttention(num_layers=2,
                                    latent_dim=self.latent_dim,
                                    input_feats=self.input_feats
                                    )


        self.input_process_obj = InputProcess(self.data_rep, 18 + 78*6, self.latent_dim)

        self.output_process_obj = OutputProcess(self.data_rep, 18 + 78*6, self.latent_dim, 18 + 78*6,
                                            self.nfeats)

 
    def mask_cond_obj(self, cond, force_mask=False):
        seq, bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)

        else:
            return cond 
    def convert_to_fp16(self):
        for module in self.modules():
            # Use the helper from your fp16_util (provided above) to convert supported modules.
            
            convert_module_to_f16(module)


    def encode_obj(self, obj_bps):
        # obj_points - [bs, n_points, 3]
        obj_points = obj_bps.view(-1, 1024*3)
        obj_emb = self.bps_encoder(obj_points) # [bs, d]
        # [1, bs, d]
        return obj_emb.unsqueeze(0)
    

    def forward(self, x, timesteps, y=None):
        
        """
        x: [batch_size, njoints, nfeats, max_frames], denoted x_t in the paper
        timesteps: [batch_size] (int)
        """

        x_human, x_obj = x[:,:476], x[:,476:]
        # x_human, x_obj = x[:,:263], x[:,263:]
        


        # Build embedding vector
        emb = self.embed_timestep(timesteps)  # [1, bs, d]
        emb += self.encode_obj(y['obj_bps'].float())

        force_mask = y.get('uncond', False)
        if 'text' in self.cond_mode:
            enc_text = self.encode_text(y['text'])
            emb += self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
        if 'action' in self.cond_mode:
            action_emb = self.embed_action(y['action'])
            emb += self.mask_cond(action_emb, force_mask=force_mask)

       
        
        x_human = self.input_process(x_human)
        x_obj =  self.input_process_obj(x_obj)

        

        xseq_human = torch.cat((emb, x_human), axis=0)  # [seqlen+1, bs, d]
        xseq_human = self.sequence_pos_encoder(xseq_human)  # [seqlen+1, bs, d]
        human_mid = self.seqTransEncoder_start(xseq_human)

        xseq_obj = torch.cat((emb, x_obj), axis=0)
        xseq_obj = self.sequence_pos_encoder(xseq_obj)
        # obj_mid = self.seqTransEncoder_obj_pose_start(xseq_obj)

        obj_mid = self.seqTransEncoder_obj_pose_start(xseq_obj)


        if self.args.multi_backbone_split < self.num_layers:
            dec_output_human, dec_output_obj = self.mutual_attn(human_mid[1:], obj_mid[1:])
            output_human = self.seqTransEncoder_end(torch.cat([human_mid[:1], dec_output_human], 0))[1:]
            output_obj = self.seqTransEncoder_obj_pose_end(torch.cat([obj_mid[:1], dec_output_obj], 0))[1:]


        output_human = self.output_process(output_human)
        output_obj = self.output_process_obj(output_obj)

        human_data = output_human.permute(0, 2, 3, 1).squeeze().float()
        obj_data = output_obj.permute(0, 2, 3, 1).squeeze().float()


        return get_repre(human_data, obj_data)

    def trainable_parameters(self):
        return [p for name, p in self.named_parameters() if p.requires_grad]


    def freeze_block(self, block):
        block.eval()
        for p in block.parameters():
            p.requires_grad = False



 


class MutualAttention(nn.Module):
    def __init__(self, num_layers, latent_dim, input_feats):
        super().__init__()

        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_heads = 4
        self.ff_size = 1024
        self.dropout = 0.1
        self.activation = 'gelu'
        self.input_feats = input_feats

        seqTransDecoderLayer_obj = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

        seqTransDecoderLayer_human = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

        
        self.seqTransDecoder_human_pose = nn.TransformerDecoder(seqTransDecoderLayer_human,
                                                         num_layers=self.num_layers)

        self.seqTransDecoder_obj_pose = nn.TransformerDecoder(seqTransDecoderLayer_obj,
                                                         num_layers=self.num_layers)



    def forward(self, x_human, x_obj,mask=None):
        dec_output_human = self.seqTransDecoder_human_pose(tgt=x_human, memory=x_obj,memory_key_padding_mask=mask, tgt_key_padding_mask=mask)
        dec_output_obj = self.seqTransDecoder_obj_pose(tgt=x_obj, memory=x_human,memory_key_padding_mask=mask, tgt_key_padding_mask=mask)
        return dec_output_human, dec_output_obj