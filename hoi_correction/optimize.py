import os
import numpy as np
import torch
from hoi_correction.utils import vertex_normals
from hoi_correction.loss import point2point_signed
from hoi_correction.prior import *
import trimesh
from pytorch3d.transforms import axis_angle_to_matrix,rotation_6d_to_matrix,matrix_to_rotation_6d
from torch.autograd import Variable
import torch.optim as optim
import copy
import argparse
from tqdm import tqdm
import pickle
import trimesh


from human_body_prior.body_model.body_model import BodyModel

DEVICE_NUMBER=0

device = torch.device(f"cuda:{DEVICE_NUMBER}" if torch.cuda.is_available() else "cpu")

#device=torch.device('cpu')
SMPLX_PATH='./models/smplx'
surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
surface_model_neutral_fname = os.path.join(SMPLX_PATH, "SMPLX_NEUTRAL.npz")

dmpl_fname = None
num_dmpls = None 
num_expressions = None
num_betas = 16 
sbj_m_female = BodyModel(bm_fname=surface_model_female_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname).to(device)
    
sbj_m_male = BodyModel(bm_fname=surface_model_male_fname,
                    num_betas=num_betas,
                    num_expressions=num_expressions,
                    num_dmpls=num_dmpls,
                    dmpl_fname=dmpl_fname).to(device)
sbj_m_all={'male':sbj_m_male,'female':sbj_m_female}

hand_prior=HandPrior(prior_path='./assets',device=device)
hand_distance_init=0


## Rhand id of smplx
rhand_idx=np.load('./assets/smplx_hand_index/rhand_smplx_ids.npy')


WHETHER_TOUCH_LEFT=0
WHETHER_TOUCH_RIGHT=0
WHETHER_MIDTOUCH_LEFT=0
WHETHER_MIDTOUCH_RIGHT=0


## MANO MEAN OF HANDS
rhand_mean=np.load(f'./assets/rhand_mean.npy')
rhand_mean_torch_single=torch.from_numpy(rhand_mean.reshape(1,-1,3)).float().to(device)
lhand_mean=np.load(F'./assets/lhand_mean.npy')
lhand_mean_torch_single=torch.from_numpy(lhand_mean.reshape(1,-1,3)).float().to(device)

RHAND_INDEXES_DETAILED=[]

## FULL HAND INDEX 'hand_778_{i}_{j}.npy': i indicate the finger index, j indicates the knuckle index
for i in range(5):
    for j in range(3):
        RHAND_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/hand_778_{i}_{j}.npy'))
# PALM INDEX
RHAND_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/hand_778_5.npy'))

## HALF HAND INDEX 'hand_778_small_{i}_{j}.npy': HALF of the HAND, inside of the palms's side, used for optimizing contact without introducing severe penetration
## i means specific finger, j indicates the knuckle index
RHAND_SMALL_INDEXES_DETAILED=[]
for i in range(5):
    for j in range(3):
        RHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/hand_778_small_{i}_{j}.npy'))
# HALF PALM
RHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/hand_778_small_5.npy'))

LHAND_INDEXES_DETAILED=[]
for i in range(5):
    for j in range(3):
        LHAND_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/lhand_778_{i}_{j}.npy'))
LHAND_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/lhand_778_5.npy'))

LHAND_SMALL_INDEXES_DETAILED=[]
for i in range(5):
    for j in range(3):
        LHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/lhand_778_small_{i}_{j}.npy'))
LHAND_SMALL_INDEXES_DETAILED.append(np.load(f'./assets/smplx_hand_index/lhand_778_small_5.npy'))


## NOT USED, Finger indexes
RHAND_INDEXES=[]
for i in range(6):
    RHAND_INDEXES.append(np.load(f'./assets/smplx_hand_index/hand_778_{i}.npy'))
RHAND_SMALL_INDEXES=[]
for i in range(6):
    RHAND_SMALL_INDEXES.append(np.load(f'./assets/smplx_hand_index/hand_778_small_{i}.npy'))
LHAND_INDEXES=[]
for i in range(6):
    LHAND_INDEXES.append(np.load(f'./assets/smplx_hand_index/lhand_778_{i}.npy'))
LHAND_SMALL_INDEXES=[]
for i in range(6):
    LHAND_SMALL_INDEXES.append(np.load(f'./assets/smplx_hand_index/lhand_778_small_{i}.npy'))




SIMPLIFIED_MESH = {
    "backpack":"backpack/backpack_f1000.ply",
    'basketball':"basketball/basketball_f1000.ply",
    'boxlarge':"boxlarge/boxlarge_f1000.ply",
    'boxtiny':"boxtiny/boxtiny_f1000.ply",
    'boxlong':"boxlong/boxlong_f1000.ply",
    'boxsmall':"boxsmall/boxsmall_f1000.ply",
    'boxmedium':"boxmedium/boxmedium_f1000.ply",
    'chairblack': "chairblack/chairblack_f2500.ply",
    'chairwood': "chairwood/chairwood_f2500.ply",
    'monitor': "monitor/monitor_closed_f1000.ply",
    'keyboard':"keyboard/keyboard_f1000.ply",
    'plasticcontainer':"plasticcontainer/plasticcontainer_f1000.ply",
    'stool':"stool/stool_f1000.ply",
    'tablesquare':"tablesquare/tablesquare_f2000.ply",
    'toolbox':"toolbox/toolbox_f1000.ply",
    "suitcase":"suitcase/suitcase_f1000.ply",
    'tablesmall':"tablesmall/tablesmall_f1000.ply",
    'yogamat': "yogamat/yogamat_f1000.ply",
    'yogaball':"yogaball/yogaball_f1000.ply",
    'trashbin':"trashbin/trashbin_f1000.ply",
}

## SMOOTHING LOSS
def smooth_mask(hand_pose_rec_o,hand_verts_o,mask):
    hand_pose_rec=torch.zeros_like(hand_pose_rec_o).to(hand_pose_rec_o.device)
    hand_pose_rec=hand_pose_rec+hand_pose_rec_o*mask
    hand_pose_rec=hand_pose_rec+hand_pose_rec_o.detach()*(1-mask)

    hand_verts=torch.zeros_like(hand_verts_o).to(hand_pose_rec_o.device)
    hand_verts=hand_verts+hand_verts_o*mask
    hand_verts=hand_verts+hand_verts_o.detach()*(1-mask)
    loss1 = 0.25 * torch.sum((((hand_pose_rec[1:-1] - hand_pose_rec[:-2]) - (hand_pose_rec[2:] - hand_pose_rec[1:-1])) ** 2)) + \
                                        0.5 * torch.sum(((hand_pose_rec[1:] - hand_pose_rec[:-1]) ** 2))
    
    loss2= 0.25 * torch.sum((((hand_verts[1:-1] - hand_verts[:-2]) - (hand_verts[2:] - hand_verts[1:-1])) ** 2)) + \
                                        0.5 * torch.sum(((hand_verts[1:] - hand_verts[:-1]) ** 2))
    return loss1+loss2
def obj_forward(raw_points, obj_rot_6d, obj_transl):
        # N_points, 3
        # B, 6
        # B, 3
    B = obj_rot_6d.shape[0]
    obj_rot = rotation_6d_to_matrix(obj_rot_6d[:, :]).permute(0, 2, 1)  # B,3,3 don't forget to transpose
    obj_points_pred = torch.matmul(raw_points.unsqueeze(0)[:, :, :3], obj_rot) + obj_transl.unsqueeze(1)
    
    return obj_points_pred

full_mesh = {
    "backpack":"backpack/backpack.obj",
    'basketball':"basketball/basketball.obj",
    'boxlarge':"boxlarge/boxlarge.obj",
    'boxtiny':"boxtiny/boxtiny.obj",
    'boxlong':"boxlong/boxlong.obj",
    'boxsmall':"boxsmall/boxsmall.obj",
    'boxmedium':"boxmedium/boxmedium.obj",
    'chairblack': "chairblack/chairblack.obj",
    'chairwood': "chairwood/chairwood.obj",
    'monitor': "monitor/monitor.obj",
    'keyboard':"keyboard/keyboard.obj",
    'plasticcontainer':"plasticcontainer/plasticcontainer.obj",
    'stool':"stool/stool.obj",
    'tablesquare':"tablesquare/tablesquare.obj",
    'toolbox':"toolbox/toolbox.obj",
    "suitcase":"suitcase/suitcase.obj",
    'tablesmall':"tablesmall/tablesmall.obj",
    'yogamat': "yogamat/yogamat.obj",
    'yogaball':"yogaball/yogaball.obj",
    'trashbin':"trashbin/trashbin.obj",
}
JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "jaw",
    "left_eye_smplhf",
    "right_eye_smplhf",
    "left_index1",
    "left_index2",
    "left_index3",
    "left_middle1",
    "left_middle2",
    "left_middle3",
    "left_pinky1",
    "left_pinky2",
    "left_pinky3",
    "left_ring1",
    "left_ring2",
    "left_ring3",
    "left_thumb1",
    "left_thumb2",
    "left_thumb3",
    "right_index1",
    "right_index2",
    "right_index3",
    "right_middle1",
    "right_middle2",
    "right_middle3",
    "right_pinky1",
    "right_pinky2",
    "right_pinky3",
    "right_ring1",
    "right_ring2",
    "right_ring3",
    "right_thumb1",
    "right_thumb2",
    "right_thumb3",
    "nose",
    "right_eye",
    "left_eye",
    "right_ear",
    "left_ear",
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
    "right_eye_brow1",
    "right_eye_brow2",
    "right_eye_brow3",
    "right_eye_brow4",
    "right_eye_brow5",
    "left_eye_brow5",
    "left_eye_brow4",
    "left_eye_brow3",
    "left_eye_brow2",
    "left_eye_brow1",
    "nose1",
    "nose2",
    "nose3",
    "nose4",
    "right_nose_2",
    "right_nose_1",
    "nose_middle",
    "left_nose_1",
    "left_nose_2",
    "right_eye1",
    "right_eye2",
    "right_eye3",
    "right_eye4",
    "right_eye5",
    "right_eye6",
    "left_eye4",
    "left_eye3",
    "left_eye2",
    "left_eye1",
    "left_eye6",
    "left_eye5",
    "right_mouth_1",
    "right_mouth_2",
    "right_mouth_3",
    "mouth_top",
    "left_mouth_3",
    "left_mouth_2",
    "left_mouth_1",
    "left_mouth_5",  # 59 in OpenPose output
    "left_mouth_4",  # 58 in OpenPose output
    "mouth_bottom",
    "right_mouth_4",
    "right_mouth_5",
    "right_lip_1",
    "right_lip_2",
    "lip_top",
    "left_lip_2",
    "left_lip_1",
    "left_lip_3",
    "lip_bottom",
    "right_lip_3",
    # Face contour
    "right_contour_1",
    "right_contour_2",
    "right_contour_3",
    "right_contour_4",
    "right_contour_5",
    "right_contour_6",
    "right_contour_7",
    "right_contour_8",
    "contour_middle",
    "left_contour_8",
    "left_contour_7",
    "left_contour_6",
    "left_contour_5",
    "left_contour_4",
    "left_contour_3",
    "left_contour_2",
    "left_contour_1",
]

## HARDCODED, restrict the local rotation to be within a rang of motion(ROM)
def restrict_angles(theta,theta_max,theta_min,mode,flag,alpha=0.01):
    MASK_MAX=(theta-theta_max)>0
    MASK_MIN=(theta-theta_min)<0
    T=theta.shape[0]
    loss_max=torch.sum(MASK_MAX.detach().float()*(theta-theta_max)**2)
    loss_min=torch.sum(MASK_MIN.detach().float()*(theta_min-theta)**2)
    if mode==0:
        return 10*loss_max**2+10*loss_min**2+torch.sum(theta**2)*alpha
    else:
        return 1*loss_max+loss_min*1


def optimize1(name,fname=''):
    human_npz_path=os.path.join(name,"human.npz")
    object_npz_path=os.path.join(name,"object.npz")
    with np.load(human_npz_path, allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    #print(poses.shape)
    with np.load(object_npz_path, allow_pickle=True) as f:
        #print(f.files)
        obj_angles,obj_trans,obj_name=f['angles'],f['trans'],str(f['name'])
    SMPLX_PATH='./models/smplx'
  
    sbj_m=sbj_m_all[gender]
    
    OBJ_PATH='./data/omomo/objects'
    obj_dir_name=os.path.join(OBJ_PATH,obj_name)
    MMESH=trimesh.load(os.path.join(obj_dir_name,obj_name+'.obj'))
    verts_obj=np.array(MMESH.vertices)
    faces_obj=np.array(MMESH.faces)
    MEAN=np.mean(verts_obj,0)
    #verts_obj=verts_obj-MEAN
    verts_sampled=np.load(os.path.join(obj_dir_name,'sample_points.npy'))
    
    obj_info={'verts': verts_obj,
                                'faces': faces_obj,
                                
                                'verts_sample': verts_sampled,
                                }
    frame_times=poses.shape[0]
    body_pose=torch.from_numpy(poses[:, 3:66]).float().to(device)
    betas_tensor=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)
    trans_tensor=torch.from_numpy(trans).float().to(device)
    root_tensor=torch.from_numpy(poses[:, :3]).float().to(device)
    hand_pose_tensor=torch.from_numpy(poses[:, 66:156]).float().to(device)
    
    # HAND MEAN
    rhand_mean=np.load(f'./assets/rhand_mean.npy')
    rhand_mean_torch=torch.from_numpy(rhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    lhand_mean=np.load(f'./assets/lhand_mean.npy')
    lhand_mean_torch=torch.from_numpy(lhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    #hand_pose_rec=Variable(torch.tensor([0.0,0.0,0.0]).float().reshape(1,1,3).repeat(frame_times,15,1).to(device),requires_grad=True)
    hand_pose_tensor=torch.cat([lhand_mean_torch,rhand_mean_torch],dim=1).reshape(-1,90)
    smplx_output = sbj_m(pose_body=body_pose, 
                            pose_hand=hand_pose_tensor, 
                            betas=betas_tensor, 
                            root_orient=root_tensor, 
                            trans=trans_tensor)
    
    verts_sbj_mean=smplx_output.v

    obj_trans_tensor=torch.from_numpy(obj_trans).float().to(device)
    obj_rot_mat_tensor=axis_angle_to_matrix(torch.from_numpy(obj_angles).float().to(device))
    obj_rot_inv_tensor=torch.linalg.inv(obj_rot_mat_tensor)
    obj_6d_tensor=matrix_to_rotation_6d(obj_rot_mat_tensor).float()
    verts_obj=obj_forward(torch.from_numpy(verts_obj).float().to(device),obj_6d_tensor,obj_trans_tensor)
    obj_normals=vertex_normals(verts_obj,torch.tensor(obj_info['faces'].astype(np.float32)).unsqueeze(0).repeat(verts_obj.shape[0],1,1).to(device))

    # Hand IDX
    rhand_idx=np.load('./assets/smplx_hand_index/rhand_smplx_ids.npy')
    lhand_idx_path='./assets/smplx_hand_index/lhand_smplx_ids.npy'
    lhand_idx=np.load(lhand_idx_path)
    h2o_signed_mean=point2point_signed(verts_sbj_mean[:,rhand_idx],verts_obj)[1]
    
    ## DEFINE THE CONTACT STATE: Pre Contact -> Contact -> Post Contact
    
    min_distance_mean_hand=torch.min(h2o_signed_mean,dim=1)[0]
    MASK1=min_distance_mean_hand<=0.02
    # Handcrafted, 
    thresh2=0.20
    k_l=1/(0.02-thresh2)
    b_l=thresh2/(thresh2-0.02)
    # Don't have the tendency to contact object if distance is far enough
    MASK2=min_distance_mean_hand>thresh2 
    WHETHER_TOUCH_RIGHT_L=torch.zeros_like(min_distance_mean_hand).float().to(device)
    WHETHER_TOUCH_RIGHT_L[MASK1]=1
    WHETHER_TOUCH_RIGHT_L[MASK2]=0
    MASK_MID=~MASK1 & ~MASK2
    
    # Linear weight, to optimize the hand from pre contact to contact; contact to post contact
    WHETHER_TOUCH_RIGHT_L[MASK_MID]=min_distance_mean_hand[MASK_MID]*k_l+b_l 
    WHETHER_TOUCH_RIGHT_L=WHETHER_TOUCH_RIGHT_L.float().detach().reshape(-1,1)
    WHETHER_TOUCH_RIGHT=MASK1.float().detach().reshape(-1,1)
    WHETHER_OPTIMIZE_RIGHT= (~MASK2).float().detach().reshape(-1,1)
    WHETHER_MIDTOUCH_RIGHT=(~MASK1).float().detach().reshape(-1,1)

    ## DEFINE THE CONTACT STATE: Pre Contact -> Contact -> Post Contact, same as the above

    frame_times_right=torch.sum(WHETHER_TOUCH_RIGHT)
    
    h2o_signed_mean=point2point_signed(verts_sbj_mean[:,lhand_idx],verts_obj)[1]

    min_distance_mean_hand=torch.min(h2o_signed_mean,dim=1)[0]
    MASK1=min_distance_mean_hand<=0.02
    thresh2=0.20
    k_l=1/(0.02-thresh2)
    b_l=thresh2/(thresh2-0.02)
    MASK2=min_distance_mean_hand>thresh2
    WHETHER_TOUCH_LEFT_L=torch.zeros_like(min_distance_mean_hand).float().to(device)
    WHETHER_TOUCH_LEFT_L[MASK1]=1
    WHETHER_TOUCH_LEFT_L[MASK2]=0
    MASK_MID=~MASK1 & ~MASK2
    WHETHER_TOUCH_LEFT_L[MASK_MID]=min_distance_mean_hand[MASK_MID]*k_l+b_l
    WHETHER_TOUCH_LEFT_L=WHETHER_TOUCH_LEFT_L.float().detach().reshape(-1,1)
    WHETHER_TOUCH_LEFT=MASK1.float().detach().reshape(-1,1)
    WHETHER_OPTIMIZE_LEFT= (~MASK2).float().detach().reshape(-1,1)
    WHETHER_MIDTOUCH_LEFT=(~MASK1).float().detach().reshape(-1,1)

    
    frame_times_left=torch.sum(WHETHER_TOUCH_LEFT)
    

    

      
    def calc_loss(verts,jtr,hand_pose_rec,epoch,left_or_right):
        
        if left_or_right: ## LEFT HAND OR RIGHT HAND
            WHETHER_TOUCH=WHETHER_TOUCH_RIGHT
            WHETHER_TOUCH_L=WHETHER_TOUCH_RIGHT_L
            frame_times_touch=frame_times_right
            hand_idx=np.load('./assets/smplx_hand_index/rhand_smplx_ids.npy')
            
            HAND_INDEXES=RHAND_INDEXES_DETAILED
            HAND_SMALL_INDEXES=RHAND_SMALL_INDEXES_DETAILED
           
            hand_mean_single=rhand_mean_torch_single
            WHETHER_OPTIMIZE=WHETHER_OPTIMIZE_RIGHT
            WHETHER_MIDTOUCH=WHETHER_MIDTOUCH_RIGHT
        else:
            WHETHER_TOUCH=WHETHER_TOUCH_LEFT
            WHETHER_TOUCH_L=WHETHER_TOUCH_LEFT_L
            frame_times_touch=frame_times_left
            lhand_idx_path='./assets/smplx_hand_index/lhand_smplx_ids.npy'

            hand_idx=np.load(lhand_idx_path)
            # HAND_INDEXES=LHAND_INDEXES
            # HAND_SMALL_INDEXES=LHAND_SMALL_INDEXES
            HAND_INDEXES=LHAND_INDEXES_DETAILED
            HAND_SMALL_INDEXES=LHAND_SMALL_INDEXES_DETAILED
           
                #WHETHER_MID_TOUCH=WHETHER_MIDTOUCH_LEFT
            hand_mean_single=lhand_mean_torch_single
            WHETHER_OPTIMIZE=WHETHER_OPTIMIZE_LEFT
            WHETHER_MIDTOUCH=WHETHER_MIDTOUCH_LEFT

        

        
        with torch.enable_grad():       
            
            if epoch==0:
                global hand_distance_init
                hand_distance_init=(torch.norm((jtr[:,[45,48,51]]-jtr[:,[42,45,48]]),dim=-1)).detach()
            
            
            o2h_signed,sbj2obj, o2h_idx, sbj2obj_idx, o2h, sbj2obj_vector = point2point_signed(verts[:,hand_idx], verts_obj,y_normals=obj_normals,return_vector=True) #y_normals=obj_normals
            
  
            if left_or_right==1:
                loss_touch=0.05*torch.sum(((jtr[:,[41,42,44,45,47,48,50,51,53,54]]-jtr[:,[40,41,43,44,46,47,49,50,52,53]])**2)*WHETHER_TOUCH.view(-1,1,1))#+ 1*torch.sum(h2o_signed**2)
            elif left_or_right==0:

                loss_touch=0.05*torch.sum(((jtr[:,[26, 27, 29, 30, 32, 33, 35, 36, 38, 39]]-jtr[:,[25, 26, 28, 29, 31, 32, 34, 35, 37, 38]])**2)*WHETHER_TOUCH.view(-1,1,1))#+ 1*torch.sum(h2o_signed**2)

            collision_loss=torch.tensor(0.0).to(device)
            loss_dist_o=torch.tensor(0.0).to(device)
            
            loss_verts_reg= torch.tensor(0)
           
            thresh=0.00
            
            loss_touch=torch.tensor(0.0).to(device)
            
            ## CONTACT CALCULATION (HALF OF THE HAND THAT IS LIKELY TO CONTACT)
            for i in range(16):
                
                num_verts=HAND_INDEXES[i].shape[0]
                sd_i=sbj2obj[:,HAND_INDEXES[i]]
                MASK_I=sd_i<thresh
               
                whether_pene_time=torch.sum(MASK_I,dim=-1) ## T
                MASK_TIME=(whether_pene_time>0).detach()
                SUMM=torch.sum(MASK_TIME)
                if torch.sum(MASK_TIME)>0:
                    sd_pen=(sd_i*WHETHER_TOUCH_L.view(-1,1))[MASK_TIME]
                    zeros_s2o, ones_s2o = torch.zeros_like(sd_pen).float().to(device), torch.ones_like(sd_pen).float().to(device)
                    calc_dist=sd_pen-thresh
                    mask_pen=((sd_pen-thresh)<0).float()*(torch.abs(sd_pen)<0.03).float()
                    num_pen=torch.sum(mask_pen,dim=-1).reshape(-1,1)
                    loss_dist_o+=torch.sum(torch.abs(calc_dist)*mask_pen/(num_pen+1e-8))*2 #num_pen+1e-8)

                OP_MASK_TIME=~MASK_TIME
                # num_verts_small=HAND_SMALL_INDEXES[i].shape[0]
                sbj2obj_mask_first=sbj2obj*WHETHER_TOUCH_L
                sd_closer=sbj2obj_mask_first[OP_MASK_TIME][:,HAND_SMALL_INDEXES[i]]#/num_verts_small
                loss_touch+=5*torch.sum(torch.abs(sd_closer)**2)
                    
            euler_angles=hand_pose_rec[:,:,[2,1,0]]
            if not left_or_right:
                euler_angles=euler_angles*(torch.tensor([-1.0,-1.0,1.0]).to(device).reshape(1,1,3))

            ## ROM Restirction for Hands
            thumb_pinky_out_notmid=euler_angles[:,[0,3,9]]
            thumb_pinky_out_mid=euler_angles[:,[1,2,4,5]] 
            thumb_pinky_out3=euler_angles[:,[10,11]]

            pinky_in=euler_angles[:,[6,7,8]]
            thumb_in=euler_angles[:,12:15]

            ## Y 0.4 Z 0.5,-0.4
            # original 1.3
            theta_max1_1=torch.tensor([1.10,0.09,0.13]).reshape(1,1,3).float().to(device)
            theta_min1_1=torch.tensor([-0.8,-0.08,-0.2]).reshape(1,1,3).float().to(device)
            loss_pinky_thumb_out_notmid=restrict_angles(thumb_pinky_out_notmid,theta_max1_1,theta_min1_1,mode=1,flag='1')
            
            theta_max1_2=torch.tensor([1.10,0.15,0.12]).reshape(1,1,3).float().to(device)
            theta_min1_2=torch.tensor([-0.1,-0.10,-0.15]).reshape(1,1,3).float().to(device)
            loss_pinky_thumb_out_mid=restrict_angles(thumb_pinky_out_mid,theta_max1_2,theta_min1_2,mode=1,flag='2')

            theta_max1_3=torch.tensor([1.10,0.15,0.10]).reshape(1,1,3).float().to(device)
            theta_min1_3=torch.tensor([-0.1,-0.10,-0.35]).reshape(1,1,3).float().to(device)
            loss_pinky_thumb_out3=restrict_angles(thumb_pinky_out3,theta_max1_3,theta_min1_3,mode=1,flag='2-2')
            
            loss_pinky_thumb_out=loss_pinky_thumb_out_notmid+loss_pinky_thumb_out_mid+loss_pinky_thumb_out3
            
            theta_max2=torch.tensor([1.10,0.5,1.10]).reshape(1,1,3).float().to(device)
            theta_min2=torch.tensor([[-0.8,-0.5,-0.8],[-0.4,-0.5,-0.8],[-0.5,-0.5,-0.8]]).reshape(1,3,3).float().to(device)

            loss_pinky=restrict_angles(pinky_in,theta_max2,theta_min2,mode=1,flag='3')
            #                                                  -0.1
            theta_max3=torch.tensor([[0.45,0.45,1.5],[0.45,0.45,-0.1],[0.45,0.45,1.5]]).reshape(1,3,3).float().to(device)
            
            theta_min3=torch.tensor([[-0.5,-0.5,-0.2],[-0.5,-0.5,-0.8],[-0.5,-0.5,-0.8]]).reshape(1,3,3).float().to(device)

            loss_thumb=restrict_angles(thumb_in,theta_max3,theta_min3,mode=1,flag='4')

            loss_rot_reg =loss_pinky_thumb_out+loss_pinky+loss_thumb
            
            # ## restrict angles
            if left_or_right:
                d1=0.2*torch.sum(((jtr[:,[41,42,44,45,50,51,41,42]]-jtr[:,[44,45,50,51,47,48,53,54]])**2)*WHETHER_TOUCH_L.view(-1,1,1))
                d2=0.2*torch.sum(((verts[:,[7669,7794,7905]]-verts[:,[7794,7905,8022]])**2)*WHETHER_TOUCH_L.view(-1,1,1))
            else:
                d1=0.2*torch.sum(((jtr[:,[26, 27, 29, 30, 35, 36, 26, 27]]-jtr[:,[29, 30, 35, 36, 32, 33, 38, 39]])**2)*WHETHER_TOUCH_L.view(-1,1,1))
                d2=0.2*torch.sum(((verts[:,[4933,5058,5169]]-verts[:,[5058,5169,5286]])**2)*WHETHER_TOUCH_L.view(-1,1,1))

            loss_rot_reg+=2-d1-d2
            
            
            
            # smoothing
            if epoch>100:
                hand_verts=verts[:,hand_idx]
                loss_hand_pose_v_reg=smooth_mask(hand_pose_rec,hand_verts,(WHETHER_TOUCH).view(-1,1,1))*0.5
                loss_hand_pose_v_reg+=smooth_mask(hand_pose_rec,hand_verts,(WHETHER_OPTIMIZE-WHETHER_TOUCH).view(-1,1,1))
                loss_hand_pose_v_reg+=smooth_mask(hand_pose_rec,hand_verts,1-WHETHER_OPTIMIZE.view(-1,1,1))*2
   
                ## Initialization reg
                loss_hand_pose_v_reg+=0.05*torch.sum((hand_pose_rec-hand_mean_single.view(1,-1,3))**2*(1-WHETHER_TOUCH_L).view(-1,1,1))
                
            elif epoch<=100:
                loss_hand_pose_v_reg=torch.tensor(0).to(device)
           
            ## Relieve Contact sliding
            if epoch>200:
                CONTACT_MASK=(torch.abs(sbj2obj)<0.01).unsqueeze(2).float().detach()#*((sbj2obj)>=0).unsqueeze(2).float().detach()
                delta_inv_minus=torch.einsum('tij,tjk->tik',(verts[:,hand_idx]-sbj2obj_vector.detach()-obj_trans_tensor.reshape(-1,1,3)),obj_rot_inv_tensor.permute(0,2,1)) # T,3,3
                ## Contact Sliding Loss
                if epoch%2==0:
                    delta_temporal_inv_minus=((delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[1:]*WHETHER_TOUCH[1:].view(-1,1,1))**2 ##T,N,3
                else:
                    delta_temporal_inv_minus= ((delta_inv_minus[1:]-delta_inv_minus[:-1])*CONTACT_MASK[1:]*WHETHER_TOUCH[1:].view(-1,1,1))**2##T,N,3
                loss_hand_pose_v_reg+=0.01*torch.sum(delta_temporal_inv_minus)
                ## Hand Prior Loss
                hand_prior_loss=hand_prior(hand_pose_rec.view(-1,45),left_or_right=left_or_right).squeeze(0)
                loss_hand_pose_v_reg+=0.1*torch.sum(hand_prior_loss**2*WHETHER_OPTIMIZE)
                      
            loss_body_v_reg = torch.tensor(0.0).to(device)#100 * torch.mean((((body_rec[1:-1] - body_rec[:-2]) - (body_rec[2:] - body_rec[1:-1])) ** 2).sum(dim=2).sum(dim=1)) + 100 * torch.mean(((body_rec[1:] - body_rec[:-1]) ** 2).sum(dim=2).sum(dim=1)) + 1000 * (loss_left + loss_right)
            loss_v_reg = 1 * (loss_hand_pose_v_reg + loss_body_v_reg ) +loss_rot_reg
            

            loss = (
                    loss_dist_o+loss_touch+
                    loss_v_reg
                    )

            loss_dict = {}
            loss_dict['total'] = loss.detach().cpu().numpy()

            loss_dict['collision'] = loss_dist_o.detach().cpu().numpy()
            loss_dict['reg'] = loss_touch.detach().cpu().numpy()
            loss_dict['reg_v'] = loss_v_reg.detach().cpu().numpy()
        return loss,collision_loss,loss_dict
    def calc_loss_common(hand_pose_rec,epoch):
        SBJ_OUTPUT=sbj_m(pose_body=body_pose, 
                            pose_hand=hand_pose_rec.view(-1,90), 
                            betas=betas_tensor, 
                            root_orient=root_tensor, 
                            trans=trans_tensor)
        verts=SBJ_OUTPUT.v
        jtr=SBJ_OUTPUT.Jtr
        save_path='./save/omomo2_1500_mano_square_bigparameter_0.01_up'
        os.makedirs(save_path,exist_ok=True)
        save_path=os.path.join(save_path,fn)
        

        loss_left,collision_loss_left,loss_dict_left=calc_loss(verts,jtr,hand_pose_rec[:,:15,:],epoch,0)
        loss_right,collision_loss_right,loss_dict_right=calc_loss(verts,jtr,hand_pose_rec[:,15:,:],epoch,1)
        loss_dict_all={}

        LOSS_RATIO=1
        for key in loss_dict_right.keys():
            loss_dict_all[key]=(loss_dict_right[key]+loss_dict_left[key])*LOSS_RATIO
        loss_all=loss_left+loss_right
        collision_loss_all=collision_loss_left+collision_loss_right
        return loss_all*LOSS_RATIO,collision_loss_all,loss_dict_all,None
        


    tmp_smplhparams = {}
    tmp_objparams = {}
    rhand_mean=np.load(f'./assets/rhand_mean.npy')
    rhand_mean_torch=torch.from_numpy(rhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    lhand_mean=np.load(f'./assets/lhand_mean.npy')
    lhand_mean_torch=torch.from_numpy(lhand_mean.reshape(1,-1,3)).float().to(device).repeat(frame_times,1,1)
    #hand_pose_rec=Variable(torch.tensor([0.0,0.0,0.0]).float().reshape(1,1,3).repeat(frame_times,30,1).to(device),requires_grad=True)
    hand_pose_rec=Variable(torch.cat([lhand_mean_torch,rhand_mean_torch],dim=1),requires_grad=True)

    
    optimizer=optim.Adam([hand_pose_rec],lr=0.001)

    for ii in (tqdm(range(1000))):
        optimizer.zero_grad()
        loss, coll,loss_dict,endflag = calc_loss_common(hand_pose_rec,ii)
        if endflag:
            break
        # losses_str = ' '.join(['{}: {:.4f} | '.format(x, loss_dict[x]) for x in loss_dict.keys()])
        loss.backward(retain_graph=False)
        optimizer.step()
        tmp_smplhparams['hand_pose'] = copy.deepcopy(hand_pose_rec.detach())
    
    
    hand_pose=tmp_smplhparams['hand_pose'].view(-1,90).detach().cpu().numpy()
    export_file = f"./data/omomo_correct/sequences_canonical"
    os.makedirs(export_file, exist_ok=True)
    with np.load(human_npz_path, allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    poses[:,-30*3:] = hand_pose
    save_path = os.path.join(export_file, fname,'human.npz')
    np.savez(save_path,**{'poses':poses,'betas':betas,'trans':trans,'gender':np.array(gender)})

    save_path = os.path.join(export_file, fname,'loss.pkl')
    with open(save_path,'wb') as f:
        pickle.dump(loss_dict,f)
    return tmp_smplhparams, tmp_objparams
def parse_args():
    
                      
    parser = argparse.ArgumentParser()        
   
    parser.add_argument('--number',type=int,default=0)
    parser.add_argument('--dataset',type=str)
    

    
    args = parser.parse_args()                                      
    return args

if __name__ == '__main__':
    args=parse_args()
    
    root_path = 'data/omomo/sequences_canonical'

    export_file = f"./data/omomo_correct/sequences_canonical"
    for i,fn in tqdm(enumerate(sorted(os.listdir(root_path)))):
        try:
            name=os.path.join(root_path,fn)
            optimize1(name,fn)
        except:
            pass

        
        
    
    
    
    