import os
import numpy as np
import torch
import argparse

import trimesh
import pickle
from hoi_correction.libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer

import smplx
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle,rotation_6d_to_matrix,matrix_to_rotation_6d
from torch.autograd import Variable
import torch.optim as optim
import copy

from human_body_prior.body_model.body_model import BodyModel

from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

smpl_all={}
smpl=0
whether_touch=0
lhand_idx_path='./assets/smplh_hand_index/lhand_smplh_ids.npy'
lhand_idx=np.load(lhand_idx_path)

rhand_idx_path='./assets/smplh_hand_index/rhand_smplh_ids.npy'
rhand_idx=np.load(rhand_idx_path)


   

    

def set_smpl_all(dataset):
    
    MODEL_PATH='./models'

    global smpl_all
    if dataset in ['behave']:
        smpl_type='smplh_10'
        sbj_m_female = SMPL_Layer(center_idx=0, gender='female', num_betas=10,
                               model_root=str(os.path.join(MODEL_PATH,'smplh/mano_v1_2/models/')),hands=True).to(device)
        
    
        sbj_m_male = SMPL_Layer(center_idx=0, gender='male', num_betas=10,
                        model_root=str(os.path.join(MODEL_PATH,'smplh/mano_v1_2/models/')),hands=True).to(device)
        smpl_all={'male':sbj_m_male,'female':sbj_m_female}
        

    elif dataset in [ 'neuraldome','imhd']:
        smpl_type='smplh_16'
        SMPLH_PATH = os.path.join(MODEL_PATH,'smplh')
        surface_model_male_fname = os.path.join(SMPLH_PATH,'female', "model.npz")
        surface_model_female_fname = os.path.join(SMPLH_PATH, "male","model.npz")
        surface_model_neutral_fname = os.path.join(SMPLH_PATH, "neutral","model.npz")

        dmpl_fname = None
        num_dmpls = None 
        num_expressions = None
        num_betas = 16 
        

        smplh16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname).to(device)
        smplh16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname).to(device)
        smplh16_model_neutral = BodyModel(bm_fname=surface_model_neutral_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname).to(device)

        smpl_all = {'male': smplh16_model_male, 'female': smplh16_model_female,'neutral':smplh16_model_neutral}
    elif dataset == 'chairs':
        smpl_type='smplx_10'
        smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                        gender = 'male',
                        use_pca=False,
                        ext='pkl').to(device)
                           
        smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="female",
                        use_pca=False,
                        ext='pkl').to(device)
        smpl_all = {'male': smplx_model_male, 'female': smplx_model_female}
    elif dataset=='intercap':
        smpl_type='smplx_10'
        smplx_model_male = smplx.create(MODEL_PATH, model_type='smplx',
                        gender = 'male',
                        use_pca=True,num_pca_comps=12,
                        ext='pkl').to(device)
                           
        smplx_model_female = smplx.create(MODEL_PATH, model_type='smplx',
                        gender="female",
                        use_pca=True,num_pca_comps=12,
                        ext='pkl').to(device)
        smpl_all = {'male': smplx_model_male, 'female': smplx_model_female}
    elif dataset == 'omomo':
        smpl_type="smplx_16"
        SMPLX_PATH = MODEL_PATH+'/smplx'
        dmpl_fname = None
        num_dmpls = None 
        num_expressions = None
        num_betas = 16 
        surface_model_male_fname = os.path.join(SMPLX_PATH,"SMPLX_MALE.npz")
        surface_model_female_fname = os.path.join(SMPLX_PATH,"SMPLX_FEMALE.npz")
        smplx16_model_male = BodyModel(bm_fname=surface_model_male_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname).to(device)
        smplx16_model_female = BodyModel(bm_fname=surface_model_female_fname,
                        num_betas=num_betas,
                        num_expressions=num_expressions,
                        num_dmpls=num_dmpls,
                        dmpl_fname=dmpl_fname).to(device)
        smpl_all = {'male': smplx16_model_male, 'female': smplx16_model_female}
    return smpl_type
        
def set_smpl(gender):
    global smpl
    smpl=smpl_all[gender]
def trans_6d_aa(rot_6d):
    T=rot_6d.shape[0]
    return matrix_to_axis_angle(rotation_6d_to_matrix(rot_6d.view(T,-1,6).float()).float()).view(T,-1)

def forward_human(smpl_type,glo_rot_,body_rot_,hand_rot,beta,body_trans):
    glo_rot=trans_6d_aa(glo_rot_)
    body_rot=trans_6d_aa(body_rot_)


    if smpl_type=='smplh_10':
        
        verts,joints,_,_=smpl(torch.cat([glo_rot,body_rot,hand_rot],dim=1),th_trans=body_trans,th_betas=beta)
        faces=smpl.th_faces
        
    elif smpl_type == 'smplx_10':
        frame_times=glo_rot.shape[0]
        hand_rot_half=hand_rot.shape[1]//2
        

        smplx_output = smpl(body_pose=body_rot,
                                global_orient=glo_rot,
                                left_hand_pose=hand_rot[:,:hand_rot_half],
                                right_hand_pose=hand_rot[:,hand_rot_half:],
                                jaw_pose = torch.zeros([frame_times,3]).float().to(device),
                                reye_pose = torch.zeros([frame_times,3]).float().to(device),
                                leye_pose = torch.zeros([frame_times,3]).float().to(device),
                                expression = torch.zeros([frame_times,10]).float().to(device),
                                betas=beta,
                                transl=body_trans) 
            
            
        verts = (smplx_output.vertices)
        joints=smplx_output.joints
        

        faces = torch.from_numpy(smpl.faces.astype(np.int32)).to(device)
    else:
        smplx_output = smpl(pose_body=body_rot, 
                            pose_hand=hand_rot, 
                            betas=beta,
                            root_orient=glo_rot, 
                            trans=body_trans)
        
        verts = smplx_output.v
        faces = smpl.f
        joints=smplx_output.Jtr
    
    return verts, joints,faces
    
    



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


def optimize1(name,dataset_name,smpl_type):

    human_npz_path=os.path.join(name,"human.npz")
    object_npz_path=os.path.join(name,"object.npz")
    with np.load(human_npz_path, allow_pickle=True) as f:
        poses, betas, trans, gender = f['poses'], f['betas'], f['trans'], str(f['gender'])
    frame_times=poses.shape[0]
    
    hand_pose=torch.from_numpy(poses[:,66:]).float().to(device)
    
    with np.load(object_npz_path, allow_pickle=True) as f:
        #print(f.files)
        obj_angles,obj_trans,obj_name=f['angles'],f['trans'],str(f['name'])
    

   
    body_pose=torch.from_numpy(poses[:,:66]).float().to(device)
    body_trans=torch.from_numpy(trans[:]).float().to(device)
    obj_trans=torch.from_numpy(obj_trans[:]).float().to(device)
    obj_angles=torch.from_numpy(obj_angles[:]).float().to(device)
    beta=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float().to(device)

   
    OBJ_PATH=f'./data/{dataset_name}/objects'
    if dataset_name=='behave':
        MMESH=trimesh.load(os.path.join(OBJ_PATH,obj_name,obj_name+'.obj'))
        V1=MMESH.vertices
        V1=V1-np.mean(V1,0)
        obj_points = torch.tensor(V1).float().to(device)
    else:
        obj_dir_name=os.path.join(OBJ_PATH,obj_name)
        MMESH=trimesh.load(os.path.join(obj_dir_name,obj_name+'.obj'))
        V1=MMESH.vertices
        V1=V1-np.mean(V1,0)
        obj_points = torch.tensor(V1).float().to(device)

   
    T = body_pose.shape[0]
    obj_rot =matrix_to_rotation_6d(axis_angle_to_matrix(obj_angles.float()).float()).float().reshape(T,-1)
    glo_rot = matrix_to_rotation_6d(axis_angle_to_matrix((body_pose[:, :3].view(T, 3)).float()).float()).float().reshape(T,-1)
    body_rot = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose[:, 3:66].view(T, -1,3).float()).float()).float().reshape(T,-1)
    hand_rot=hand_pose.view(T, -1).float()
    
    
    
    set_smpl(gender)
    
    verts_gt,jtr_gt,smpl_th_faces= forward_human(smpl_type,glo_rot,body_rot,hand_rot,beta,body_trans)
    floor_height=torch.min(verts_gt[:,:,1]).item()
    left_foot = jtr_gt[:, 10]
    right_foot = jtr_gt[:, 11]
    # print(left_foot.shape)
    delta_left = torch.norm(left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]], dim=1) + 1e-6
    delta_right = torch.norm(right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]], dim=1) + 1e-6
    

    
    def calc_loss(body_rec, transl_rec, glo_rot_rec, obj_transl_rec, obj_rot_rec, hand_pose_rec, ratio,epoch,smpl_type,dataset_name): 
        with torch.enable_grad():       
            #pose = matrix_to_axis_angle(rotation_6d_to_matrix(torch.cat([glo_rot_rec, body_rec, hand_pose_rec], dim=1)).float()).view(T,-1)
            verts,jtr,faces=forward_human(smpl_type,glo_rot_rec,body_rec,hand_pose_rec,beta,transl_rec)
            
            obj_transl = obj_transl_rec
            obj_points_pred = torch.matmul(obj_points.unsqueeze(0), rotation_6d_to_matrix(obj_rot_rec.float()).permute(0, 2, 1)) + obj_transl.unsqueeze(1)
            
            left_foot = jtr[:, 10]
            right_foot = jtr[:, 11]
            delta_left = torch.norm(left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]], dim=1) + 1e-6
            delta_right = torch.norm(right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]], dim=1) + 1e-6
            left_static = copy.deepcopy((delta_left < 0.008).detach())
            right_static = copy.deepcopy((delta_right < 0.008).detach())
            # insert mean
            if left_static.any():
                loss_left = torch.sum((((left_foot[1:, [0, 2]] - left_foot[:-1, [0, 2]])[left_static]) ** 2))
            else:
                loss_left = 0
            if right_static.any():
                loss_right = torch.sum((((right_foot[1:, [0, 2]] - right_foot[:-1, [0, 2]])[right_static]) ** 2))
            else:
                loss_right = 0
            weight_mask =1
           
            
            # regulation
            loss_obj_transl_reg = 10*0.1 * torch.sum((obj_transl - obj_trans).abs())
            loss_obj_rot_reg = 10*0.1 * torch.sum((obj_rot_rec - obj_rot).abs())
            loss_transl_reg = 1*0.1 * torch.sum((transl_rec - body_trans).abs())
            loss_glo_rot_reg = 1*0.1 * torch.sum((glo_rot_rec - glo_rot).abs())
            
            loss_body_reg = 4*0.1 * torch.sum(((body_rec - body_rot).abs()).sum(dim=1))#+0.0*torch.sum(torch.mean(torch.abs(verts-verts_gt)))
            loss_transl_v_reg = 1 *weight_mask* torch.sum(((transl_rec[1:-1] - transl_rec[:-2]) - (transl_rec[2:] - transl_rec[1:-1])) ** 2) + \
                                1 * torch.sum(((transl_rec[1:] - transl_rec[:-1])) ** 2)
            loss_glo_rot_v_reg = 1 *weight_mask* torch.sum(((glo_rot_rec[1:-1] - glo_rot_rec[:-2]) - (glo_rot_rec[2:] - glo_rot_rec[1:-1])) ** 2) + \
                                1 * torch.sum(((glo_rot_rec[1:] - glo_rot_rec[:-1])) ** 2)
            loss_hand_pose_v_reg = 1 *weight_mask* torch.sum(((hand_pose_rec[1:-1] - hand_pose_rec[:-2]) - (hand_pose_rec[2:] - hand_pose_rec[1:-1])) ** 2) + \
                                    1 * torch.sum((hand_pose_rec[1:] - hand_pose_rec[:-1]) ** 2)
            loss_obj_v_reg = 1 *weight_mask* torch.sum(((obj_transl_rec[1:-1] - obj_transl_rec[:-2]) - (obj_transl_rec[2:] - obj_transl_rec[1:-1])) ** 2) + \
                             1 * torch.sum(((obj_transl_rec[1:] - obj_transl_rec[:-1])) ** 2) +\
                             1 *weight_mask* torch.sum(((obj_rot_rec[1:-1] - obj_rot_rec[:-2]) - (obj_rot_rec[2:] - obj_rot_rec[1:-1])) ** 2) + \
                             1 * torch.sum(((obj_rot_rec[1:] - obj_rot_rec[:-1])) ** 2)+0.1 *weight_mask* torch.sum((((obj_points_pred[1:-1] - obj_points_pred[:-2]) - (obj_points_pred[2:] - obj_points_pred[1:-1])) ** 2).sum(dim=-1).mean(dim=1)) + \
                             0.1 * torch.sum(((obj_points_pred[1:] - obj_points_pred[:-1]) ** 2).sum(dim=-1).mean(dim=1))
            loss_body_v_reg = weight_mask*1 * torch.sum((((body_rec[1:-1] - body_rec[:-2]) - (body_rec[2:] - body_rec[1:-1])) ** 2).sum(dim=1)) + 1 * torch.sum(((body_rec[1:] - body_rec[:-1]) ** 2).sum(dim=1)) + 20 * (loss_left + loss_right)+\
            weight_mask*0.1 * torch.sum((((verts[1:-1] - verts[:-2]) - (verts[2:] - verts[1:-1])) ** 2).sum(dim=-1).mean(dim=1)) + 0.1 * torch.sum(((verts[1:] - verts[:-1]) ** 2).sum(dim=-1).mean(dim=1))
            
            
           
            if epoch>40:
                loss_v_reg = 2 * (loss_hand_pose_v_reg + loss_obj_v_reg + loss_body_v_reg + loss_transl_v_reg + loss_glo_rot_v_reg)
            else:
                loss_v_reg = 1 * (loss_obj_v_reg)*0
            loss_all_reg=(loss_obj_transl_reg + loss_obj_rot_reg + loss_body_reg + loss_transl_reg + loss_glo_rot_reg)

            loss_all_reg=loss_all_reg/10
            if dataset_name in ['behave','intercap']:
                loss_all_reg=loss_all_reg/5
            loss_all_reg+=10*torch.sum((hand_pose_rec - hand_rot)**2)

            
            loss = (
                    
                     loss_all_reg +
                    loss_v_reg
                    )
          

            loss_dict = {}
            loss_dict['total'] = loss.detach().cpu().numpy()

            loss_dict['reg'] = (loss_obj_transl_reg + loss_obj_rot_reg + loss_body_reg + loss_transl_reg + loss_glo_rot_reg).detach().cpu().numpy()
            loss_dict['reg_v'] = loss_v_reg.detach().cpu().numpy()
            # if epoch==0:
            #     visualize_body_obj(-verts.detach().cpu().numpy(),faces.detach().cpu().numpy()[...,::-1],-obj_points_pred.detach().cpu().numpy(),obj_th_faces.detach().cpu().numpy()[...,::-1],save_path=os.path.join('./fullbody_example_original'),save_gif=True,save_both=False,save_npz=True)
        return loss, loss_dict

    best_eval_grasp = 1e7
    tmp_smplhparams = {}
    tmp_objparams = {}
    obj_transl_rec = Variable(copy.deepcopy(obj_trans).to(device), requires_grad=True)
    obj_rot_rec = Variable(copy.deepcopy((obj_rot)).to(device),
                                requires_grad=True)  # 6d

    transl_rec = Variable(copy.deepcopy(body_trans).to(device), requires_grad=True)
    glo_rot_rec = Variable(copy.deepcopy((glo_rot)).float().to(device),
                                requires_grad=True)  # 6d
    body_rec = Variable(copy.deepcopy((body_rot)).float().to(device),
                                requires_grad=True)

    hand_pose_rec = Variable(copy.deepcopy((hand_rot)).float().to(device), requires_grad=True)
    optimizer = optim.Adam([body_rec, transl_rec, glo_rot_rec, obj_transl_rec, obj_rot_rec, hand_pose_rec],
                                lr=0.002)

    for ii in (range(100)):
        optimizer.zero_grad()
        loss, loss_dict = calc_loss(body_rec, transl_rec, glo_rot_rec, obj_transl_rec, obj_rot_rec, hand_pose_rec, ii / 100,ii,smpl_type,dataset_name) ## insert
        losses_str = ' '.join(['{}: {:.4f} | '.format(x, loss_dict[x]) for x in loss_dict.keys()])
       
        loss.backward(retain_graph=False)
        optimizer.step()
        eval_grasp = loss
        
    

        

    tmp_smplhparams = {}
    tmp_objparams = {}
    tmp_objparams['obj_trans'] = obj_transl_rec.detach().cpu().numpy()
    tmp_objparams['obj_angle'] = trans_6d_aa(obj_rot_rec).detach().cpu().numpy()
  
    tmp_smplhparams['transl'] = transl_rec.detach().cpu().numpy()
    tmp_smplhparams['body_pose'] = trans_6d_aa(body_rec).detach().cpu().numpy().reshape(T, -1)
    tmp_smplhparams['hand_pose'] = hand_pose_rec.detach().cpu().numpy().reshape(T, -1)
    tmp_smplhparams['glo_rot'] = trans_6d_aa(glo_rot_rec).detach().cpu().numpy().reshape(T, -1)
    tmp_objparams['obj_name']=obj_name
    tmp_smplhparams['gender']=gender

    return tmp_smplhparams, tmp_objparams, loss_dict
    

def parse_args():
    
                      
    parser = argparse.ArgumentParser()        
                                                                    
    parser.add_argument('--dataset',type=str)
    parser.add_argument('--number',type=int,default=0)
    

    
    args = parser.parse_args()                                       
    return args

if __name__ == '__main__':
    args=parse_args()
    dataset_name=args.dataset
    SECTION_NUMBER=args.number
    root_path=f'./data/{dataset_name}/sequences_canonical' 
    
    
    smpl_type=set_smpl_all(dataset_name)

    for i,fn in tqdm(enumerate(sorted(os.listdir(root_path)))):
        try:
            name=os.path.join(root_path,fn)
            export_file = f"./data/{dataset_name}_correct/sequences_canonical/"
            os.makedirs(export_file, exist_ok=True)
            
            save_path_h = os.path.join(export_file, fn,'human.npz')
            save_path_o = os.path.join(export_file, fn,'object.npz')
       
            tmp_smplhparams, tmp_objparams,loss_dict=optimize1(name,dataset_name,smpl_type)
            np.savez(save_path_h,**tmp_smplhparams)
            np.savez(save_path_o,**tmp_objparams)
            with open(os.path.join(export_file,fn,'loss.pkl'),'wb') as f:
                pickle.dump(loss_dict,f)
           
           
        except:
            pass
  