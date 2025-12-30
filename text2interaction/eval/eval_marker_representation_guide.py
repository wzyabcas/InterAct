# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
# from sentence_transformers import SentenceTransformer
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import load_model
from utils import dist_util
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from utils.guidance import Guide_Contactv2
from model.hoi_diff import HOIDiff as used_model
from diffusion.gaussian_diffusion import LocalMotionDiffusion
from bps_torch.bps import bps_torch

from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder
from utils.eval_t2m_utils import *
from common.quaternion import rotation_6d_to_matrix
# from sentence_transformers import SentenceTransformer

def load_motion_dataset(args, max_frames, n_frames, training_stage=3): 
    data_conf = DatasetConfig(
        name=args.dataset,
        batch_size=args.batch_size,
        num_frames=max_frames,
        obj_split=args.obj_split,
        split='test',
        hml_mode='text_only',
        training_stage=training_stage,debug=args.debug)
    data = get_dataset_loader(data_conf)
    data.fixed_length = n_frames
    return data



@torch.no_grad()
def eval_t2hoi(val_loader, motion_model, motion_diffusion, textencoder,motionencoder,std_enc,mean_enc,repeat_id):

    mm_batch_num=2 # insert
    MAX_REPEAT=2# insert
    motion_multimodality=[]# insert
    mutlimodality_select_times=1 # insert

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    # text_f_list=[]
    # sbert_f_list=[] # only once
    # sbert_f_list=[] # only once
    matching_score_real = 0
    matching_score_pred = 0

    # print(num_quantizer)

    # assert num_quantizer >= len(time_steps) and num_quantizer >= len(cond_scales)

    nb_sample = 0
    l1_dist = 0
    num_poses = 1
    # for i in range(1):
    ## insert
    for batch_idx,batch in enumerate(val_loader):
        motions, model_kwargs=batch ## insert
        motions = motions[:,:485]
        motions_copy = motions 
        bs = len(motions)
        model_kwargs['y']['scale'] = torch.ones(args.batch_size, device=dist_util.dev())
        model_kwargs['y']['obj_bps'] = model_kwargs['y']['obj_bps'].to(dist_util.dev())
        model_kwargs['y']['obj_points'] = model_kwargs['y']['obj_points'].to(dist_util.dev())
        motions = motions.cpu().permute(0, 2, 3, 1).squeeze().float()

        text_key = 'text' if 'text' in model_kwargs['y'] else 'action'
        text_list_gt = model_kwargs['y'][text_key]
        # print(text_list_gt)
        lengths = model_kwargs['y']['lengths'].cpu().numpy()
        motions_obj = motions[...,476:].float()
        obj_points = model_kwargs['y']['obj_points']
        obj_points = obj_points[:,None,...].repeat(1,motions.shape[1],1,1)
        all_obj_name = model_kwargs['y']['seq_name']
        vertices = obj_points.reshape(-1,obj_points.shape[2],3).float().cpu()

        angle, trans = motions_obj[..., :6].reshape(-1,6).float(), motions_obj[..., 6:9].reshape(-1,3).float()

        torch_rot = rotation_6d_to_matrix(angle)
        obj_points = torch.bmm(vertices.float(), torch_rot.transpose(1, 2)) + trans[:, None, :]
        bps_obj_times = bps_obj.repeat(trans.shape[0], 1, 1).float() + trans[:, None, :]
        bps_time_nonrot = bps_torch.encode(x=obj_points, \
            feature_type=['deltas'], \
            custom_basis=bps_obj_times)['deltas'] # T X N X 3 
        gt_motions = torch.cat((motions[...,:231].cuda(),bps_time_nonrot.reshape(motions.shape[0], motions.shape[1],128*3).cuda()),dim=-1)

        guide_fn_contact = Guide_Contactv2(classifiler_scale=args.classifier_scale)
        sample_fn = motion_diffusion.p_sample_loop

        sample = sample_fn(
            motion_model,
            (args.batch_size, motion_model.njoints + 18+78*6, motion_model.nfeats,  max(lengths)),    # + 6 object pose
            clip_denoised=False,
            # clip_denoised=not args.predict_xstart,
            model_kwargs=model_kwargs,
            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            cond_fn=guide_fn_contact,
        )
        # sample = motions_copy
        sample = sample[:,:485]
        sample = sample.cpu().permute(0, 2, 3, 1).squeeze().float()
        # sample = sample.cpu().permute(0, 2, 3, 1).squeeze().float()
        sample_obj = sample[..., 476:485].float()
        angle, trans = sample_obj[..., :6].reshape(-1,6), sample_obj[..., 6:9].reshape(-1,3)
        
        
        torch_rot = rotation_6d_to_matrix(angle)
        obj_points = torch.bmm(vertices.float(), torch_rot.transpose(1, 2)) + trans[:, None, :]
        bps_obj_times = bps_obj.repeat(trans.shape[0], 1, 1).float() + trans[:, None, :]
        bps_time = bps_torch.encode(x=obj_points, \
            feature_type=['deltas'], \
            custom_basis=bps_obj_times)['deltas'] # T X N X 3 
        pred_motions = torch.cat((sample[...,:231].cuda(),bps_time.reshape(sample.shape[0], sample.shape[1],128*3).cuda()),dim=-1)
        
        # pred_motions = pred_motions
        # gt_motions = gt_motions
        pred_motions = (pred_motions - mean_enc)/std_enc
        gt_motions = (gt_motions-mean_enc)/std_enc
        mlength_list =lengths

        em_pred=motionencoder(pred_motions,mlength_list).loc
        em=motionencoder(gt_motions,mlength_list).loc
        et=textencoder(text_list_gt).loc
        et_pred=et
        # sbert_feature=torch.from_numpy(sbert_encoder.encode(text_list_gt))
        num_poses+=sum(mlength_list)
        
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)
        # sbert_f_list.append(sbert_feature) # only once
        # text_f_list.append(et)

        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
        if batch_idx<mm_batch_num:
            print(batch_idx,'MMM')
            motion_multimodality_batch = []
            for _ in range(MAX_REPEAT):
                sample = sample_fn(
                    motion_model,
                    (args.batch_size, motion_model.njoints + 18+78*6, motion_model.nfeats,  max(lengths)),    # + 6 object pose
                    clip_denoised=False,
                    # clip_denoised=not args.predict_xstart,
                    model_kwargs=model_kwargs,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=True,
                    dump_steps=None,
                    noise=None,
                    const_noise=False,
                    cond_fn=None,
                )
                # sample = motions_copy
                sample = sample[:,:485]
                sample = sample.cpu().permute(0, 2, 3, 1).squeeze().float()
                # sample = sample.cpu().permute(0, 2, 3, 1).squeeze().float()
                sample_obj = sample[..., 476:485]
                angle, trans = sample_obj[..., :6].reshape(-1,6), sample_obj[..., 6:9].reshape(-1,3)
                torch_rot = rotation_6d_to_matrix(angle)
                obj_points = torch.bmm(vertices.float(), torch_rot.transpose(1, 2)) + trans[:, None, :]
                bps_obj_times = bps_obj.repeat(trans.shape[0], 1, 1).float() + trans[:, None, :]
                bps_time = bps_torch.encode(x=obj_points, \
                    feature_type=['deltas'], \
                    custom_basis=bps_obj_times)['deltas'] # T X N X 3 
                pred_motions = torch.cat((sample[...,:231].cuda(),bps_time.reshape(sample.shape[0], sample.shape[1],128*3).cuda()),dim=-1)
                # pred_motions = pred_motions
                pred_motions = (pred_motions - mean_enc)/std_enc
                mm_motion_feature=motionencoder(pred_motions,mlength_list).loc ## BS,D

                motion_multimodality_batch.append(mm_motion_feature.unsqueeze(1)) ## BS,1,D
                # break
            motion_multimodality_batch = torch.cat(motion_multimodality_batch, dim=1) #(bs, NUM_REPEAT, d)
        motion_multimodality.append(motion_multimodality_batch)
        
    
    ## insert
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy() ## BS*MM_BATCH_NUM,NUM_REPEAT,D
    multimodality = calculate_multimodality(motion_multimodality, mutlimodality_select_times)
    l1_dist=multimodality


    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()



    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 50)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 50)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    #l1_dist = l1_dist / num_poses

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = "--> \t Eva. Re %d:, FID. %.4f, Diversity Real. %.4f, Diversity. %.4f, R_precision_real. (%.4f, %.4f, %.4f), R_precision. (%.4f, %.4f, %.4f), matching_real. %.4f, matching_pred. %.4f, mae. %.4f" % \
          (repeat_id, fid, diversity_real, diversity, R_precision_real[0], R_precision_real[1], R_precision_real[2],
           R_precision[0], R_precision[1], R_precision[2], matching_score_real, matching_score_pred, l1_dist)
    # logger.info(msg)
    print(msg)
    return fid, diversity, R_precision, matching_score_pred, l1_dist





if __name__ == "__main__":
    # model=SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
    args = generate_args()
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 300 
    fps = 30
    n_frames = min(max_frames, int(args.motion_length*fps))
    dataset = args.dataset
    dist_util.setup_dist(args.device)
    # bps 
    bps_torch = bps_torch()
    bps_obj = np.load("./assets/eval/bps_basis_set_128_1.npy")
    bps_obj = torch.from_numpy(bps_obj)
    val_loader = load_motion_dataset(args, max_frames, n_frames)
    motion_model, motion_diffusion = load_model(args, val_loader, dist_util.dev(), ModelClass=used_model, DiffusionClass=LocalMotionDiffusion, diff_steps=1000,model_path=args.model_path)
    
    ########################### eval model ####################################
    std_path = "./assets/eval/std_train_markersbps.npy"
    std_enc = np.load(std_path)
    std_enc = torch.from_numpy(std_enc).float().cuda()
    mean_path = "./assets/eval/mean_train_markersbps.npy"
    mean_enc = np.load(mean_path)
    mean_enc = torch.from_numpy(mean_enc).float().cuda()
    eval_model_epoch = "2099"
    eval_type = "markersbps"
    path=f"./assets/eval/{eval_type}.ckpt"
    A=torch.load(path)
    STAT_DICT=A['state_dict']
    filtered_dict = {k[12:]: v for k, v in STAT_DICT.items() if k.startswith("textencoder")}
    filtered_dict2 = {k[14:]: v for k, v in STAT_DICT.items() if k.startswith("motionencoder")}
    #A['state_dict']=filtered_dict
    modelpath = 'distilbert-base-uncased'

    textencoder = DistilbertActorAgnosticEncoder(modelpath,latent_dim=256, 
    ff_size=1024,num_layers=4).cuda()
    textencoder.load_state_dict(filtered_dict)

    ## nfeats 231: motion dimension, 128*3: bps dimension
    motionencoder=ActorAgnosticEncoder(nfeats=231+128*3,vae=True,latent_dim=256,ff_size=1024,
                                    num_layers=4).cuda()
    motionencoder.load_state_dict(filtered_dict2)
    motionencoder.eval()
    textencoder.eval()
    
    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    mm = []
    repeat_time = 2
    for i in range(repeat_time):
        with torch.no_grad():
            best_fid, best_div, Rprecision, best_matching, best_mm = \
                eval_t2hoi(val_loader, motion_model, motion_diffusion, textencoder, motionencoder,std_enc,mean_enc,repeat_id=i)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(Rprecision[0])
        top2.append(Rprecision[1])
        top3.append(Rprecision[2])
        matching.append(best_matching)
        mm.append(best_mm)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    mm = np.array(mm)

    f = open(f'./sample/{eval_type}_{eval_model_epoch}_{dataset}_{name}_{niter}_correct_eval.txt', 'w')
    print('final result:')
    print('final result:', file=f, flush=True)

    msg_final = f"\tFID: {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tDiversity: {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tTOP1: {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMatching: {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}\n" \
                f"\tMultimodality:{np.mean(mm):.3f}, conf.{np.std(mm) * 1.96 / np.sqrt(repeat_time):.3f}\n\n"
    # logger.info(msg_final)
    print(msg_final)
    print(msg_final, file=f, flush=True)
    

