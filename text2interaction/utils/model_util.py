from model.mdm import MDM
from model.cfg_sampler import wrap_model
from diffusion import gaussian_diffusion as gd
from diffusion.respace import  space_timesteps
from utils.parser_util import get_cond_mode
from diffusion.gaussian_diffusion import DiffusionConfig, LocalMotionDiffusion, AffordDiffusion
import torch

def load_model_wo_clip(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])


def load_model(args, data, device, ModelClass=MDM, DiffusionClass=LocalMotionDiffusion, diff_steps=1000,  model_path=None):
    model, diffusion = create_model_and_diffusion(args, data, ModelClass=ModelClass, DiffusionClass=DiffusionClass, diff_steps=diff_steps) 
    print(f"Loading pretrained checkpoints from [{model_path}]...")
    state_dict = torch.load(model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)
    model.to(device)
    model.eval()  # disable random masking
    model = wrap_model(model, args)
    return model, diffusion



def load_pretrained_mdm(model, state_dict):
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.')for k in missing_keys])


def load_split_mdm(model, state_dict, cutting_point):
    new_state_dict = {}
    orig_trans_prefix = 'seqTransEncoder.'
    for k, v in state_dict.items():
        if k.startswith(orig_trans_prefix):
            orig_layer = int(k.split('.')[2])
            orig_suffix = '.'.join(k.split('.')[3:])
            target_split = 'seqTransEncoder_start.' if orig_layer < cutting_point else 'seqTransEncoder_end.'
            target_layer = orig_layer if orig_layer < cutting_point else orig_layer - cutting_point
            new_k = target_split + 'layers.' + str(target_layer) + '.' + orig_suffix
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    # assert len(unexpected_keys) == 0
    # assert all([k.startswith('clip_model.') for k in missing_keys])




def create_model_and_diffusion(args, data, ModelClass=MDM, DiffusionClass=LocalMotionDiffusion, diff_steps=1000):

    model = ModelClass(**get_model_args(args, data))
    # model = MDMObj(**get_model_args(args, data))
    diffusion = create_gaussian_diffusion(args, diff_steps, DiffusionClass)
    return model, diffusion


def get_model_args(args, data):

    # default args
    clip_version = 'ViT-B/32'
    action_emb = 'tensor'
    cond_mode = get_cond_mode(args)
    if hasattr(data.dataset, 'num_actions'):
        num_actions = data.dataset.num_actions
    else:
        num_actions = 1
    
    data_rep = 'hml_vec'

    njoints = 476+18+78*6
    nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats, 'num_actions': num_actions,
            'translation': True, 'pose_rep': 'rot6d', 'glob': True, 'glob_rot': True,
            'latent_dim': 512, 'ff_size': 1024, 'num_layers': 8, 'num_heads': 4,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob, 'action_emb': action_emb, 'arch': args.arch,
            'emb_trans_dec': args.emb_trans_dec, 'clip_version': clip_version, 'dataset': args.dataset, 'args': args}

def create_gaussian_diffusion_mdm(args):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    if hasattr(args, 'lambda_target_loc'):
        lambda_target_loc = args.lambda_target_loc
    else:
        lambda_target_loc = 0.

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
    )



def create_gaussian_diffusion(args, diff_steps=1000, DiffusionClass=LocalMotionDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = diff_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]


    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        conf=DiffusionConfig(
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON
                             if not predict_xstart else
                             gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE
                 if not args.sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma else gd.ModelVarType.LEARNED_RANGE),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        ),
    )