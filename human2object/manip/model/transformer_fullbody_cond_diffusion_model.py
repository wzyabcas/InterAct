import math

from tqdm.auto import tqdm

from einops import rearrange, reduce

from inspect import isfunction

import torch
from torch import nn
import torch.nn.functional as F

from manip.model.transformer_module import Decoder


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TransformerDiffusionModel(nn.Module):
    def __init__(
        self,
        d_input_feats,
        d_feats,
        d_model,
        n_dec_layers,
        n_head,
        d_k,
        d_v,
        max_timesteps,
        token_obj_geom=False,
    ):
        super().__init__()

        self.d_feats = d_feats
        self.d_model = d_model
        self.n_head = n_head
        self.n_dec_layers = n_dec_layers
        self.d_k = d_k
        self.d_v = d_v
        self.max_timesteps = max_timesteps
        self.token_obj_geom = token_obj_geom

        # Input: BS X D X T
        # Output: BS X T X D'
        self.motion_transformer = Decoder(d_feats=d_input_feats, d_model=self.d_model, \
            n_layers=self.n_dec_layers, n_head=self.n_head, d_k=self.d_k, d_v=self.d_v, \
            max_timesteps=self.max_timesteps, use_full_attention=True, token_obj_geom=token_obj_geom)

        self.linear_out = nn.Linear(self.d_model, self.d_feats)

        # For noise level t embedding
        dim = 64
        learned_sinusoidal_dim = 16
        time_dim = dim * 4

        learned_sinusoidal_cond = False
        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        sinu_pos_emb = SinusoidalPosEmb(dim)
        fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, d_model)
        )

    def forward(self, src, noise_t, condition, padding_mask=None, obj_geom=None):
        # src: BS X T X D
        # noise_t: int

        src = torch.cat((src, condition), dim=-1)

        noise_t_embed = self.time_mlp(noise_t) # BS X d_model
        noise_t_embed = noise_t_embed[:, None, :] # BS X 1 X d_model

        bs = src.shape[0]
        offset = 2 if self.token_obj_geom else 1
        num_steps = src.shape[1] + offset

        if padding_mask is None:
            # In training, no need for masking
            padding_mask = torch.ones(bs, 1, num_steps).to(src.device).bool() # BS X 1 X timesteps

        # Get position vec for position-wise embedding
        pos_vec = torch.arange(num_steps) # timesteps
        pos_vec = pos_vec[None, None, :].to(src.device).repeat(bs, 1, 1) # BS X 1 X timesteps

        if self.token_obj_geom:
            assert obj_geom is not None
            obj_geom = obj_geom[:, None, :] # BS X 1 X d_model
            token_embed = torch.cat((noise_t_embed, obj_geom), dim=1)
        else:
            token_embed = noise_t_embed

        data_input = src.transpose(1, 2).detach() # BS X D X T
        feat_pred, _ = self.motion_transformer(data_input, padding_mask, pos_vec, obj_embedding=token_embed)

        output = self.linear_out(feat_pred[:, offset:]) # BS X T X D

        return output # predicted noise, the same size as the input

class CondGaussianDiffusion(nn.Module):
    def __init__(
        self,
        opt,
        d_feats,
        d_model,
        n_head,
        n_dec_layers,
        d_k,
        d_v,
        max_timesteps,
        out_dim,
        timesteps = 1000,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine',
        p2_loss_weight_gamma = 0.,
        p2_loss_weight_k = 1,
        batch_size=None,
        seperate_obj_contact=True,
        multi_task=False,
        bps_dim=1024,
        loss_weights=None,
    ):
        super().__init__()
        self.use_bps_info = getattr(opt, 'use_bps_info', False)
        self.multi_task = multi_task
        self.loss_weights = loss_weights
        self.token_obj_geom = getattr(opt, 'token_obj_geom', False)
        self.use_quat = getattr(opt, 'use_quat', False)

        cond_dim = 77*3 + (0 if self.token_obj_geom else 256)

        self.denoise_fn = TransformerDiffusionModel(d_input_feats=d_feats+cond_dim, d_feats=d_feats, d_model=d_model, n_head=n_head, \
                    d_k=d_k, d_v=d_v, n_dec_layers=n_dec_layers, max_timesteps=max_timesteps, token_obj_geom=self.token_obj_geom)
        # Input condition and noisy motion, noise level t, predict gt motion

        self.objective = objective
        bps_out_dim = d_model if self.token_obj_geom else 256
        self.bps_encoder = nn.Sequential(
            nn.Linear(in_features=bps_dim*3, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=bps_out_dim),
        )
        try:
            print("self.bps_encoder: ", self.bps_encoder)
        except:
            pass
        self.seq_len = max_timesteps - 1
        self.out_dim = out_dim

        if beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting

        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, x_cond, padding_mask, clip_denoised, obj_geom=None):
        model_output = self.denoise_fn(x, t, x_cond, padding_mask, obj_geom=obj_geom)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, x_cond, padding_mask=None, clip_denoised=True, obj_geom=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, x_cond=x_cond, \
            padding_mask=padding_mask, clip_denoised=clip_denoised, obj_geom=obj_geom)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, x_start, x_cond, padding_mask=None, obj_geom=None):
        device = self.betas.device

        b = shape[0]
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), x_cond, padding_mask=padding_mask, obj_geom=obj_geom)

        return x # BS X T X D

    @torch.no_grad()
    def sample(self, x_start, x_cond, cond_mask=None, padding_mask=None):
        # naive conditional sampling by replacing the noisy prediction with input target data.
        self.denoise_fn.eval()
        if self.use_bps_info:
            self.bps_encoder.eval()

        x_cond_processed = self.process_cond(x_cond)
        if self.token_obj_geom:
            obj_canon_bps = x_cond['obj_canon_bps']
            obj_canon_bps = rearrange(obj_canon_bps, 'b n d -> b (n d)')
            obj_bps_feat = self.bps_encoder(obj_canon_bps) # BS x 1 x 256
            sample_res = self.p_sample_loop(x_start.shape, \
                x_start, x_cond_processed, padding_mask, obj_geom=obj_bps_feat)
        else:
            sample_res = self.p_sample_loop(x_start.shape, \
                x_start, x_cond_processed, padding_mask)
        # BS X T X D

        self.denoise_fn.train()
        if self.use_bps_info:
            self.bps_encoder.train()

        return sample_res

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, x_cond, t, noise=None, padding_mask=None, obj_geom=None):
        # x_start: BS X T X D
        # x_cond: BS X T X D_cond
        # padding_mask: BS X 1 X T
        b, timesteps, d_input = x_start.shape # BS X T X D(3+n_joints*4)
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise) # noisy motion in noise level t.

        model_out = self.denoise_fn(x, t, x_cond, padding_mask, obj_geom=obj_geom)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')

        offset = (2 if self.token_obj_geom else 1)
        pad_msk = padding_mask[:, 0, offset:][:, :, None]
        if padding_mask is not None:
            loss = self.loss_fn(model_out, target, reduction = 'none') * pad_msk
        else:
            loss = self.loss_fn(model_out, target, reduction = 'none') # BS X T X D

        if self.loss_weights is None:
            loss = reduce(loss, 'b ... -> b (...)', 'mean') # BS X (T*D)
            loss = loss * extract(self.p2_loss_weight, t, loss.shape)
            return loss.mean()
        else:
            loss = {}
            motion_dim = 6 if not self.use_quat else 7
            pred_motion, gt_motion, start = model_out[..., :motion_dim], target[..., :motion_dim], motion_dim
            loss_motion = self.loss_fn(pred_motion, gt_motion, reduction = 'none') * pad_msk
            loss_motion = reduce(loss_motion, 'b ... -> b (...)', 'mean')
            loss_motion = loss_motion * extract(self.p2_loss_weight, t, loss_motion.shape)
            loss['motion'] = loss_motion.mean()
            if 'humanContactDist' in self.loss_weights.keys():
                pred_humanVec, gt_humanVec = model_out[..., start:start+77], target[..., start:start+77]
                start += 77
                loss_humanVec = self.loss_fn(pred_humanVec, gt_humanVec, reduction = 'none') * pad_msk
                loss_humanVec = reduce(loss_humanVec, 'b ... -> b (...)', 'mean')
                loss_humanVec = loss_humanVec * extract(self.p2_loss_weight, t, loss_humanVec.shape)
                loss['humanContactDist'] = loss_humanVec.mean()
            return loss

    def forward(self, x_start, x_cond, cond_mask=None, padding_mask=None):
        # x_start: BS X T X D
        # ori_x_cond: BS X T X D'
        bs = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, (bs,), device=x_start.device).long()
        x_cond_processed = self.process_cond(x_cond)
        if self.token_obj_geom:
            obj_canon_bps = x_cond['obj_canon_bps']
            obj_canon_bps = rearrange(obj_canon_bps, 'b n d -> b (n d)')
            obj_bps_feat = self.bps_encoder(obj_canon_bps) # BS x 1 x 256
            curr_loss = self.p_losses(x_start, x_cond_processed, t, padding_mask=padding_mask, obj_geom=obj_bps_feat)
        else:
            curr_loss = self.p_losses(x_start, x_cond_processed, t, padding_mask=padding_mask)

        return curr_loss

    def process_cond(self, x_cond):
        if isinstance(x_cond, dict):
            marker_pos = x_cond['marker_pos'] # BS X T X (77 X 3)
            x_cond_processed = marker_pos

            if not self.token_obj_geom:
                obj_canon_bps = x_cond['obj_canon_bps']
                obj_canon_bps = rearrange(obj_canon_bps, 'b n d -> b 1 (n d)')
                obj_bps_feat = self.bps_encoder(obj_canon_bps) # BS x 1 x 256
                obj_bps_feat = obj_bps_feat.repeat(1, marker_pos.shape[1], 1)
                x_cond_processed = torch.cat((x_cond_processed, obj_bps_feat), dim=-1)
        else:
            # the default setting used in OMOMO, x_cond is the hand joint positions
            x_cond_processed = x_cond
        return x_cond_processed
