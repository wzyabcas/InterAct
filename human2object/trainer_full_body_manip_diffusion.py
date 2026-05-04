import argparse
import os
import numpy as np
import yaml
from pathlib import Path

import wandb

import torch
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch.utils import data

from ema_pytorch import EMA

from manip.data.hand_foot_dataset_all import MarkerManipDataset
from manip.model.transformer_fullbody_cond_diffusion_model import CondGaussianDiffusion as FullBodyCondGaussianDiffusion

from scipy.spatial.transform import Rotation
import pytorch_lightning as pl


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer(object):
    def __init__(
        self,
        opt,
        diffusion_model,
        *,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=10000000,
        gradient_accumulate_every=1,
        amp=False,
        step_start_ema=2000,
        ema_update_every=10,
        save_and_sample_every=10000,
        results_folder='./results',
        use_wandb=True,
        load_num=None,
        loss_weights=None,
    ):
        super().__init__()
        self.multi_task = getattr(opt, 'multi_task', False)
        self.loss_weights = loss_weights
        self.token_obj_geom = getattr(opt, 'token_obj_geom', False)
        self.use_quat = getattr(opt, 'use_quat', False)

        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(config=opt, project=opt.wandb_pj_name, entity=opt.entity,
                       name=opt.exp_name, dir=opt.save_dir)

        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.optimizer = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)

        self.results_folder = results_folder.replace("weights", f"weights_interval{save_and_sample_every}")
        os.makedirs(self.results_folder, exist_ok=True)

        self.vis_folder = results_folder.replace("weights", "vis_res")

        self.opt = opt

        self.window = opt.window

        self.use_object_split = getattr(self.opt, 'use_object_split', False)

        self.data_root_folder = self.opt.data_root_folder
        self.for_quant_eval = self.opt.for_quant_eval

        self.prep_dataloader(window_size=opt.window, load_num=load_num)

        self.use_bps_info = getattr(self.opt, "use_bps_info", False)
        self.use_human_contact_dist = getattr(self.opt, "use_human_contact_dist", False)
        self.eval_save_path = getattr(self.opt, "eval_save_path", None)
        if self.eval_save_path is not None:
            self.eval_save_path = os.path.join(self.eval_save_path, self.opt.exp_name)
            os.makedirs(self.eval_save_path, exist_ok=True)

        if getattr(opt, 'load_last_ckpt', False):
            try:
                weights = os.listdir(self.results_folder)
                weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
                weight_path = max(weights_paths, key=os.path.getctime)
                print(f"Loaded weight: {weight_path}")
                milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
                self.load(milestone)
            except:
                print("No weights found. Training from scratch.")

    def prep_dataloader(self, window_size, load_num=None):
        bps_dim = getattr(self.opt, "bps_dim", 1024)
        correct_data = getattr(self.opt, "correct_data", False)
        if not self.for_quant_eval:
            print("Loading train dataset...")
            train_dataset = MarkerManipDataset(train=True, data_root_folder=self.data_root_folder,
                window=window_size, use_object_splits=self.use_object_split, load_num=load_num,
                use_all_data=self.opt.use_all_data, bps_dim=bps_dim, corrected_data=correct_data,
                split_train_val=self.opt.split_train_val)
            self.ds = train_dataset
            self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size,
                shuffle=True, pin_memory=True, num_workers=8))

        print("Loading val dataset...")
        val_dataset = MarkerManipDataset(train=False, data_root_folder=self.data_root_folder,
            window=window_size, use_object_splits=self.use_object_split, load_num=load_num,
            use_all_data=self.opt.use_all_data, bps_dim=bps_dim, corrected_data=correct_data,
            split_train_val=self.opt.split_train_val)
        self.val_ds = val_dataset
        self.val_dl = cycle(data.DataLoader(self.val_ds, batch_size=1, # change back to self.batch_size
            shuffle=True, pin_memory=True, num_workers=8, drop_last=True))

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        save_path = os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt')
        torch.save(data, save_path)
        return save_path

    def load(self, milestone, pretrained_path=None):
        if pretrained_path is None:
            pretrained_path = os.path.join(self.results_folder, 'model-'+str(milestone)+'.pt')

        print("loading from: ", pretrained_path)
        data = torch.load(pretrained_path)

        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema.load_state_dict(data['ema'], strict=False)
        self.scaler.load_state_dict(data['scaler'])

        return pretrained_path

    def get_data(self, data_dict):
        obj_com = data_dict['obj_trans'].cuda()
        if not self.use_quat:
            obj_angles = data_dict['obj_angles'].cuda()
        else:
            obj_angles = data_dict['obj_quat'].cuda()
        data = torch.cat([obj_com, obj_angles], dim=-1)
        if not self.multi_task:
            raise NotImplementedError
        else:
            ori_data_cond = {}
            ori_data_cond['obj_canon_bps'] = data_dict['obj_canon_bps_data'].cuda()
            ori_data_cond['marker_pos'] = data_dict['motion'][..., :77*3].cuda()
            ori_data_cond['unnorm_marker_pos'] = data_dict['markers'].cuda()
            if self.use_human_contact_dist:
                dist = data_dict['closest_vectors'].cuda().norm(dim=-1)
                data = torch.cat([data, dist], dim=-1)

        return data, ori_data_cond

    def train(self):
        init_step = self.step
        for idx in range(init_step, self.train_num_steps):
            self.optimizer.zero_grad()

            nan_exists = False
            for i in range(self.gradient_accumulate_every):
                data_dict = next(self.dl)
                data, ori_data_cond = self.get_data(data_dict)

                cond_mask = None
                actual_seq_len = data_dict['seq_len'] + 1
                mask_length = self.window + (2 if self.token_obj_geom else 1)
                tmp_mask = torch.arange(mask_length).expand(data.shape[0],
                                mask_length) < actual_seq_len[:, None].repeat(1, mask_length) + (2 if self.token_obj_geom else 1)
                padding_mask = tmp_mask[:, None, :].to(data.device)

                with autocast(enabled=self.amp):
                    loss_diffusion = self.model(data, ori_data_cond, cond_mask, padding_mask)

                    if self.loss_weights is None:
                        loss = loss_diffusion
                    else:
                        loss = loss_diffusion['motion']
                        print("origin loss = ", loss)
                        for key in self.loss_weights:
                            loss += self.loss_weights[key] * loss_diffusion[key]
                        print("curr loss = ", loss)

                    if torch.isnan(loss).item():
                        print('WARNING: NaN loss. Skipping to next data...')
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                    parameters = [p for p in self.model.parameters() if p.grad is not None]
                    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0).to(data.device) for p in parameters]), 2.0)
                    if torch.isnan(total_norm):
                        print('WARNING: NaN gradients. Skipping to next data...')
                        nan_exists = True
                        torch.cuda.empty_cache()
                        continue

                    if self.use_wandb:
                        log_dict = {"Train/Loss/Total Loss": loss.item()}
                        wandb.log(log_dict, step=self.step)
                        if isinstance(loss_diffusion, dict):
                            for key in loss_diffusion:
                                wandb.log({"Train/Loss/"+key: loss_diffusion[key].item()}, step=self.step)

                    if idx % 10 == 0 and i == 0:
                        print("Step: {0}".format(idx))
                        LossStr = "Loss: %.4f" % (loss.item())
                        if isinstance(loss_diffusion, dict):
                            for key in loss_diffusion:
                                LossStr += ", " + key + ": %.4f" % (loss_diffusion[key].item())
                        print(LossStr)

            if nan_exists:
                continue

            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.ema.update()

            if self.step != 0 and self.step % 10 == 0:
                self.ema.ema_model.eval()

                with torch.no_grad():
                    val_data_dict = next(self.val_dl)
                    val_data, ori_data_cond = self.get_data(val_data_dict)

                    cond_mask = None
                    actual_seq_len = val_data_dict['seq_len'] + 1
                    mask_length = self.window + (2 if self.token_obj_geom else 1)
                    tmp_mask = torch.arange(mask_length).expand(val_data.shape[0],
                                    mask_length) < actual_seq_len[:, None].repeat(1, mask_length) + (2 if self.token_obj_geom else 1)
                    padding_mask = tmp_mask[:, None, :].to(val_data.device)

                    val_loss_diffusion = self.model(val_data, ori_data_cond, cond_mask, padding_mask)
                    if self.loss_weights is None:
                        val_loss = val_loss_diffusion
                    else:
                        val_loss = loss_diffusion['motion']
                        for key in self.loss_weights:
                            val_loss += self.loss_weights[key] * val_loss_diffusion[key]
                    if self.use_wandb:
                        val_log_dict = {"Validation/Loss/Total Loss": val_loss.item()}
                        wandb.log(val_log_dict, step=self.step)
                        if isinstance(val_loss_diffusion, dict):
                            for key in val_loss_diffusion:
                                wandb.log({"Validation/Loss/"+key: val_loss_diffusion[key].item()}, step=self.step)

                    milestone = self.step // self.save_and_sample_every

                    if self.step % self.save_and_sample_every == 0:
                        self.save(milestone)

            self.step += 1

        print('training complete')

        if self.use_wandb:
            wandb.run.finish()

    def load_weights_from_resfolder(self, ms=None):
        weights = os.listdir(self.results_folder)
        weights_paths = [os.path.join(self.results_folder, weight) for weight in weights]
        weight_path = max(weights_paths, key=os.path.getctime)
        milestone = weight_path.split("/")[-1].split("-")[-1].replace(".pt", "")
        if self.opt.milestone is not None:
            milestone = self.opt.milestone
        if ms is not None:
            milestone = ms
        pretrained_path = self.load(milestone)
        return milestone, pretrained_path

    def _compute_obj_verts(self, all_res_list, gt_res_list, data_dict):
        """Compute predicted and GT object vertices for all sequences in the batch."""
        num_seq = all_res_list.shape[0]
        mo_dim = 7 if self.use_quat else 6
        pred_trans = all_res_list[:, :, :3]
        pred_angle = all_res_list[:, :, 3:mo_dim]
        gt_trans = gt_res_list[:, :, :3]
        gt_angle = gt_res_list[:, :, 3:mo_dim]

        if self.use_quat:
            B, T, _ = pred_angle.shape
            pred_angle_np = pred_angle.detach().cpu().numpy()
            gt_angle_np = gt_angle.detach().cpu().numpy()
            pred_angle_list, gt_angle_list = [], []
            for i in range(B):
                seq_len = data_dict['seq_len'][i]
                tmp_pred = Rotation.from_quat(pred_angle_np[i, :seq_len]).as_rotvec()
                tmp_gt = Rotation.from_quat(gt_angle_np[i, :seq_len]).as_rotvec()
                if seq_len < T:
                    tmp_pred = np.concatenate([tmp_pred, np.zeros((T-seq_len, 3))], axis=0)
                    tmp_gt = np.concatenate([tmp_gt, np.zeros((T-seq_len, 3))], axis=0)
                pred_angle_list.append(tmp_pred)
                gt_angle_list.append(tmp_gt)
            pred_angle = torch.from_numpy(np.stack(pred_angle_list, axis=0)).float().cuda()
            gt_angle = torch.from_numpy(np.stack(gt_angle_list, axis=0)).float().cuda()

        pred_verts_list = []
        gt_verts_list = []
        for idx in range(num_seq):
            obj_name = data_dict['obj_name'][idx]
            dataset = data_dict['dataset'][idx]
            obj_mesh_path = os.path.join(self.val_ds.data_root_folder, dataset, "objects", f"{obj_name}/sample_points.npy")
            cur_obj_verts_ = np.load(obj_mesh_path)

            np_pred_angles = pred_angle[idx].detach().cpu().numpy()
            np_pred_trans = pred_trans[idx].detach().cpu().numpy()
            angle_matrix = Rotation.from_rotvec(np_pred_angles).as_matrix()
            cur_pred = cur_obj_verts_[None, ...]
            cur_pred = np.matmul(cur_pred, np.transpose(angle_matrix, (0, 2, 1))) + np_pred_trans[:, None, :]
            pred_verts_list.append(torch.from_numpy(cur_pred).float())

            np_gt_angles = gt_angle[idx].detach().cpu().numpy()
            np_gt_trans = gt_trans[idx].detach().cpu().numpy()
            angle_matrix = Rotation.from_rotvec(np_gt_angles).as_matrix()
            cur_gt = cur_obj_verts_[None, ...]
            cur_gt = np.matmul(cur_gt, np.transpose(angle_matrix, (0, 2, 1))) + np_gt_trans[:, None, :]
            gt_verts_list.append(torch.from_numpy(cur_gt).float())

        return pred_verts_list, gt_verts_list

    def cond_sample_res(self, load_weights=True, milestone=None, pretrained_path=None):
        if load_weights:
            milestone, pretrained_path = self.load_weights_from_resfolder(ms=milestone)
        else:
            assert pretrained_path is not None, "Need to provide pretrained_path if not loading weights."
            assert milestone is not None, "Need to provide milestone if not loading weights."

        self.ema.ema_model.eval()
        if milestone is not None and self.eval_save_path is not None:
            eval_motion_path = os.path.join(self.eval_save_path, f"ms_{milestone}")
            os.makedirs(eval_motion_path, exist_ok=True)
        else:
            eval_motion_path = None

        bs = 32 if self.for_quant_eval else 1
        test_loader = torch.utils.data.DataLoader(
            self.val_ds, batch_size=bs, shuffle=False,
            num_workers=8, pin_memory=True, drop_last=False)

        num_samples_per_seq = 5 if self.for_quant_eval else 1

        if self.for_quant_eval:
            log_dir = '/'.join(self.vis_folder.split('/')[:-1]) + f"/quant_eval_fullbodymodel_bs256_ms={milestone}.txt"
            with open(log_dir, "w") as f:
                f.write("Start Quantitative Evaluation\n")
                f.write("Checkpoint: {}\n".format(pretrained_path))
                f.write("data root folder: {}\n".format(self.data_root_folder))

        num_sample = 100 if not self.for_quant_eval else 1e9
        with torch.no_grad():
            for s_idx, val_data_dict in enumerate(test_loader):
                if s_idx >= num_sample:
                    break
                val_data, ori_data_cond = self.get_data(val_data_dict)

                cond_mask = None
                actual_seq_len = val_data_dict['seq_len'] + 1
                mask_length = self.window + (2 if self.token_obj_geom else 1)
                tmp_mask = torch.arange(mask_length).expand(val_data.shape[0],
                                mask_length) < actual_seq_len[:, None].repeat(1, mask_length) + (2 if self.token_obj_geom else 1)
                padding_mask = tmp_mask[:, None, :].to(val_data.device)

                max_num = 1 if not self.for_quant_eval else val_data.shape[0]
                for sample_idx in range(num_samples_per_seq):
                    if eval_motion_path is not None and check_exist(eval_motion_path, s_idx, sample_idx, val_data_dict):
                        total_new_gen = False
                        print(f"Passing existing data, batch_{s_idx}_{sample_idx}")
                        continue

                    all_res_list = self.ema.ema_model.sample(val_data, ori_data_cond,
                                                             cond_mask=cond_mask, padding_mask=padding_mask)

                    pred_vert_list, gt_vert_list = self._compute_obj_verts(
                        all_res_list[:max_num], val_data[:max_num], val_data_dict)

                    if self.for_quant_eval:
                        num_seq = all_res_list.shape[0]
                        for i in range(num_seq):
                            T = val_data_dict['seq_len'][i]
                            marker_i = val_data_dict['markers'][i, :T]
                            pred_verts = pred_vert_list[i][:T]
                            gt_verts = gt_vert_list[i][:T]

                            if eval_motion_path is not None:
                                curr_obj_angles = val_data_dict['obj_angles'][i].detach().cpu().numpy()
                                curr_obj_trans = val_data_dict['obj_trans'][i].detach().cpu().numpy()
                                curr_seq_name = val_data_dict['seq_name'][i]
                                obj_name = val_data_dict['obj_name'][i]
                                dataset = val_data_dict['dataset'][i]

                                mo_dim = 7 if self.use_quat else 6
                                pred_trans_i = all_res_list[i, :T, :3].detach().cpu().numpy()
                                pred_angle_i = all_res_list[i, :T, 3:mo_dim].detach().cpu().numpy()
                                if self.use_quat:
                                    pred_angle_i = Rotation.from_quat(pred_angle_i).as_rotvec()

                                save_path = os.path.join(eval_motion_path, f"batch_{s_idx}_{i:03}-th_{curr_seq_name}_dataset_{dataset}_sample_{sample_idx}.npz")
                                np.savez(save_path,
                                    pred_obj_angle=pred_angle_i,
                                    pred_obj_trans=pred_trans_i,
                                    marker=marker_i.detach().cpu().numpy(),
                                    gt_obj_angle=curr_obj_angles,
                                    gt_obj_trans=curr_obj_trans,
                                    obj_name=obj_name,
                                    seq_name=curr_seq_name,
                                    dataset=dataset,
                                    pred_verts=pred_verts.detach().cpu().numpy(),
                                    gt_verts=gt_verts.detach().cpu().numpy(),
                                )


def check_exist(eval_motion_path, s_idx, sample_idx, val_data_dict):
    num_seq = val_data_dict['seq_len'].shape[0]
    for i in range(num_seq):
        curr_seq_name = val_data_dict['seq_name'][i]
        dataset = val_data_dict['dataset'][i]
        save_path = os.path.join(eval_motion_path, f"batch_{s_idx}_{i:03}-th_{curr_seq_name}_dataset_{dataset}_sample_{sample_idx}.npz")
        if not os.path.exists(save_path):
            return False
    return True


def get_loss_weights(opt):
    loss_weights = {}
    if getattr(opt, 'w_humanContactDist', False):
        loss_weights['humanContactDist'] = opt.w_humanContactDist
    if loss_weights == {}:
        loss_weights = None
    return loss_weights


def get_repr_dim(opt):
    repr_dim = 6 if not opt.use_quat else 7
    if opt.multi_task:
        if opt.use_human_contact_dist:
            repr_dim += 77
    print("repr_dim: ", repr_dim)
    return repr_dim


def run_train(opt, device):
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=True)

    repr_dim = get_repr_dim(opt)
    loss_weights = get_loss_weights(opt)

    loss_type = "l1"

    diffusion_model = FullBodyCondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model,
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000,
                objective="pred_x0", loss_type=loss_type,
                batch_size=opt.batch_size, seperate_obj_contact=False,
                bps_dim=opt.bps_dim, multi_task=opt.multi_task, loss_weights=loss_weights)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=opt.batch_size,
        train_lr=opt.learning_rate,
        train_num_steps=400050,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=True,
        results_folder=str(wdir),
        save_and_sample_every=opt.save_and_sample_every,
        load_num=opt.load_num,
        loss_weights=loss_weights,
    )

    trainer.train()

    torch.cuda.empty_cache()


def run_sample(opt, device):
    save_dir = Path(opt.save_dir)
    wdir = save_dir / 'weights'

    repr_dim = get_repr_dim(opt)

    loss_type = "l1"

    diffusion_model = FullBodyCondGaussianDiffusion(opt, d_feats=repr_dim, d_model=opt.d_model,
                n_dec_layers=opt.n_dec_layers, n_head=opt.n_head, d_k=opt.d_k, d_v=opt.d_v,
                max_timesteps=opt.window+1, out_dim=repr_dim, timesteps=1000,
                objective="pred_x0", loss_type=loss_type,
                batch_size=opt.batch_size, seperate_obj_contact=False,
                bps_dim=opt.bps_dim, multi_task=opt.multi_task)

    diffusion_model.to(device)

    trainer = Trainer(
        opt,
        diffusion_model,
        train_batch_size=2,
        train_lr=opt.learning_rate,
        train_num_steps=400050,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=True,
        results_folder=str(wdir),
        save_and_sample_every=10000,
        use_wandb=False,
        load_num=opt.load_num,
    )

    trainer.cond_sample_res(load_weights=True, milestone=opt.milestone)

    torch.cuda.empty_cache()


def parse_opt():
    pl.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', default='runs/train', help='output folder for weights and visualizations')
    parser.add_argument('--wandb_pj_name', type=str, default='wandb_proj_name', help='wandb project name')
    parser.add_argument('--entity', default='wandb_account_name', help='W&B entity')
    parser.add_argument('--exp_name', default='stage1_exp_out', help='save to project/exp_name')
    parser.add_argument('--device', default='0', help='cuda device')

    parser.add_argument('--window', type=int, default=120, help='horizon')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='learning rate')

    parser.add_argument('--n_dec_layers', type=int, default=4, help='the number of decoder layers')
    parser.add_argument('--n_head', type=int, default=4, help='the number of heads in self-attention')
    parser.add_argument('--d_k', type=int, default=256, help='the dimension of keys in transformer')
    parser.add_argument('--d_v', type=int, default=256, help='the dimension of values in transformer')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of intermediate representation in transformer')

    parser.add_argument('--data_root_folder', default='data', help='root folder for dataset')
    parser.add_argument('--load_num', default=None, type=int)

    parser.add_argument('--use_all_data', action='store_true')
    parser.add_argument('--use_object_split', action='store_true')

    parser.add_argument('--use_bps_info', default=False, action='store_true')
    parser.add_argument('--use_human_contact_dist', default=False, action='store_true')
    parser.add_argument('--multi_task', default=False, action='store_true')

    parser.add_argument('--load_last_ckpt', default=False, action='store_true')
    parser.add_argument('--bps_dim', default=1024, type=int, choices=[1024, 256])
    parser.add_argument('--milestone', default=None, type=str)
    parser.add_argument('--save_and_sample_every', default=10000, type=int)

    parser.add_argument('--w_humanContactDist', default=None, type=float)

    parser.add_argument('--token_obj_geom', default=False, action='store_true')
    parser.add_argument('--use_quat', default=False, action='store_true')
    parser.add_argument('--split_train_val', default=False, action='store_true')

    parser.add_argument('--test_sample_res', action='store_true')
    parser.add_argument('--for_quant_eval', action='store_true')
    parser.add_argument('--eval_save_path', default=None, type=str)

    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    opt.save_dir = os.path.join(opt.project, opt.exp_name)
    opt.exp_name = opt.save_dir.split('/')[-1]
    device = torch.device(f"cuda:{opt.device}" if torch.cuda.is_available() else "cpu")
    if opt.test_sample_res:
        run_sample(opt, device)
    else:
        run_train(opt, device)
