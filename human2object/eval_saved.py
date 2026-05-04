import argparse
import os
import numpy as np

from tqdm import tqdm

from scipy.spatial.transform import Rotation
from evaluation_metrics import get_frobenious_norm_rot_only, compute_metrics_obj_marker
import pytorch_lightning as pl

from collections import defaultdict


def fetch_files(motion_path):
    grouped_files = defaultdict(list)
    for filename in os.listdir(motion_path):
        if filename.endswith(".npz"):
            prefix = "_".join(filename.split("_")[:-1])
            grouped_files[prefix].append(filename)

    for prefix in grouped_files:
        grouped_files[prefix].sort()
    sorted_prefixes = sorted(grouped_files.keys())
    grouped_files = [grouped_files[prefix] for prefix in sorted_prefixes]
    return grouped_files


def parse_opt():
    pl.seed_everything(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion_path', default='/work/nvme/bbsh/xiyanxu2/motion2obj_eval/omomoRaw_useQuat_tokenGeo_splitTrainVal_singleStageMultiTask_train_HumanmarkerDist_bps256_1108/ms_8/')
    parser.add_argument('--best_metric', default='mpvpe', choices=['mpvpe', 'mpmpe', 'contact_f1_score'])
    parser.add_argument('--contact_thre', default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    motion_path = opt.motion_path
    best_metric = opt.best_metric
    contact_threshold = float(opt.contact_thre)
    print("contact_threshold: ", contact_threshold)
    eval_save_path = os.path.join(os.path.dirname(motion_path), os.path.basename(motion_path) + '_evalres')
    if not os.path.exists(eval_save_path): os.makedirs(eval_save_path)
    log_file = os.path.join(eval_save_path, f'{best_metric}_cctThre={contact_threshold}.txt')
    with open(log_file, 'w') as f:
        f.write(f"motion_path: {motion_path}\n")
        f.write(f"best_metric: {best_metric}\n")

    grouped_files = fetch_files(motion_path)

    mpvpe_list = []
    mpmpe_list = []
    T_err_list = []
    O_err_list = []
    contact_precision_list = []
    contact_recall_list = []
    contact_acc_list = []
    contact_f1_score_list = []
    gt_contact_dist_list = []
    contact_dist_list = []

    num_motions = len(grouped_files)
    print(f"Found {num_motions} motions")
    for i in tqdm(range(num_motions)):
        num_samples_per_seq = len(grouped_files[i])
        assert num_samples_per_seq > 0
        if num_samples_per_seq != 5:
            print(f"n_samples is not 5: {num_samples_per_seq}")

        mpvpe_per_seq = []
        mpmpe_per_seq = []
        T_err_per_seq = []
        O_err_per_seq = []
        contact_precision_per_seq = []
        contact_recall_per_seq = []
        contact_acc_per_seq = []
        contact_f1_score_per_seq = []
        gt_contact_dist_per_seq = []
        contact_dist_per_seq = []

        for j in range(num_samples_per_seq):
            dict = np.load(os.path.join(motion_path, grouped_files[i][j]), allow_pickle=True)
            precomped_path = os.path.join(eval_save_path, grouped_files[i][j].replace('.npz', '_precomped.npz'))
            precomped_path = precomped_path.replace('.npz', f'_contact{contact_threshold}.npz')
            if not os.path.exists(precomped_path):
                marker = dict['marker']
                pred_verts = dict['pred_verts']
                gt_verts = dict['gt_verts']
                lhand_jpe, rhand_jpe, mpmpe, mpvpe, gt_contact_dist, contact_dist, \
                                gt_foot_sliding_jnts, foot_sliding_jnts, contact_precision, contact_recall, \
                                contact_acc, contact_f1_score = compute_metrics_obj_marker(pred_verts, gt_verts, marker)
                np.savez(precomped_path, mpmpe=mpmpe, mpvpe=mpvpe, gt_contact_dist=gt_contact_dist, contact_dist=contact_dist,
                    gt_foot_sliding_jnts=gt_foot_sliding_jnts, foot_sliding_jnts=foot_sliding_jnts,
                    contact_precision=contact_precision, contact_recall=contact_recall, contact_acc=contact_acc, contact_f1_score=contact_f1_score)
            else:
                precomped_dict = np.load(precomped_path, allow_pickle=True)
                mpmpe, mpvpe, gt_contact_dist, contact_dist, \
                    gt_foot_sliding_jnts, foot_sliding_jnts, \
                    contact_precision, contact_recall, contact_acc, contact_f1_score = precomped_dict['mpmpe'], precomped_dict['mpvpe'], precomped_dict['gt_contact_dist'], precomped_dict['contact_dist'], \
                                                                                        precomped_dict['gt_foot_sliding_jnts'], precomped_dict['foot_sliding_jnts'], \
                                                                                        precomped_dict['contact_precision'], precomped_dict['contact_recall'], precomped_dict['contact_acc'], precomped_dict['contact_f1_score']

            pred_trans = dict['pred_obj_trans']
            pred_angle = dict['pred_obj_angle']
            pred_angle_matrix = Rotation.from_rotvec(pred_angle).as_matrix()
            T = pred_trans.shape[0]

            gt_trans = dict['gt_obj_trans'][:T]
            gt_angle = dict['gt_obj_angle'][:T]
            gt_angle_matrix = Rotation.from_rotvec(gt_angle).as_matrix()

            T_err = np.linalg.norm(pred_trans - gt_trans, axis=-1).mean()
            O_err = rot_dist = get_frobenious_norm_rot_only(pred_angle_matrix.reshape(-1, 3, 3), gt_angle_matrix.reshape(-1, 3, 3))

            contact_precision_per_seq.append(contact_precision)
            contact_recall_per_seq.append(contact_recall)
            contact_acc_per_seq.append(contact_acc)
            contact_f1_score_per_seq.append(contact_f1_score)
            gt_contact_dist_per_seq.append(gt_contact_dist)
            contact_dist_per_seq.append(contact_dist)
            mpvpe_per_seq.append(mpvpe)
            mpmpe_per_seq.append(mpmpe)
            T_err_per_seq.append(T_err)
            O_err_per_seq.append(O_err)

        contact_precision_per_seq = np.asarray(contact_precision_per_seq)
        contact_recall_per_seq = np.asarray(contact_recall_per_seq)
        contact_acc_per_seq = np.asarray(contact_acc_per_seq)
        contact_f1_score_per_seq = np.asarray(contact_f1_score_per_seq)
        gt_contact_dist_per_seq = np.asarray(gt_contact_dist_per_seq)
        contact_dist_per_seq = np.asarray(contact_dist_per_seq)
        mpvpe_per_seq = np.asarray(mpvpe_per_seq)
        mpmpe_per_seq = np.asarray(mpmpe_per_seq)
        T_err_per_seq = np.asarray(T_err_per_seq)
        O_err_per_seq = np.asarray(O_err_per_seq)

        if best_metric == 'mpvpe':
            best_sample_idx = mpvpe_per_seq.argmin(axis=0)
        elif best_metric == 'mpmpe':
            best_sample_idx = mpmpe_per_seq.argmin(axis=0)
        elif best_metric == 'contact_f1_score':
            best_sample_idx = contact_f1_score_per_seq.argmax(axis=0)

        contact_precision_seq = contact_precision_per_seq[best_sample_idx]
        contact_recall_seq = contact_recall_per_seq[best_sample_idx]
        contact_acc_seq = contact_acc_per_seq[best_sample_idx]
        contact_f1_score_seq = contact_f1_score_per_seq[best_sample_idx]
        gt_contact_dist_seq = gt_contact_dist_per_seq[best_sample_idx]
        contact_dist_seq = contact_dist_per_seq[best_sample_idx]
        mpvpe_seq = mpvpe_per_seq[best_sample_idx]
        mpmpe_seq = mpmpe_per_seq[best_sample_idx]
        T_err_seq = T_err_per_seq[best_sample_idx]
        O_err_seq = O_err_per_seq[best_sample_idx]

        contact_precision_list.append(contact_precision_seq)
        contact_recall_list.append(contact_recall_seq)
        contact_acc_list.append(contact_acc_seq)
        contact_f1_score_list.append(contact_f1_score_seq)
        gt_contact_dist_list.append(gt_contact_dist_seq)
        contact_dist_list.append(contact_dist_seq)
        mpvpe_list.append(mpvpe_seq)
        mpmpe_list.append(mpmpe_seq)
        T_err_list.append(T_err_seq)
        O_err_list.append(O_err_seq)

        if i % 100 == 0 or i == num_motions - 1:
            mean_contact_precision = np.asarray(contact_precision_list).mean()
            mean_contact_recall = np.asarray(contact_recall_list).mean()
            mean_contact_acc = np.asarray(contact_acc_list).mean()
            mean_contact_f1_score = np.asarray(contact_f1_score_list).mean()
            mean_gt_contact_dist = np.asarray(gt_contact_dist_list).mean()
            mean_contact_dist = np.asarray(contact_dist_list).mean()
            mean_mpvpe = np.asarray(mpvpe_list).mean()
            mean_mpmpe = np.asarray(mpmpe_list).mean()
            mean_T_err = np.asarray(T_err_list).mean()
            mean_O_err = np.asarray(O_err_list).mean()

            print("*"*40 + "Quantitative Evaluation" + "*"*40)
            print("The number of sequences: {0}".format(len(mpvpe_list)))
            print("MPVPE: {0}".format(mean_mpvpe))
            print("MPMPE: {0}".format(mean_mpmpe))
            print("T_err: {0}, O_err: {1}".format(mean_T_err, mean_O_err))
            print("Contact precision: {0}, Contact recall: {1}".format(mean_contact_precision, mean_contact_recall))
            print("Contact Acc: {0}, Contact F1 score: {1}".format(mean_contact_acc, mean_contact_f1_score))
            print("Contact dist: {0}, GT Contact dist: {1}".format(mean_contact_dist, mean_gt_contact_dist))

            with open(log_file, "a") as f:
                f.write("*"*40 + "Quantitative Evaluation" + "*"*40 + "\n")
                f.write("The number of sequences: {0}\n".format(len(mpvpe_list)))
                f.write("MPVPE: {0}\n".format(mean_mpvpe))
                f.write("MPMPE: {0}\n".format(mean_mpmpe))
                f.write("T_err: {0}, O_err: {1}\n".format(mean_T_err, mean_O_err))
                f.write("Contact precision: {0}, Contact recall: {1}\n".format(mean_contact_precision, mean_contact_recall))
                f.write("Contact Acc: {0}, Contact F1 score: {1}\n".format(mean_contact_acc, mean_contact_f1_score))
                f.write("Contact dist: {0}, GT Contact dist: {1}\n".format(mean_contact_dist, mean_gt_contact_dist))
