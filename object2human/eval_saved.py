import argparse
import os
import numpy as np
from tqdm import tqdm
from evaluation_metrics import compute_metrics_marker
import pytorch_lightning as pl
from copy import deepcopy
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
    parser.add_argument('--motion_path', default='???')
    parser.add_argument('--best_metric', default='full_jpe', choices=['mpvpe', 'full_jpe', 'full_contact_f1_score'])
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
    
    global_hand_jpe_list = [] 
    global_lhand_jpe_list = []
    global_rhand_jpe_list = [] 
    mpvpe_list = []
    mpjpe_list = []
    gt_foot_sliding_jnts_list = []
    foot_sliding_jnts_list = []
    dict_template = {
        "jpe": [],
        "contact_precision": [],
        "contact_recall": [],
        "contact_acc": [],
        "contact_f1_score": [],
        "contact_dist": [],
        "gt_contact_dist": []   
    }
    list_dict = {
        "hand": deepcopy(dict_template),
        "non_hand": deepcopy(dict_template),
        "foot": deepcopy(dict_template),
        "full": deepcopy(dict_template)
    }

    num_motions = len(grouped_files)
    print(f"Found {num_motions} motions")
    for i in tqdm(range(num_motions)):
        num_samples_per_seq = len(grouped_files[i])
        assert num_samples_per_seq > 0
        if num_samples_per_seq != 10:
            print(f"n_samples is not 10: {num_samples_per_seq}")

        hand_jpe_per_seq = []
        lhand_jpe_per_seq = []
        rhand_jpe_per_seq = []
        mpvpe_per_seq = []
        gt_foot_sliding_jnts_per_seq = []
        foot_sliding_jnts_per_seq = []
        per_seq_dict = {
            "hand": deepcopy(dict_template),
            "non_hand": deepcopy(dict_template),
            "foot": deepcopy(dict_template),
            "full": deepcopy(dict_template)
        }

        gt_contacts = None
        for j in range(num_samples_per_seq):
            dict = np.load(os.path.join(motion_path, grouped_files[i][j]), allow_pickle=True)
            pred_marker_j = dict['pred_marker']
            gt_marker_j = dict['gt_marker']
            obj_verts = dict['obj_verts']
            lhand_jpe, rhand_jpe, hand_jpe, mpvpe, \
                gt_foot_sliding_jnts, foot_sliding_jnts, dicts, gt_contacts = \
                    compute_metrics_marker(pred_marker_j, gt_marker_j, obj_verts, precomp_gt_contact=gt_contacts, contact_threh=contact_threshold)
            
            hand_jpe_per_seq.append(hand_jpe)
            lhand_jpe_per_seq.append(lhand_jpe)
            rhand_jpe_per_seq.append(rhand_jpe) 
            mpvpe_per_seq.append(mpvpe)
            gt_foot_sliding_jnts_per_seq.append(gt_foot_sliding_jnts)
            foot_sliding_jnts_per_seq.append(foot_sliding_jnts)
            for key, val in per_seq_dict.items():
                for k in val.keys():
                    val[k].append(dicts[key][k])
        
        hand_jpe_per_seq = np.asarray(hand_jpe_per_seq)
        lhand_jpe_per_seq = np.asarray(lhand_jpe_per_seq)
        rhand_jpe_per_seq = np.asarray(rhand_jpe_per_seq)
        mpvpe_per_seq = np.asarray(mpvpe_per_seq)
        gt_foot_sliding_jnts_per_seq = np.asarray(gt_foot_sliding_jnts_per_seq)
        foot_sliding_jnts_per_seq = np.asarray(foot_sliding_jnts_per_seq)
        for key, val in per_seq_dict.items():
            for k in val.keys():
                val[k] = np.asarray(val[k])

        if best_metric == 'mpvpe':
            best_sample_idx = mpvpe_per_seq.argmin(axis=0) # sample_num
        elif best_metric == 'full_jpe':
            best_sample_idx = per_seq_dict['full']['jpe'].argmin(axis=0)
        elif best_metric == 'full_contact_f1_score':
            best_sample_idx = per_seq_dict['full']['contact_f1_score'].argmax(axis=0)

        hand_jpe = hand_jpe_per_seq[best_sample_idx] # BS 
        lhand_jpe = lhand_jpe_per_seq[best_sample_idx]
        rhand_jpe = rhand_jpe_per_seq[best_sample_idx]
        mpvpe = mpvpe_per_seq[best_sample_idx]
        gt_foot_sliding_jnts = gt_foot_sliding_jnts_per_seq[best_sample_idx]
        foot_sliding_jnts = foot_sliding_jnts_per_seq[best_sample_idx]
        for key, val in per_seq_dict.items():
            for k in val.keys():
                val[k] = val[k][best_sample_idx]
        
        global_hand_jpe_list.append(hand_jpe)
        global_lhand_jpe_list.append(lhand_jpe)
        global_rhand_jpe_list.append(rhand_jpe)
        mpvpe_list.append(mpvpe)
        gt_foot_sliding_jnts_list.append(gt_foot_sliding_jnts)
        foot_sliding_jnts_list.append(foot_sliding_jnts)
        for key, val in list_dict.items():
            for k in val.keys():
                val[k].append(per_seq_dict[key][k])
        
        if i % 100 == 0 or i == num_motions - 1:
            mean_hand_jpe = np.asarray(global_hand_jpe_list).mean() 
            mean_lhand_jpe = np.asarray(global_lhand_jpe_list).mean()
            mean_rhand_jpe = np.asarray(global_rhand_jpe_list).mean()
            mean_mpvpe = np.asarray(mpvpe_list).mean()
            mean_gt_fsliding_jnts = np.asarray(gt_foot_sliding_jnts_list).mean()
            mean_fsliding_jnts = np.asarray(foot_sliding_jnts_list).mean()

            print("*****************************************Quantitative Evaluation*****************************************")
            print("The number of sequences: {0}".format(len(mpvpe_list)))
            print("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
            print("MPVPE: {0}".format(mean_mpvpe))
            print("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
            for key, val in list_dict.items():
                for k in val.keys():
                    mean_val = np.asarray(val[k]).mean()
                    print(f"{key} {k}: {mean_val}")
            
            with open(log_file, 'a') as f:
                f.write("*****************************************Quantitative Evaluation*****************************************\n")
                f.write("The number of sequences: {0}\n".format(len(mpvpe_list)))
                f.write("Left Hand JPE: {0}, Right Hand JPE: {1}, Two Hands JPE: {2}\n".format(mean_lhand_jpe, mean_rhand_jpe, mean_hand_jpe))
                f.write("MPVPE: {0}\n".format(mean_mpvpe))
                f.write("Foot sliding jnts: {0}, GT Foot sliding jnts: {1}\n".format(mean_fsliding_jnts, mean_gt_fsliding_jnts))
                for key, val in list_dict.items():
                    for k in val.keys():
                        mean_val = np.asarray(val[k]).mean()
                        f.write(f"{key} {k}: {mean_val}\n")