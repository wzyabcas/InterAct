import numpy as np
import torch 

def compute_foot_sliding_for_marker(pred):
    # pred, gt: T X 77 X 3 
    T = pred.shape[0]

    # Put human mesh to floor z = 0 and compute.
    floor_height = np.min(pred[..., 1])
    pred[..., 1] -= floor_height

    foot_thre_dict = {
        0.04: [32, 25, 20, 21, 16, 54, 55, 59, 64, 50, 55] # left toe + right toe
    }
    foot_sliding_sum, joint_num = 0, 0
    for thresh, foot_ids in foot_thre_dict.items():
        foot_joints = pred[:, foot_ids, :]
        foot_disp = np.linalg.norm(foot_joints[1:, :, [0,2]] - foot_joints[:-1, :, [0,2]], axis=-1) # T x J
        foot_stick_labels = (foot_joints[:-1, :, 1] < thresh).astype(np.float32) # T x J
        foot_sliding_stats = np.abs(foot_disp * (2 - 2 ** (foot_joints[:-1, :, 1]/thresh))) * foot_stick_labels # T x J
        foot_sliding_sum += np.sum(foot_sliding_stats)
        joint_num += len(foot_ids)
    
    sliding = foot_sliding_sum / (T * joint_num) * 1000
    return sliding 

from scipy.spatial import KDTree

def get_min_dist(A, B):
    T = A.shape[0]
    min_distances = []
    for t in range(T):
        tree = KDTree(B[t])
        distances, indices = tree.query(A[t])
        min_distances.append(np.min(distances))
    return np.array(min_distances)

def compute_body_contact_metrics(marker_idx, pred, gt, obj_verts, precomp_gt_contact=None, contact_threh=0.1):
    pred_mk = pred[:, marker_idx, :]
    gt_mk = gt[:, marker_idx, :]
    jpe = np.linalg.norm(pred_mk - gt_mk, axis=-1).mean() * 100

    N, M = len(marker_idx), obj_verts.shape[1]
    
    use_KDTree = True
    if use_KDTree:
        pred_mk2obj_dist_min = get_min_dist(pred_mk, obj_verts)
        pred_contact = (pred_mk2obj_dist_min < contact_threh)
        if precomp_gt_contact is None:
            gt_mk2obj_dist_min = get_min_dist(gt_mk, obj_verts)
            gt_contact = (gt_mk2obj_dist_min < contact_threh)
        else:
            gt_mk2obj_dist_min = precomp_gt_contact['dist']
            gt_contact = precomp_gt_contact['contact']
    else:
        pred_mk2obj_dist = np.sqrt(((np.tile(pred_mk[..., None, :], (1, 1, M, 1)) - obj_verts[:,None,...])**2).sum(axis=-1)) # T X N x M
        pred_mk2obj_dist = pred_mk2obj_dist.min(axis=-1) # T x N
        pred_mk2obj_dist_min = pred_mk2obj_dist.min(axis=1) # T
        pred_contact = (pred_mk2obj_dist_min < contact_threh)

        if precomp_gt_contact is None:
            gt_mk2obj_dist = np.sqrt(((np.tile(gt_mk[..., None, :], (1, 1, M, 1)) - obj_verts[:,None,...])**2).sum(axis=-1)) # T X N x M
            gt_mk2obj_dist = gt_mk2obj_dist.min(axis=-1) # T x N
            gt_mk2obj_dist_min = gt_mk2obj_dist.min(axis=1) # T
            gt_contact = (gt_mk2obj_dist_min < contact_threh)
        else:
            gt_mk2obj_dist_min = precomp_gt_contact['dist']
            gt_contact = precomp_gt_contact['contact']

    T = pred.shape[0]

    contact_dist = 0
    gt_contact_dist = 0 
    gt_contact_cnt = 0
    for idx in range(T):
        if gt_contact[idx]:
            gt_contact_cnt += 1 
            contact_dist += pred_mk2obj_dist_min[idx]
            gt_contact_dist += gt_mk2obj_dist_min[idx]

    if gt_contact_cnt == 0:
        contact_dist = 0 
        gt_contact_dist = 0 
    else:
        contact_dist = contact_dist/float(gt_contact_cnt)
        gt_contact_dist = gt_contact_dist/float(gt_contact_cnt)

    TP, FP, TN, FN = 0, 0, 0, 0
    for idx in range(T):
        gt_in_contact = gt_contact[idx]
        pred_in_contact = pred_contact[idx]
        if gt_in_contact and pred_in_contact: TP += 1
        if (not gt_in_contact) and pred_in_contact: FP += 1
        if (not gt_in_contact) and (not pred_in_contact): TN += 1
        if gt_in_contact and (not pred_in_contact): FN += 1
    
    contact_acc = (TP+TN)/(TP+FP+TN+FN)

    if (TP+FP) == 0: # Prediction no contact!!!
        contact_precision = 0
    else:
        contact_precision = TP/(TP+FP)
    
    if (TP+FN) == 0: # GT no contact! 
        contact_recall = 0
    else:
        contact_recall = TP/(TP+FN)

    if contact_precision == 0 and contact_recall == 0:
        contact_f1_score = 0 
    else:
        contact_f1_score = 2 * (contact_precision * contact_recall)/(contact_precision+contact_recall) 
   
    contact_dict = {
        "jpe": jpe,
        "contact_precision": contact_precision,
        "contact_recall": contact_recall,
        "contact_acc": contact_acc,
        "contact_f1_score": contact_f1_score,
        "contact_dist": contact_dist,
        "gt_contact_dist": gt_contact_dist   
    }
    return contact_dict, {"contact": gt_contact, "dist": gt_mk2obj_dist_min}

def compute_metrics_marker(pred, gt, obj_verts, precomp_gt_contact=None, contact_threh=0.1):
    # pred: T X 77 X 3
    # gt: T X 77 X 3
    # obj_verts: T X M X 3

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    if isinstance(obj_verts, torch.Tensor):
        obj_verts = obj_verts.detach().cpu().numpy()

    T = pred.shape[0]
    lhand_idx, rhand_idx = [72, 73, 74, 75, 76], [67, 68, 69, 70, 71]
    hand_idx = lhand_idx + rhand_idx
    lhand_pred, rhand_pred = pred[:, lhand_idx, :], pred[:, rhand_idx, :] # T x 5 x 3
    lhand_gt, rhand_gt = gt[:, lhand_idx, :], gt[:, rhand_idx, :] # T x 5 x 3
    lhand_jpe = np.linalg.norm(lhand_pred - lhand_gt, axis=-1).mean() * 100
    rhand_jpe = np.linalg.norm(rhand_pred - rhand_gt, axis=-1).mean() * 100
    hand_jpe = (lhand_jpe+rhand_jpe)/2.0

    # Calculate MPVPE
    root_idx = [0]
    pred_local, gt_local = pred - pred[:, root_idx], gt - gt[:, root_idx]
    mpvpe = np.linalg.norm(pred_local - gt_local, axis=-1).mean()

    # Calculate foot sliding
    foot_sliding_jnts = compute_foot_sliding_for_marker(pred)
    gt_foot_sliding_jnts = compute_foot_sliding_for_marker(gt)
    gt_contacts = {}

    # Calculate contact score
    gt_contacts["hand"] = None if precomp_gt_contact is None else precomp_gt_contact["hand"]
    hand_contact_dict, gt_ct = compute_body_contact_metrics(hand_idx, pred, gt, obj_verts, precomp_gt_contact=gt_contacts["hand"], contact_threh=contact_threh)
    gt_contacts["hand"] = gt_ct

    gt_contacts["non_hand"] = None if precomp_gt_contact is None else precomp_gt_contact["non_hand"]
    non_hand_contact_dict, gt_ct = compute_body_contact_metrics([i for i in range(77) if i not in hand_idx], pred, gt, obj_verts, precomp_gt_contact=gt_contacts["non_hand"], contact_threh=contact_threh)
    gt_contacts["non_hand"] = gt_ct

    foot_idx = [32, 25, 20, 21, 16] + [54, 55, 59, 64, 50, 55]
    gt_contacts["foot"] = None if precomp_gt_contact is None else precomp_gt_contact["foot"]
    foot_contact_dict, gt_ct = compute_body_contact_metrics(foot_idx, pred, gt, obj_verts, precomp_gt_contact=gt_contacts["foot"], contact_threh=contact_threh)
    gt_contacts["foot"] = gt_ct

    # full jpe = mpmpe
    gt_contacts["full"] = None if precomp_gt_contact is None else precomp_gt_contact["full"]
    full_contact_dict, gt_ct = compute_body_contact_metrics([i for i in range(77)], pred, gt, obj_verts, precomp_gt_contact=gt_contacts["full"], contact_threh=contact_threh)
    gt_contacts["full"] = gt_ct

    dicts = {
        "hand": hand_contact_dict,
        "non_hand": non_hand_contact_dict,
        "foot": foot_contact_dict,
        "full": full_contact_dict
    }

    return lhand_jpe, rhand_jpe, hand_jpe, mpvpe, \
            gt_foot_sliding_jnts, foot_sliding_jnts, dicts, gt_contacts