import numpy as np
import torch
from scipy.spatial import KDTree


def get_frobenious_norm_rot_only(x, y):
    # x, y: N X 3 X 3
    error = 0.0
    for i in range(len(x)):
        x_mat = x[i][:3, :3]
        y_mat_inv = np.linalg.inv(y[i][:3, :3])
        error_mat = np.matmul(x_mat, y_mat_inv)
        ident_mat = np.identity(3)
        error += np.linalg.norm(ident_mat - error_mat, 'fro')
    return error / len(x)


def get_min_dist(A, B):
    T = A.shape[0]
    min_distances = []
    for t in range(T):
        tree = KDTree(B[t])
        distances, indices = tree.query(A[t])
        min_distances.append(np.min(distances))
    return np.array(min_distances)


def compute_metrics_obj_marker(pred, gt, obj_verts):
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

    # Calculate MPVPE
    root_idx = [0]
    pred_local, gt_local = pred - pred[:, root_idx], gt - gt[:, root_idx]
    mpvpe = np.linalg.norm(pred_local - gt_local, axis=-1).mean()

    mpmpe = np.linalg.norm(pred - gt, axis=-1).mean()

    # Calculate contact score
    contact_threh = 0.05

    gt_lhand2obj_dist_min = get_min_dist(obj_verts, gt)
    gt_lhand_contact = (gt_lhand2obj_dist_min < contact_threh)

    lhand2obj_dist_min = get_min_dist(obj_verts, pred)
    lhand_contact = (lhand2obj_dist_min < contact_threh)

    # Compute the distance between hand joint and object for frames that are in contact with object in GT.
    contact_dist = 0
    gt_contact_dist = 0
    gt_contact_cnt = 0
    for idx in range(T):
        if gt_lhand_contact[idx]:
            gt_contact_cnt += 1
            contact_dist += min(lhand2obj_dist_min[idx], lhand2obj_dist_min[idx])
            gt_contact_dist += min(gt_lhand2obj_dist_min[idx], gt_lhand2obj_dist_min[idx])

    if gt_contact_cnt == 0:
        contact_dist = 0
        gt_contact_dist = 0
    else:
        contact_dist = contact_dist/float(gt_contact_cnt)
        gt_contact_dist = gt_contact_dist/float(gt_contact_cnt)

    # Compute precision and recall for contact.
    TP, FP, TN, FN = 0, 0, 0, 0
    for idx in range(T):
        gt_in_contact = (gt_lhand_contact[idx])
        pred_in_contact = (lhand_contact[idx])
        if gt_in_contact and pred_in_contact: TP += 1
        if (not gt_in_contact) and pred_in_contact: FP += 1
        if (not gt_in_contact) and (not pred_in_contact): TN += 1
        if gt_in_contact and (not pred_in_contact): FN += 1

    contact_acc = (TP+TN)/(TP+FP+TN+FN)

    if (TP+FP) == 0:
        contact_precision = 0
    else:
        contact_precision = TP/(TP+FP)

    if (TP+FN) == 0:
        contact_recall = 0
    else:
        contact_recall = TP/(TP+FN)

    if contact_precision == 0 and contact_recall == 0:
        contact_f1_score = 0
    else:
        contact_f1_score = 2 * (contact_precision * contact_recall)/(contact_precision+contact_recall)

    return 0, 0, mpmpe, mpvpe, gt_contact_dist, contact_dist, \
            0, 0, contact_precision, contact_recall, contact_acc, contact_f1_score
