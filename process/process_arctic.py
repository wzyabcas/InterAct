"""
Combined script to process ARCTIC dataset:
1. Convert raw sequences to InterAct format
2. Split sequences into chunks
3. Scale object files from mm to meters
"""

import os
import os.path as op
import numpy as np
import torch
import json
import re
import spacy
import trimesh
import shutil
from glob import glob
from scipy.spatial.transform import Rotation as R
import smplx
from object_tensors import ObjectTensors
from preprocess_dataset import construct_loader
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu()

MODEL_PATH = './models'
DESC_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s+([^,]+),\s*(.+?)\s*$")
SMPLX_MODEL_P = {
    "male": "./models/smplx/SMPLX_MALE.npz",
    "female": "./models/smplx/SMPLX_FEMALE.npz",
    "neutral": "./models/smplx/SMPLX_NEUTRAL.npz",
}
_NLP: Optional[object] = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


@dataclass(frozen=True)
class Annotation:
    start: int
    end: int
    motion: str
    hand: str


def parse_description(path: Path) -> List[Annotation]:
    annotations: List[Annotation] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = DESC_PATTERN.match(line)
        if not match:
            raise ValueError(f"Invalid annotation line in {path}: {raw_line!r}")
        start, end = int(match.group(1)), int(match.group(2))
        motion = match.group(3).strip()
        hand = match.group(4).strip()
        if end < start:
            raise ValueError(f"End < start in {path}: {raw_line!r}")
        annotations.append(Annotation(start=start, end=end, motion=motion, hand=hand))
    if not annotations:
        raise ValueError(f"No annotations found in {path}")
    return annotations


def merge_intervals(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def choose_best_cut(
    start_frame: int,
    candidate_cuts: Sequence[int],
    min_len: int,
    max_len: int,
    target_len: int,
) -> int:
    valid: List[int] = []
    for cut in candidate_cuts:
        seg_len = cut - start_frame + 1
        if min_len <= seg_len <= max_len:
            valid.append(cut)
    if not valid:
        return -1
    return min(valid, key=lambda c: (abs((c - start_frame + 1) - target_len), c))


def compute_chunk_ranges(
    frame_count: int,
    annotations: Sequence[Annotation],
    min_len: int,
    max_len: int,
    target_len: int,
) -> List[Tuple[int, int]]:
    if frame_count < min_len:
        return [(0, frame_count - 1)]

    intervals = [(a.start, a.end) for a in annotations]
    clusters = merge_intervals(intervals)
    preferred_cuts = sorted(
        {
            cluster_start - 1
            for cluster_start, _ in clusters[1:]
            if 0 <= cluster_start - 1 < frame_count - 1
        }
    )
    safe_cuts = sorted(
        {
            a.end
            for a in annotations
            if 0 <= a.end < frame_count - 1
        }
        | set(preferred_cuts)
        | {frame_count - 1}
    )

    ranges: List[Tuple[int, int]] = []
    start = 0
    idx = 0
    while start < frame_count:
        if frame_count - start <= max_len:
            ranges.append((start, frame_count - 1))
            break

        while idx < len(preferred_cuts) and preferred_cuts[idx] < start:
            idx += 1

        local_preferred: List[int] = []
        scan = idx
        while scan < len(preferred_cuts):
            cut = preferred_cuts[scan]
            seg_len = cut - start + 1
            if seg_len > max_len:
                break
            if seg_len >= min_len:
                local_preferred.append(cut)
            scan += 1

        cut = choose_best_cut(start, local_preferred, min_len, max_len, target_len)
        if cut < 0:
            local_safe = [c for c in safe_cuts if start + min_len - 1 <= c <= start + max_len - 1]
            cut = choose_best_cut(start, local_safe, min_len, max_len, target_len)
            if cut < 0:
                cut = min(start + max_len - 1, frame_count - 1)

        ranges.append((start, cut))
        start = cut + 1

    if len(ranges) >= 2 and (ranges[-1][1] - ranges[-1][0] + 1) < min_len:
        prev_start, _ = ranges[-2]
        tail_end = ranges[-1][1]
        candidates = [
            c
            for c in safe_cuts
            if prev_start + min_len - 1 <= c <= prev_start + max_len - 1
            and min_len <= (tail_end - (c + 1) + 1) <= max_len
        ]
        if candidates:
            best = min(candidates, key=lambda c: abs((c - prev_start + 1) - target_len))
            ranges[-2] = (prev_start, best)
            ranges[-1] = (best + 1, tail_end)

    out: List[Tuple[int, int]] = []
    for start, end in ranges:
        if not out:
            out.append((start, end))
            continue
        length = end - start + 1
        if length < min_len:
            prev_start, _ = out[-1]
            if (end - prev_start + 1) <= max_len:
                out[-1] = (prev_start, end)
            else:
                out.append((start, end))
        else:
            out.append((start, end))
    return out


def slice_npz(data: Dict[str, np.ndarray], start: int, end: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] >= end + 1:
            out[key] = value[start : end + 1]
        else:
            out[key] = value
    return out


def build_text_lines(
    annotations: Sequence[Annotation],
    seq_name: str,
    chunk_start: int,
    chunk_end: int,
) -> List[str]:
    object_name = seq_name.split("_", 2)[1] if "_" in seq_name else "object"
    selected: List[Annotation] = []
    for ann in annotations:
        if ann.end < chunk_start or ann.start > chunk_end:
            continue
        selected.append(ann)

    both_by_motion: Dict[str, List[Tuple[int, int]]] = {}
    for ann in selected:
        if ann.hand.strip().lower() == "both hands":
            both_by_motion.setdefault(ann.motion.lower(), []).append((ann.start, ann.end))

    filtered: List[Annotation] = []
    for ann in selected:
        hand = ann.hand.strip().lower()
        if hand == "both hands":
            filtered.append(ann)
            continue

        suppress = False
        for b_start, b_end in both_by_motion.get(ann.motion.lower(), []):
            overlaps = not (ann.end < b_start or ann.start > b_end)
            if overlaps:
                suppress = True
                break
        if not suppress:
            filtered.append(ann)

    lines: List[str] = []
    for ann in filtered:
        lines.append(f"{ann.motion} {object_name} with {ann.hand}")
    return lines


def build_sentence_for_chunk(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    if len(lines) == 1:
        return f"{lines[0]}."
    return f"{', '.join(lines[:-1])}, and {lines[-1]}."


def build_token_pos_string(sentence: str) -> str:
    sentence = sentence.replace("-", "")
    doc = _get_nlp()(sentence)
    tokens: List[str] = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if token.pos_ in {"NOUN", "VERB"} and word.lower() != "left":
            out = token.lemma_
        else:
            out = word
        tokens.append(f"{out.lower()}/{token.pos_}")
    return " ".join(tokens)


def sequence_to_description_path(description_root: Path, seq_name: str) -> Path:
    parts = seq_name.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected sequence directory name: {seq_name}")
    subject, action_name = parts
    return description_root / subject / action_name / "description.txt"


def split_sequence_outputs(
    sequence_dir: Path,
    description_root: Path,
    split_output_root: Path,
    min_len: int = 200,
    max_len: int = 400,
    target_len: int = 300,
) -> None:
    seq_name = sequence_dir.name
    desc_path = sequence_to_description_path(description_root, seq_name)
    if not desc_path.exists():
        print(f"[WARN] Missing description: {seq_name} -> {desc_path}")
        return

    human_path = sequence_dir / "human.npz"
    object_path = sequence_dir / "object.npz"
    if not human_path.exists() or not object_path.exists():
        print(f"[WARN] Missing npz files in: {sequence_dir}")
        return

    annotations = parse_description(desc_path)
    with np.load(human_path, allow_pickle=True) as human_npz, np.load(object_path, allow_pickle=True) as object_npz:
        human_data = {k: human_npz[k] for k in human_npz.files}
        object_data = {k: object_npz[k] for k in object_npz.files}

    if "poses" not in human_data:
        raise ValueError(f"'poses' key missing in {human_path}")
    frame_count = int(human_data["poses"].shape[0])
    if frame_count <= 0:
        raise ValueError(f"Invalid frame count in {human_path}: {frame_count}")

    ranges = compute_chunk_ranges(
        frame_count=frame_count,
        annotations=annotations,
        min_len=min_len,
        max_len=max_len,
        target_len=target_len,
    )
    for start, end in ranges:
        sub_name = f"{seq_name}_{start}"
        out_dir = split_output_root / sub_name
        out_dir.mkdir(parents=True, exist_ok=True)

        human_chunk = slice_npz(human_data, start, end)
        object_chunk = slice_npz(object_data, start, end)
        np.savez(out_dir / "human.npz", **human_chunk)
        np.savez(out_dir / "object.npz", **object_chunk)
        lines = build_text_lines(annotations, seq_name, start, end)
        sentence = build_sentence_for_chunk(lines)
        pos_tokens = build_token_pos_string(sentence)
        text_row = f"{sentence}#{pos_tokens}#0.0#0.0"
        (out_dir / "text.txt").write_text(text_row, encoding="utf-8")

    split_str = ", ".join([f"{s}-{e}" for s, e in ranges])

# ============ Part 1: arctic_to_inter utilities ============
def build_smplx(batch_size, gender, vtemplate):
    subj_m = smplx.create(
        model_path=SMPLX_MODEL_P[gender],
        model_type="smplx",
        gender=gender,
        num_pca_comps=45,
        v_template=vtemplate,
        flat_hand_mean=True,
        use_pca=False,
        batch_size=batch_size,
    )
    return subj_m

def construct_layers(dev, subject_id):
    with open("./data/arctic/raw/meta/misc.json", "r") as f:
        misc = json.load(f)
    vtemplate_p = f"./data/arctic/raw/meta/subject_vtemplates/{subject_id}.obj"
    mesh = trimesh.load_mesh(vtemplate_p)
    vtemplate = mesh.vertices
    gender = misc[subject_id]["gender"]
    mano_layers = {
        "smplx": build_smplx(1, gender, vtemplate),
    }
    for layer in mano_layers.values():
        layer.to(dev)
    return mano_layers

def thing2dev(thing, dev):
    if hasattr(thing, "to"):
        thing = thing.to(dev)
        return thing
    if isinstance(thing, list):
        return [thing2dev(ten, dev) for ten in thing]
    if isinstance(thing, tuple):
        return tuple(thing2dev(list(thing), dev))
    if isinstance(thing, dict):
        return {k: thing2dev(v, dev) for k, v in thing.items()}
    if isinstance(thing, torch.Tensor):
        return thing.to(dev)
    return thing

def _resolve_gender(subject_id):
    with open("./data/arctic/raw/meta/misc.json", "r") as f:
        misc = json.load(f)
    gender = misc[subject_id]["gender"]
    return gender

def _pack_poses_axis_angle(batch):
    parts = [
        batch["smplx_global_orient"],
        batch["smplx_body_pose"],
        batch["smplx_left_hand_pose"],
        batch["smplx_right_hand_pose"],
        batch["smplx_jaw_pose"],
        batch["smplx_leye_pose"],
        batch["smplx_reye_pose"],
    ]
    parts = [p.reshape(p.shape[0], -1) for p in parts]
    poses = torch.cat(parts, dim=1)
    return poses

def _concat(list_of_arrays, axis=0):
    if not list_of_arrays:
        return None
    return np.concatenate(list_of_arrays, axis=axis)


def process_seq_params_direct(mano_p, dev, statcams, layers):
    with torch.no_grad():
        sid, seqname = mano_p.split("/")[-2:]
        seqname = seqname.replace(".mano.npy", "")
        out_root = f"./data/arctic/sequences/{sid}/{seqname}"
        os.makedirs(out_root, exist_ok=True)

        loader = construct_loader(mano_p)

        curr_bs = None
        smplx_m = None
        gender_str = "neutral"
        betas_subject = None

        poses_seq = []
        transl_seq = []
        obj_angles_seq = []
        obj_transl_seq = []
        obj_arti_seq = []
        qname_seq = []
        rotation_x = R.from_euler("x", -np.pi / 2.0, degrees=False)

        for batch in loader:
            batch = thing2dev(batch, dev)
            B = batch["smplx_transl"].shape[0]

            if B != curr_bs:
                curr_bs = B
                smplx_m = layers["smplx"]
                gender_str = _resolve_gender(sid)

                betas_tensor = getattr(smplx_m, "betas", None)
                if betas_tensor is None:
                    num_betas = getattr(smplx_m, "num_betas", 16)
                    betas_subject = np.zeros((num_betas,), dtype=np.float32)
                else:
                    bt = betas_tensor.detach().cpu().numpy()
                    betas_subject = (bt[0] if bt.ndim == 2 else bt).astype(np.float32)

            poses = _pack_poses_axis_angle(batch).detach().cpu().numpy()
            transl = batch["smplx_transl"].detach().cpu().numpy()

            betas_model = getattr(smplx_m, "betas", None)
            if betas_model is None:
                num_betas = int(getattr(smplx_m, "num_betas", 10))
                betas_batch = torch.zeros((B, num_betas), device=dev, dtype=batch["smplx_transl"].dtype)
            else:
                if betas_model.ndim == 1:
                    betas_batch = betas_model[None, :].repeat(B, 1)
                elif betas_model.shape[0] == B:
                    betas_batch = betas_model
                else:
                    betas_batch = betas_model[:1].repeat(B, 1)
            expression_batch = torch.zeros((B, 10), device=dev, dtype=batch["smplx_transl"].dtype)

            smplx_output = smplx_m(
                body_pose=batch["smplx_body_pose"],
                global_orient=batch["smplx_global_orient"],
                left_hand_pose=batch["smplx_left_hand_pose"],
                right_hand_pose=batch["smplx_right_hand_pose"],
                jaw_pose=batch["smplx_jaw_pose"],
                leye_pose=batch["smplx_leye_pose"],
                reye_pose=batch["smplx_reye_pose"],
                transl=batch["smplx_transl"],
                betas=betas_batch,
                expression=expression_batch,
            )
            pelvis_old = smplx_output.joints[:, 0, :].detach().cpu().numpy()

            # Match GRAB preprocessing: rotate all global motions by -90deg around X.
            human_root = R.from_rotvec(poses[:, :3])
            rotated_global_orient = (rotation_x * human_root).as_rotvec().astype(np.float32)
            poses[:, :3] = rotated_global_orient
            transl = rotation_x.apply(transl).astype(np.float32)

            smplx_output_rot = smplx_m(
                body_pose=batch["smplx_body_pose"],
                global_orient=torch.from_numpy(rotated_global_orient).to(
                    device=dev, dtype=batch["smplx_global_orient"].dtype
                ),
                left_hand_pose=batch["smplx_left_hand_pose"],
                right_hand_pose=batch["smplx_right_hand_pose"],
                jaw_pose=batch["smplx_jaw_pose"],
                leye_pose=batch["smplx_leye_pose"],
                reye_pose=batch["smplx_reye_pose"],
                transl=torch.from_numpy(transl).to(device=dev, dtype=batch["smplx_transl"].dtype),
                betas=betas_batch,
                expression=expression_batch,
            )
            pelvis_new = smplx_output_rot.joints[:, 0, :].detach().cpu().numpy()

            poses_seq.append(poses)
            transl_seq.append(transl)

            obj_arti = batch["obj_arti"].view(-1, 1).detach().cpu().numpy()
            obj_grot = batch["obj_rot"].detach().cpu().numpy()
            obj_trans_m = (batch["obj_trans"] / 1000.0).detach().cpu().numpy()

            obj_rots = R.from_rotvec(obj_grot)
            obj_grot = (rotation_x * obj_rots).as_rotvec()
            obj_trans_delta = rotation_x.apply(obj_trans_m - pelvis_old)
            obj_trans_m = pelvis_new + obj_trans_delta

            obj_angles_seq.append(obj_grot)
            obj_transl_seq.append(obj_trans_m)
            obj_arti_seq.append(obj_arti)

            qnames = batch["query_names"]
            if isinstance(qnames, (list, tuple)):
                qname_seq.extend([str(x) for x in qnames])
            else:
                qname_seq.append(str(qnames))

        poses_all = _concat(poses_seq, axis=0)
        trans_all = _concat(transl_seq, axis=0)
        angles_all = _concat(obj_angles_seq, axis=0)
        otrans_all = _concat(obj_transl_seq, axis=0)
        obj_arti_all = _concat(obj_arti_seq, axis=0)
        obj_name = qname_seq[0] if len(qname_seq) > 0 else ""

        human_path = op.join(out_root, "human.npz")
        np.savez(human_path,
                 poses=poses_all,
                 trans=trans_all,
                 betas=betas_subject,
                 gender=gender_str)

        object_path = op.join(out_root, "object.npz")
        np.savez(object_path,
                 angles=angles_all,
                 trans=otrans_all,
                 arti=obj_arti_all,
                 name=np.array(obj_name, dtype=object))

        return out_root, sid, seqname

# ============ Part 2: transform_arc utilities ============
def build_smpl_model(gender, subject_id):
    vtemplate_p = f"./data/arctic/raw/meta/subject_vtemplates/{subject_id}.obj"
    mesh = trimesh.load_mesh(vtemplate_p)
    vtemplate = mesh.vertices
    smpl_model = smplx.create(
        model_path=MODEL_PATH,
        model_type="smplx",
        gender=gender,
        num_pca_comps=45,
        v_template=vtemplate,
        flat_hand_mean=True,
        use_pca=False,
    )
    return smpl_model

def visualize_smpl_arctic(poses, betas, trans, gender, subject_id):
    frame_times = poses.shape[0]
    smpl_model = build_smpl_model(gender, subject_id)

    smplx_output = smpl_model(body_pose=torch.from_numpy(poses[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses[:, :3]).float(),
        left_hand_pose=torch.from_numpy(poses[:, 66:111]).float(),
        right_hand_pose=torch.from_numpy(poses[:, 111:156]).float(),
        jaw_pose=torch.from_numpy(poses[:, 156:159]).float(),
        leye_pose=torch.from_numpy(poses[:, 159:162]).float(),
        reye_pose=torch.from_numpy(poses[:, 162:165]).float(),
        expression=torch.zeros([frame_times,10]).float(),
        betas=torch.from_numpy(betas[None, :]).repeat(frame_times, 1).float(),
        transl=torch.from_numpy(trans).float(),)
    verts = to_cpu(smplx_output.vertices)
    joints= to_cpu(smplx_output.joints)
    faces = smpl_model.faces.astype(np.int32)
    
    return verts, joints, faces, poses

def _normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def estimate_forward_from_joints(joints_frame0):
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # Primary estimate from collar directions around the spine joint:
    # - (3 -> 14) points to body right, so facing is +90 deg around +Y.
    # - (3 -> 13) points to body left, so facing is -90 deg around +Y.
    spine = joints_frame0[3]
    right_collar_vec = joints_frame0[14] - spine
    left_collar_vec = joints_frame0[13] - spine
    right_collar_vec[1] = 0.0
    left_collar_vec[1] = 0.0

    forward_candidates = []
    if np.linalg.norm(right_collar_vec) >= 1e-6:
        right_collar_vec = _normalize(right_collar_vec).reshape(3,)
        forward_candidates.append(np.cross(right_collar_vec, up))
    if np.linalg.norm(left_collar_vec) >= 1e-6:
        left_collar_vec = _normalize(left_collar_vec).reshape(3,)
        forward_candidates.append(np.cross(up, left_collar_vec))

    if len(forward_candidates) > 0:
        forward = np.sum(np.asarray(forward_candidates), axis=0)
    else:
        # Fallback to shoulder/hip based estimate when collar vectors are degenerate.
        right_vec = joints_frame0[17] - joints_frame0[16]
        if np.linalg.norm(right_vec) < 1e-6:
            right_vec = joints_frame0[2] - joints_frame0[1]
        right_vec[1] = 0.0
        if np.linalg.norm(right_vec) < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right_vec = _normalize(right_vec).reshape(3,)
        forward = np.cross(right_vec, up)

    forward[1] = 0.0
    if np.linalg.norm(forward) < 1e-6:
        if len(forward_candidates) > 0:
            forward = forward_candidates[0]
            forward[1] = 0.0
        else:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if np.linalg.norm(forward) < 1e-6:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return _normalize(forward).reshape(3,)

def _rot_between(a, b):
    a = _normalize(a).reshape(3,)
    b = _normalize(b).reshape(3,)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = _normalize(np.cross(a, [1,0,0])) if abs(a[0]) < 0.9 else _normalize(np.cross(a, [0,1,0]))
        return R.from_rotvec(np.pi * axis).as_matrix()
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3, dtype=np.float64)
    vx = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c)/(s*s))

def _leftmul_rotvec(rotvecs, R_left):
    Rt = R.from_rotvec(rotvecs)
    Rl = R.from_matrix(R_left)
    return (Rl * Rt).as_rotvec()

def closest_axis_unit(v):
    v = _normalize(v).reshape(3,)
    axes = np.array([[ 1,0,0],[0, 1,0],[0,0, 1],
                     [-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    dots = axes @ v
    return axes[np.argmax(np.abs(dots))]


def canonicalize_chunk_first_frame_to_plus_z(
    human_data: Dict[str, np.ndarray],
    object_data: Dict[str, np.ndarray],
    subject_id: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    poses = human_data.get("poses", None)
    trans = human_data.get("trans", None)
    betas = human_data.get("betas", None)
    if poses is None or trans is None or betas is None or poses.shape[0] == 0:
        return human_data, object_data

    gender_raw = human_data.get("gender", "neutral")
    if isinstance(gender_raw, np.ndarray):
        gender = gender_raw.item() if gender_raw.shape == () else str(gender_raw.reshape(-1)[0])
    else:
        gender = str(gender_raw)

    _, joints_cur, _, _ = visualize_smpl_arctic(poses, betas, trans, gender, subject_id)
    joints_cur_np = joints_cur.cpu().numpy()
    r_cur = joints_cur_np[:, 0, :]

    forward0 = estimate_forward_from_joints(joints_cur_np[0].astype(np.float64))
    # Canonicalize facing to -Z (180 deg opposite of +Z target).
    yaw = np.arctan2(forward0[0], -forward0[2])
    if abs(yaw) < 1e-8:
        return human_data, object_data

    R_face = R.from_rotvec(np.array([0.0, yaw, 0.0], dtype=np.float64)).as_matrix()
    orang = poses[:, :3]
    rest = poses[:, 3:]
    orang_face = _leftmul_rotvec(orang, R_face)

    p0_cur = r_cur[0].astype(np.float64)
    r_target_face = (R_face @ (r_cur - p0_cur[None, :]).T).T + p0_cur[None, :]

    poses_face = np.concatenate([orang_face, rest], axis=1).astype(np.float32)
    T = poses.shape[0]
    smpl_model = build_smpl_model(gender, subject_id)
    smpl_out = smpl_model(
        body_pose=torch.from_numpy(poses_face[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses_face[:, :3]).float(),
        left_hand_pose=torch.from_numpy(poses_face[:, 66:111]).float(),
        right_hand_pose=torch.from_numpy(poses_face[:, 111:156]).float(),
        jaw_pose=torch.from_numpy(poses_face[:, 156:159]).float(),
        leye_pose=torch.from_numpy(poses_face[:, 159:162]).float(),
        reye_pose=torch.from_numpy(poses_face[:, 162:165]).float(),
        expression=torch.zeros([T, 10]).float(),
        betas=torch.from_numpy(betas[None, :]).repeat(T, 1).float(),
        transl=torch.zeros([T, 3]).float(),
    )
    r_local_face = smpl_out.joints.detach().cpu().numpy()[:, 0, :]
    trans_face = (r_target_face - r_local_face).astype(np.float32)

    out_human = dict(human_data)
    out_human["poses"] = poses_face
    out_human["trans"] = trans_face

    out_object = dict(object_data)
    if "angles" in out_object:
        out_object["angles"] = _leftmul_rotvec(out_object["angles"], R_face).astype(np.float32)
    if "trans" in out_object:
        out_object["trans"] = ((R_face @ (out_object["trans"] - r_cur).T).T + r_target_face).astype(np.float32)
    return out_human, out_object


def transform_sequence_to_interact(human_npz, obj_npz, pivot='pelvis0', subject_id="s01"):
    poses  = human_npz['poses']
    trans  = human_npz['trans']
    betas  = human_npz['betas']
    gender = human_npz['gender'].item() if human_npz['gender'].shape == () else human_npz['gender']

    verts0, joints0, _, _ = visualize_smpl_arctic(poses, betas, trans, gender, subject_id)
    joints0_np = joints0.cpu().numpy()
    r = joints0_np[:, 0, :]

    j0 = joints0_np[0]
    up_src_cont = -_normalize(0.5*(j0[7]+j0[8]) - j0[0])
    up_src = closest_axis_unit(up_src_cont)

    up_tgt = np.array([0,1,0], dtype=np.float64)
    R_can  = _rot_between(up_src, up_tgt)

    if pivot == 'pelvis0':
        p0 = r[0].astype(np.float64)
    else:
        p0 = np.zeros(3, dtype=np.float64)

    orang = poses[:, :3]
    rest  = poses[:, 3:]

    orang_new = _leftmul_rotvec(orang, R_can)

    r_target = (R_can @ (r - p0[None,:]).T).T + p0[None,:]

    T = poses.shape[0]
    poses_upd = np.concatenate([orang_new, rest], axis=1).astype(np.float32)

    smpl_model = build_smpl_model(gender, subject_id)
    smpl_out = smpl_model(
        body_pose=torch.from_numpy(poses_upd[:, 3:66]).float(),
        global_orient=torch.from_numpy(poses_upd[:, :3]).float(),
        left_hand_pose=torch.from_numpy(poses_upd[:, 66:111]).float(),
        right_hand_pose=torch.from_numpy(poses_upd[:, 111:156]).float(),
        jaw_pose=torch.from_numpy(poses_upd[:, 156:159]).float(),
        leye_pose=torch.from_numpy(poses_upd[:, 159:162]).float(),
        reye_pose=torch.from_numpy(poses_upd[:, 162:165]).float(),
        expression=torch.zeros([T,10]).float(),
        betas=torch.from_numpy(betas[None,:]).repeat(T,1).float(),
        transl=torch.zeros([T,3]).float(),
    )
    joints_local_new = smpl_out.joints.detach().cpu().numpy()
    r_local_new = joints_local_new[:, 0, :]

    trans_new = (r_target - r_local_new).astype(np.float32)

    obj_angles = obj_npz['angles']
    obj_trans  = obj_npz['trans']
    name       = obj_npz['name'].item() if obj_npz['name'].shape == () else str(obj_npz['name'])
    arti       = obj_npz.get('arti', None)

    obj_angles_new = _leftmul_rotvec(obj_angles, R_can)
    obj_trans_new = (R_can @ (obj_trans - r).T).T + r_target

    new_poses = np.concatenate([orang_new.astype(np.float32), rest.astype(np.float32)], axis=1)

    verts_h, _, _, _ = visualize_smpl_arctic(new_poses, betas, trans_new, gender, subject_id)
    min_y_h = float(verts_h[0, :, 1].min().cpu().numpy())

    min_y = min_y_h

    shift_y = max(0.0, -min_y)
    if shift_y > 0:
        dy = np.array([0.0, shift_y, 0.0], dtype=np.float32)
        trans_new = trans_new + dy[None, :]
        obj_trans_new = obj_trans_new + dy[None, :]

    new_human = dict(
        poses=new_poses,
        betas=betas,
        trans=trans_new,
        gender=gender
    )
    new_obj = dict(
        angles=obj_angles_new.astype(np.float32),
        trans=obj_trans_new.astype(np.float32),
        name=name
    )

    if arti is not None:
        new_obj['arti'] = arti

    return new_human, new_obj

# ============ Part 3: scale_to_meters utilities ============
def scale_obj_file_inplace(input_path, scale_factor):
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    temp_path = input_path + '.tmp'
    with open(temp_path, 'w') as f:
        for line in lines:
            if line.startswith('v '):
                parts = line.split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                x_scaled = x * scale_factor
                y_scaled = y * scale_factor
                z_scaled = z * scale_factor
                f.write(f"v {x_scaled} {y_scaled} {z_scaled}\n")
            else:
                f.write(line)
    
    os.replace(temp_path, input_path)

def scale_json_coordinates_inplace(input_path, scale_factor):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    def scale_recursive(obj):
        if isinstance(obj, list):
            if all(isinstance(x, (int, float)) for x in obj):
                return [x * scale_factor for x in obj]
            else:
                return [scale_recursive(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: scale_recursive(value) for key, value in obj.items()}
        else:
            return obj
    
    scaled_data = scale_recursive(data)
    
    temp_path = input_path + '.tmp'
    with open(temp_path, 'w') as f:
        json.dump(scaled_data, f, indent=4)
    
    os.replace(temp_path, input_path)

def scale_object_files(input_dir, output_dir, scale_factor=0.001):
    """
    Copy all files from input_dir to output_dir, then scale specific files.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy all files from input to output
    for item in os.listdir(input_dir):
        src_path = os.path.join(input_dir, item)
        dst_path = os.path.join(output_dir, item)
        
        if os.path.isfile(src_path):
            shutil.copy2(src_path, dst_path)
        elif os.path.isdir(src_path):
            if os.path.exists(dst_path):
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
    
    # Scale specific files in the output directory
    obj_files = ['mesh.obj', 'mesh_tex.obj', 'top.obj', 'bottom.obj']
    json_files = ['object_params.json', 'top_keypoints_300.json', 'bottom_keypoints_300.json']
    
    for obj_file in obj_files:
        file_path = os.path.join(output_dir, obj_file)
        if os.path.exists(file_path):
            scale_obj_file_inplace(file_path, scale_factor)
    
    for json_file in json_files:
        file_path = os.path.join(output_dir, json_file)
        if os.path.exists(file_path):
            scale_json_coordinates_inplace(file_path, scale_factor)

# ============ Main processing logic ============
def main():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    description_root = Path("./data/arctic/description")
    seg_output_root = Path("./data/arctic/sequences")
    seg_output_root.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    with open(f"./data/arctic/raw/meta/misc.json", "r") as f:
        misc = json.load(f)

    statcams = {}
    for sub in misc.keys():
        statcams[sub] = {
            "world2cam": torch.FloatTensor(np.array(misc[sub]["world2cam"])),
            "intris_mat": torch.FloatTensor(np.array(misc[sub]["intris_mat"])),
        }

    # Find all .mano.npy files
    mano_ps = glob(f"./data/arctic/raw/raw_seqs/*/*.mano.npy")
    
    
    # Process each sequence
    for mano_p in mano_ps:
        sid = mano_p.split("/")[-2]
        
        # Initialize layers for this subject
        layers = construct_layers(dev, sid)
        object_tensor = ObjectTensors()
        object_tensor.to(dev)
        layers["object"] = object_tensor
        
        # Step 1: Process raw sequence (arctic_to_inter)
        out_root, subject_id, seqname = process_seq_params_direct(mano_p, dev, statcams, layers)
        
        # Step 2: Build unsplit InterAct-format sequence and split it.
        human_path = os.path.join(out_root, "human.npz")
        object_path = os.path.join(out_root, "object.npz")

        with np.load(human_path, allow_pickle=True) as H:
            human_out = {k: H[k] for k in H.files}
        with np.load(object_path, allow_pickle=True) as O:
            object_out = {k: O[k] for k in O.files}

        output_dir = os.path.join(str(seg_output_root), subject_id + "_" + seqname)
        os.makedirs(output_dir, exist_ok=True)
        np.savez(os.path.join(output_dir, "human.npz"), **human_out)
        np.savez(os.path.join(output_dir, "object.npz"), **object_out)
        split_sequence_outputs(
            sequence_dir=Path(output_dir),
            description_root=description_root,
            split_output_root=seg_output_root,
        )
        # Keep only split chunks in sequences_seg.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        
        # Delete intermediate files
        if os.path.exists(out_root):
            shutil.rmtree(out_root)
            parent_dir = os.path.dirname(out_root)
            if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                os.rmdir(parent_dir)
    
    # Step 3: Scale all object files from mm to meters
    input_objects_root = "./data/arctic/raw/meta/object_vtemplates"
    output_objects_root = "./data/arctic/objects"
    
    for obj_name in os.listdir(input_objects_root):
        input_obj_dir = os.path.join(input_objects_root, obj_name)
        if os.path.isdir(input_obj_dir):
            output_obj_dir = os.path.join(output_objects_root, obj_name)
            scale_object_files(input_obj_dir, output_obj_dir, scale_factor=0.001)

if __name__ == "__main__":
    main()

