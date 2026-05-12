import sys
sys.path.append("../../")

import os
import numpy as np
import joblib
import json
import trimesh
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from scipy.spatial.transform import Rotation

from manip.lafan1.utils import rotate_at_frame_w_obj, find_closest_points_and_vectors
import traceback


MODEL_PATH = "../data/models/"
SMPLH_PATH = MODEL_PATH + "smplh/"


class MarkerManipDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_all_data=False,
        load_num=None,
        precompute_contact=True,
        bps_dim=1024,
        split_train_val=False,
        **kwargs
    ):
        self.train = train
        self.load_num = load_num
        self.use_all_data = use_all_data
        self.bps_dim = bps_dim

        self.precompute_contact = precompute_contact

        self.window = window

        self.split_train_val = split_train_val

        self.data_root_folder = data_root_folder

        self.window_data_dict = {}
        self.data_dict = {}

        self.bps_path = "./manip/data/bps.pt"
        self.subsample_idx_path = "./manip/data/bps_subsample_idx.pt"

        prefix = "alldata_" if use_all_data else ""
        min_max_mean_std_data_path = os.path.join(data_root_folder, f"{prefix}joints_min_max_mean_std_data_window_"+str(self.window)+".p")

        # corrected_data=False, use_all_data=True
        self.datasets = ['behave', 'intercap', 'neuraldome', 'grab', 'chairs', 'omomo', 'imhd']
        self.datasets = ['behave'] # for debug

        self.prep_bps_data()
        self.window_data_dict, self.s_idx = {}, 0

        for dataset in self.datasets:
            dataset_path = os.path.join(data_root_folder, dataset)
            seq_data_path = os.path.join(dataset_path, "sequences_canonical")
            processed_data_path = os.path.join(dataset_path, "sequences_canonical_window")
            if os.path.exists(processed_data_path):
                start_idx = self.s_idx
                self.load_window_data_dict(dataset)
                self.get_canon_bps_from_window_data_dict(start_idx, self.s_idx)
                if self.precompute_contact:
                    self.get_contact_from_window_data_dict(start_idx, self.s_idx)
            else:
                if os.path.exists(seq_data_path):
                    self.load_data_dict(dataset)

                print("proecssing to generate sequences_canonical_window")
                self.cal_normalize_data_input()

        if os.path.exists(min_max_mean_std_data_path):
            min_max_mean_std_jpos_data = joblib.load(min_max_mean_std_data_path)
        else:
            if self.train:
                min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
                joblib.dump(min_max_mean_std_jpos_data, min_max_mean_std_data_path)

        self.global_markers_min = torch.from_numpy(min_max_mean_std_jpos_data['global_markers_min']).float().reshape(77, 3)[None, ...]
        self.global_markers_max = torch.from_numpy(min_max_mean_std_jpos_data['global_markers_max']).float().reshape(77, 3)[None, ...]

        if self.split_train_val:
            print("Splitting train and validation data")
            split_file_name = "alldata_split_train_val.json" if self.use_all_data else "omomo_split_train_val.json"
            split_file_name = os.path.join(self.data_root_folder, split_file_name)
            if os.path.exists(split_file_name):
                with open(split_file_name, 'r') as f:
                    split_dict = json.load(f)
                target_seq_names = split_dict['train'] if self.train else split_dict['val']
            else:
                train_num = len(self.window_data_dict) // 10 * 9
                val_num = len(self.window_data_dict) - train_num
                seq_names = list([v['seq_name'] for v in self.window_data_dict.values()])
                np.random.shuffle(seq_names)
                train_seq_names = seq_names[:train_num]
                val_seq_names = seq_names[train_num:]
                with open(split_file_name, 'w') as f:
                    json.dump({'train': train_seq_names, 'val': val_seq_names}, f)
                target_seq_names = train_seq_names if self.train else val_seq_names
            self.window_data_dict = {k: v for k, v in self.window_data_dict.items() if v['seq_name'] in target_seq_names}
            new_window_data_dict = {}
            for k, v in self.window_data_dict.items():
                new_window_data_dict[len(new_window_data_dict)] = v
            self.window_data_dict = new_window_data_dict

        if self.train:
            print("Total number of windows for training:{0}".format(len(self.window_data_dict)))
        else:
            print("Total number of windows for validation:{0}".format(len(self.window_data_dict)))

    def __len__(self):
        return len(self.window_data_dict)

    def __getitem__(self, index):
        motion = torch.from_numpy(self.window_data_dict[index]['motion']).float()
        markers = torch.from_numpy(self.window_data_dict[index]['markers']).float()

        seq_name = self.window_data_dict[index]['seq_name']
        obj_name = self.window_data_dict[index]['obj_name']
        dataset = self.window_data_dict[index]['dataset']

        dest_obj_canon_bps_npy_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name, "canoned_object_bps.npy")
        obj_canon_bps_data = torch.from_numpy(np.load(dest_obj_canon_bps_npy_path))  # T X N X 3

        num_markers = 77
        normalized_markers = self.normalize_markers_min_max(markers)  # T X 77 X 3
        new_motion = torch.cat((normalized_markers.reshape(-1, num_markers*3), motion[:, num_markers*3:]), dim=1)  # T X 559

        # Load precomputed contact data
        seq_data_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name)
        contact_data = np.load(os.path.join(seq_data_path, "contact.npz"), allow_pickle=True)
        closest_vectors = torch.from_numpy(contact_data['closest_vectors']).float()

        # Add padding.
        actual_steps = markers.shape[0]
        if actual_steps < self.window:
            paded_new_motion = torch.cat((new_motion, torch.zeros(self.window-actual_steps, new_motion.shape[1])), dim=0)
            paded_markers = torch.cat((markers, torch.zeros(self.window-actual_steps, markers.shape[1], markers.shape[2])), dim=0)
            paded_closest_vectors = torch.cat((closest_vectors, torch.zeros(self.window-actual_steps, closest_vectors.shape[1], closest_vectors.shape[2])), dim=0)
            paded_obj_angles = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_angles']).float(),
                torch.zeros(self.window-actual_steps, 3)), dim=0)
            paded_obj_trans = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_trans']).float(),
                torch.zeros(self.window-actual_steps, 3)), dim=0)
            paded_quat = Rotation.from_rotvec(self.window_data_dict[index]['obj_angles']).as_quat()
            paded_quat = torch.cat((torch.from_numpy(paded_quat).float(),
                torch.zeros(self.window-actual_steps, 4)), dim=0)
        else:
            paded_new_motion = new_motion
            paded_markers = markers
            paded_closest_vectors = closest_vectors
            paded_obj_angles = torch.from_numpy(self.window_data_dict[index]['obj_angles']).float()
            paded_obj_trans = torch.from_numpy(self.window_data_dict[index]['obj_trans']).float()
            paded_quat = Rotation.from_rotvec(self.window_data_dict[index]['obj_angles']).as_quat()
            paded_quat = torch.from_numpy(paded_quat).float()

        data_input_dict = {}
        data_input_dict['motion'] = paded_new_motion
        data_input_dict['markers'] = paded_markers
        data_input_dict['closest_vectors'] = paded_closest_vectors
        data_input_dict['obj_canon_bps_data'] = obj_canon_bps_data[0]  # N X 3
        data_input_dict['obj_angles'] = paded_obj_angles
        data_input_dict['obj_quat'] = paded_quat
        data_input_dict['obj_trans'] = paded_obj_trans
        data_input_dict['dataset'] = dataset
        data_input_dict['seq_name'] = seq_name
        data_input_dict['obj_name'] = obj_name
        data_input_dict['seq_len'] = actual_steps

        return data_input_dict

    def prep_bps_data(self):
        n_obj = 1024
        r_obj = 1.0
        if not os.path.exists(self.bps_path):
            bps_obj = sample_sphere_uniform(n_points=n_obj, radius=r_obj).reshape(1, -1, 3)
            bps = {'obj': bps_obj.cpu()}
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)

        self.bps = torch.load(self.bps_path)
        self.bps_torch = bps_torch()
        self.obj_bps = self.bps['obj'].cuda()

        # bps_dim is always < 1024, subsample
        if not os.path.exists(self.subsample_idx_path):
            self.subsample_idx = torch.randperm(1024)[:self.bps_dim]
            torch.save(self.subsample_idx, self.subsample_idx_path)
        else:
            self.subsample_idx = torch.load(self.subsample_idx_path)
        self.obj_bps = self.obj_bps[:, self.subsample_idx, :]

    def load_window_data_dict(self, dataset):
        curr_num = 0
        dataset_path = os.path.join(self.data_root_folder, dataset)
        seq_data_path = os.path.join(dataset_path, "sequences_canonical_window")
        for seq in tqdm(os.listdir(seq_data_path), desc=f"Loading {dataset}"):
            if self.load_num is not None and curr_num >= self.load_num: break

            seq_path = os.path.join(seq_data_path, seq)
            object_data_path = os.path.join(seq_path, "object.npz")
            object_data = np.load(object_data_path, allow_pickle=True)

            motion_path = os.path.join(seq_path, "motion.npy")
            if not os.path.exists(motion_path): continue
            motion = np.load(motion_path)
            length = len(motion)
            if length < 60: continue

            self.window_data_dict[self.s_idx] = {}
            self.window_data_dict[self.s_idx]['obj_name'] = str(object_data['name'])
            self.window_data_dict[self.s_idx]['dataset'] = dataset
            self.window_data_dict[self.s_idx]['obj_trans'] = object_data['trans']
            self.window_data_dict[self.s_idx]['obj_angles'] = object_data['angles']
            self.window_data_dict[self.s_idx]['markers'] = np.load(os.path.join(seq_path, "markers.npy"))  # T x 77 x 3
            self.window_data_dict[self.s_idx]['motion'] = motion
            self.window_data_dict[self.s_idx]['seq_name'] = seq
            self.window_data_dict[self.s_idx]['seq_len'] = length
            self.s_idx += 1
            curr_num += 1

    def load_data_dict(self, dataset):
        self.data_dict = {}
        s_idx = 0
        dataset_path = os.path.join(self.data_root_folder, dataset)
        seq_data_path = os.path.join(dataset_path, "sequences_canonical")
        for seq in tqdm(os.listdir(seq_data_path), desc=f"Loading {dataset}"):
            fail_num = 0
            try:
                self.data_dict[s_idx] = {}
                seq_path = os.path.join(seq_data_path, seq)
                object_data_path = os.path.join(seq_path, "object.npz")
                object_data = np.load(object_data_path, allow_pickle=True)
                self.data_dict[s_idx]['obj_name'] = str(object_data['name'])
                self.data_dict[s_idx]['obj_trans'] = object_data['trans']
                self.data_dict[s_idx]['obj_angles'] = object_data['angles']
                motion = np.load(os.path.join(seq_path, "motion.npy"))
                self.data_dict[s_idx]['markers'] = motion[:, :77*3].reshape(-1, 77, 3)  # T x 77 x 3
                self.data_dict[s_idx]['motion'] = motion 
                self.data_dict[s_idx]['seq_name'] = seq
                self.data_dict[s_idx]['seq_len'] = len(self.data_dict[s_idx]['motion'])
                self.data_dict[s_idx]['dataset'] = dataset
                s_idx += 1
            except Exception as e:
                del self.data_dict[s_idx]
                fail_num += 1
                if isinstance(e, KeyboardInterrupt): raise e
                continue
        print(f"Failed to load {fail_num} sequences in {dataset}")

    def cal_normalize_data_input(self):
        for index in tqdm(self.data_dict, desc="Processing"):
            seq_name = self.data_dict[index]['seq_name']
            object_name = self.data_dict[index]['obj_name']
            markers = self.data_dict[index]['markers']  # T X 77 X 3
            motion = self.data_dict[index]['motion']  # T X 559
            obj_trans = self.data_dict[index]['obj_trans']  # T X 3
            obj_angles = self.data_dict[index]['obj_angles']  # T X 3
            dataset = self.data_dict[index]['dataset']
            dataset_path = os.path.join(self.data_root_folder, dataset)
            seq_data_path = os.path.join(dataset_path, "sequences_canonical_window")
            os.makedirs(seq_data_path, exist_ok=True)

            num_steps = self.data_dict[index]['seq_len']
            for start_t_idx in range(0, num_steps, self.window//2):
                end_t_idx = start_t_idx + self.window
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps

                if end_t_idx - start_t_idx < 60:
                    continue

                self.window_data_dict[self.s_idx] = {}
                self.window_data_dict[self.s_idx]['motion'] = motion[start_t_idx:end_t_idx, :553]
                self.window_data_dict[self.s_idx]['seq_name'] = seq_name + '_' + str(start_t_idx)
                self.window_data_dict[self.s_idx]['obj_name'] = object_name
                self.window_data_dict[self.s_idx]['markers'] = markers[start_t_idx:end_t_idx]
                self.window_data_dict[self.s_idx]['obj_trans'] = obj_trans[start_t_idx:end_t_idx]
                self.window_data_dict[self.s_idx]['obj_angles'] = obj_angles[start_t_idx:end_t_idx]
                self.window_data_dict[self.s_idx]['seq_len'] = end_t_idx - start_t_idx
                self.window_data_dict[self.s_idx]['dataset'] = dataset

                obj = {
                    'angles': self.window_data_dict[self.s_idx]['obj_angles'],
                    'trans': self.window_data_dict[self.s_idx]['obj_trans'],
                    'name': self.window_data_dict[self.s_idx]['obj_name']
                }
                save_path = os.path.join(seq_data_path, self.window_data_dict[self.s_idx]['seq_name'])
                os.makedirs(save_path, exist_ok=True)
                np.savez(os.path.join(save_path, 'object.npz'), **obj)
                np.save(os.path.join(save_path, 'markers.npy'), self.window_data_dict[self.s_idx]['markers'])
                np.save(os.path.join(save_path, 'motion.npy'), self.window_data_dict[self.s_idx]['motion'])

                self.s_idx += 1

    def get_canon_bps_from_window_data_dict(self, start_idx, end_idx):
        for k in tqdm(range(start_idx, end_idx), desc="Computing Canoned BPS"):
            try:
                window_data = self.window_data_dict[k]

                seq_name = window_data['seq_name']
                obj_name = window_data['obj_name']
                obj_trans = window_data['obj_trans']
                dataset = window_data['dataset']
                seq_data_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name)
                obj_mesh_path = os.path.join(self.data_root_folder, dataset, "objects", f"{obj_name}/{obj_name}.obj")
                dest_obj_bps_npy_path = os.path.join(seq_data_path, "canoned_object_bps.npy")
                if os.path.exists(dest_obj_bps_npy_path): continue

                mesh_obj = trimesh.load(obj_mesh_path)
                obj_verts = mesh_obj.vertices
                obj_verts = (obj_verts)[None, ...]

                torch_obj_verts = torch.from_numpy(obj_verts).float().cuda()
                torch_obj_trans = torch.from_numpy(obj_trans).float().cuda()
                torch_obj_trans = torch.zeros_like(torch_obj_trans[:1]).float().cuda()

                if not os.path.exists(dest_obj_bps_npy_path):
                    object_bps = self.compute_object_geo_bps(torch_obj_verts, torch_obj_trans)
                    np.save(dest_obj_bps_npy_path, object_bps.data.detach().cpu().numpy())

            except:
                print(f"Failed to compute BPS for {seq_name}")
                traceback.print_exc()
                continue

    def get_contact_from_window_data_dict(self, start_idx, end_idx):
        for k in tqdm(range(start_idx, end_idx), desc="Computing Contact"):
            try:
                window_data = self.window_data_dict[k]

                seq_name = window_data['seq_name']
                obj_trans = window_data['obj_trans']
                dataset = window_data['dataset']
                markers = window_data['markers']
                seq_data_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name)
                obj_bps_npy_path = os.path.join(seq_data_path, "object_bps.npy")
                dest_contact_npz_path = os.path.join(seq_data_path, "contact.npz")
                if os.path.exists(dest_contact_npz_path): continue
                assert os.path.exists(obj_bps_npy_path), f"Object BPS not found for {seq_name}"

                bps = self.bps['obj'].cpu().numpy()
                obj_bps_data = np.load(obj_bps_npy_path)  # T X N X 3
                obj_verts = bps + obj_bps_data + obj_trans[:, None, :]

                contact_threshold = 0.1
                closest_indices, closest_vectors = find_closest_points_and_vectors(markers, obj_verts)
                contact_labels = (np.linalg.norm(closest_vectors, axis=-1) < contact_threshold)  # T X 77

                obj_closest_indices, obj_closest_vectors = find_closest_points_and_vectors(obj_verts, markers)
                obj_contact_labels = (np.linalg.norm(obj_closest_vectors, axis=-1) < contact_threshold)  # T X 1024
                assert obj_contact_labels.shape[1] == 1024

                contact_dict = {
                    'contact_labels': contact_labels,
                    'closest_indices': closest_indices,
                    'closest_vectors': closest_vectors,
                    'obj_contact_labels': obj_contact_labels,
                    'obj_closest_indices': obj_closest_indices,
                    'obj_closest_vectors': obj_closest_vectors
                }
                np.savez(dest_contact_npz_path, **contact_dict)

            except:
                print(f"Failed to compute Contact for {seq_name}")
                traceback.print_exc()
                continue

    def compute_object_geo_bps(self, obj_verts, obj_trans):
        # obj_verts: T X Nv X 3, obj_trans: T X 3
        bps_object_geo = self.bps_torch.encode(x=obj_verts,
                    feature_type=['deltas'],
                    custom_basis=self.obj_bps.repeat(obj_trans.shape[0],
                    1, 1)+obj_trans[:, None, :])['deltas']  # T X N X 3

        return bps_object_geo

    def extract_min_max_mean_std_from_data(self):
        all_marker_data = []

        for s_idx in self.window_data_dict:
            curr_marker_data = self.window_data_dict[s_idx]['motion']  # T X D
            all_marker_data.append(curr_marker_data[:, :77*3])

        all_marker_data = np.vstack(all_marker_data).reshape(-1, 231)  # (N*T) X 77*3
        min_markers = all_marker_data.min(axis=0)
        max_markers = all_marker_data.max(axis=0)

        stats_dict = {}
        stats_dict['global_markers_min'] = min_markers
        stats_dict['global_markers_max'] = max_markers

        return stats_dict

    def normalize_markers_min_max(self, ori_makers):
        # ori_makers: T X 77 X 3
        normalized_markers = (ori_makers - self.global_markers_min.to(ori_makers.device)) / (self.global_markers_max.to(ori_makers.device)-self.global_markers_min.to(ori_makers.device))
        normalized_markers = normalized_markers * 2 - 1  # [-1, 1] range
        return normalized_markers  # T X 77 X 3
