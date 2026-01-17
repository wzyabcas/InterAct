import sys
sys.path.append("../../")

import os
import numpy as np
import joblib 
import trimesh 
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from bps_torch.bps import bps_torch
from bps_torch.tools import sample_sphere_uniform
from scipy.spatial.transform import Rotation

from manip.lafan1.utils import find_closest_points_and_vectors
import traceback

class MarkerManipDataset(Dataset):
    def __init__(
        self,
        train,
        data_root_folder,
        window=120,
        use_object_splits=True,
        use_all_data=False,
        load_num=None,
        precompute_contact=True,
        bps_dim=1024,
        corrected_data=False,
    ):
        self.train = train
        self.load_num = load_num
        self.use_all_data = use_all_data
        self.bps_dim = bps_dim

        self.precompute_contact = precompute_contact
        
        self.window = window

        self.use_joints24 = True 

        self.use_object_splits = use_object_splits 
        
        if not use_all_data:
            self.train_objects = ["largetable", "woodchair", "plasticbox", "largebox", "smallbox", \
                        "trashcan", "monitor", "floorlamp", "clothesstand", "vacuum"] # 10 objects 
            self.test_objects = ["smalltable", "whitechair", "suitcase", "tripod", "mop"] # 5 objects
        else:
            behave = ["trashbin", "boxlarge", "tablesmall", "yogaball"]
            chairs = ["15", "17", "24", "25"]
            grab = ["phone", "cup", "teapot", "spherelarge"]
            imhd = ["chair", "skateboard", "golf", "tennis"]
            intercap = ["chair", "toolbox", "umbrella", "cup"]
            neuraldome = ["bigsofa", "box", "talltable", "desk"]
            omomo = ["smalltable", "whitechair", "suitcase", "tripod", "mop"]
            self.test_objects = behave + chairs + grab + imhd + intercap + neuraldome + omomo

        self.data_root_folder = data_root_folder 
        
        self.window_data_dict = {}
        self.data_dict = {}

        self.bps_path = "./manip/data/bps.pt"
        self.subsample_idx_path = "./manip/data/bps_subsample_idx.pt"

        prefix = "alldata_" if use_all_data else ""
        min_max_mean_std_data_path = os.path.join(data_root_folder, f"{prefix}joints_min_max_mean_std_data_window_"+str(self.window)+".p")

        if not corrected_data:
            if use_all_data:
                self.datasets = ['behave', 'intercap', 'neuraldome', 'grab', 'chairs', 'omomo', 'imhd']
            else:
                self.datasets = ['omomo']
        else:
            print("Using corrected data")
            if use_all_data:
                self.datasets = ['behave_correct', 'intercap_correct', 'neuraldome', 'grab', 'chairs', 'omomo_correct', 'imhd']
            else:
                self.datasets = ['omomo_correct']

        self.prep_bps_data()
        self.window_data_dict, self.s_idx = {}, 0

        for dataset in self.datasets:
            dataset_path = os.path.join(data_root_folder, dataset)
            seq_data_path = os.path.join(dataset_path, "sequences_canonical")
            processed_data_path = os.path.join(dataset_path, "sequences_canonical_window")
            if os.path.exists(processed_data_path):
                start_idx = self.s_idx
                self.load_window_data_dict(dataset)
                self.get_bps_from_window_data_dict(start_idx, self.s_idx)
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
       
        if self.use_object_splits:
            self.window_data_dict = self.filter_out_object_split()

        # Get train and validation statistics. 
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

        dest_obj_bps_npy_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name,"object_bps.npy")
        obj_bps_data_ = np.load(dest_obj_bps_npy_path) # T X N X 3 
        obj_bps_data = torch.from_numpy(obj_bps_data_)
        if self.bps_dim != 1024:
            obj_bps_data = obj_bps_data[:, self.subsample_idx, :]
        
        num_markers = 77
        normalized_markers = self.normalize_markers_min_max(markers) # T X 77 X 3
        new_motion = torch.cat((normalized_markers.reshape(-1, num_markers*3), motion[:, num_markers*3:]), dim=1) # T X 559

        if not self.precompute_contact:
            # get object verts (verts based on bps)
            bps = self.bps['obj'].cpu().numpy()
            obj_verts = bps + obj_bps_data_ + self.window_data_dict[index]['obj_trans'][:, None, :]

            # get contact info
            contact_threshold = 0.1
            closest_indices, closest_vectors = find_closest_points_and_vectors(markers.cpu().numpy(), obj_verts)
            closest_indices = torch.from_numpy(closest_indices).float()
            closest_vectors = torch.from_numpy(closest_vectors).float()
            contact_labels = (closest_vectors.norm(dim=-1) < contact_threshold).float() # T X 77

            obj_closest_indices, obj_closest_vectors = find_closest_points_and_vectors(obj_verts, markers.cpu().numpy())
            obj_closest_indices = torch.from_numpy(obj_closest_indices).float()
            obj_closest_vectors = torch.from_numpy(obj_closest_vectors).float()
            obj_contact_labels = (obj_closest_vectors.norm(dim=-1) < contact_threshold).float() # T X bps_dim
            if self.bps_dim != 1024:
                obj_contact_labels = obj_contact_labels[:, self.subsample_idx]
            assert obj_contact_labels.shape[1] == self.bps_dim
        else:
            seq_data_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name)
            dest_contact_npz_path = os.path.join(seq_data_path, "contact.npz")
            contact_data = np.load(dest_contact_npz_path)
            contact_labels = torch.from_numpy(contact_data['contact_labels']).float()
            closest_indices = torch.from_numpy(contact_data['closest_indices']).float()
            closest_vectors = torch.from_numpy(contact_data['closest_vectors']).float()
            obj_contact_labels = torch.from_numpy(contact_data['obj_contact_labels']).float()
            if self.bps_dim != 1024:
                obj_contact_labels = obj_contact_labels[:, self.subsample_idx]

        # Add padding. 
        actual_steps = markers.shape[0]
        if actual_steps < self.window:
            paded_new_motion = torch.cat((new_motion, torch.zeros(self.window-actual_steps, new_motion.shape[1])), dim=0)
            paded_ori_motion = torch.cat((motion, torch.zeros(self.window-actual_steps, motion.shape[1])), dim=0)
            paded_markers = torch.cat((markers, torch.zeros(self.window-actual_steps, markers.shape[1], markers.shape[2])), dim=0)
            paded_contact_labels = torch.cat((contact_labels, torch.zeros(self.window-actual_steps, contact_labels.shape[1])), dim=0)
            paded_closest_indices = torch.cat((closest_indices, torch.zeros(self.window-actual_steps, closest_indices.shape[1])), dim=0)
            paded_closest_vectors = torch.cat((closest_vectors, torch.zeros(self.window-actual_steps, closest_vectors.shape[1], closest_vectors.shape[2])), dim=0)
            paded_obj_contact_labels = torch.cat((obj_contact_labels, torch.zeros(self.window-actual_steps, obj_contact_labels.shape[1])), dim=0)

            obj_bps_data = obj_bps_data[:actual_steps]
            paded_obj_bps = torch.cat((obj_bps_data.reshape(actual_steps, -1), \
                torch.zeros(self.window-actual_steps, obj_bps_data.reshape(actual_steps, -1).shape[1])), dim=0)
            paded_obj_angles = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_angles']).float(), \
                torch.zeros(self.window-actual_steps, 3)), dim=0)
            paded_obj_trans = torch.cat((torch.from_numpy(self.window_data_dict[index]['obj_trans']).float(), \
                torch.zeros(self.window-actual_steps, 3)), dim=0)

        else:
            paded_new_motion = new_motion
            paded_ori_motion = motion
            paded_markers = markers
            paded_contact_labels = contact_labels
            paded_closest_indices = closest_indices
            paded_closest_vectors = closest_vectors
            paded_obj_contact_labels = obj_contact_labels

            paded_obj_bps = obj_bps_data.reshape(actual_steps, -1)
            paded_obj_angles = torch.from_numpy(self.window_data_dict[index]['obj_angles']).float()
            paded_obj_trans = torch.from_numpy(self.window_data_dict[index]['obj_trans']).float()
        
        
        data_input_dict = {}
        data_input_dict['motion'] = paded_new_motion
        data_input_dict['ori_motion'] = paded_ori_motion
        data_input_dict['markers'] = paded_markers
        data_input_dict['contact_labels'] = paded_contact_labels
        data_input_dict['closest_indices'] = paded_closest_indices
        data_input_dict['closest_vectors'] = paded_closest_vectors
        data_input_dict['obj_contact_labels'] = paded_obj_contact_labels
        
        data_input_dict['obj_bps'] = paded_obj_bps
        data_input_dict['obj_angles'] = paded_obj_angles
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
            
            bps = {
                'obj': bps_obj.cpu(),
            }
            print("Generate new bps data to:{0}".format(self.bps_path))
            torch.save(bps, self.bps_path)
        
        self.bps = torch.load(self.bps_path)

        self.bps_torch = bps_torch()

        self.obj_bps = self.bps['obj'].cuda()
        self.obj_bps_1024 = self.obj_bps
        if self.bps_dim != 1024:
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

            if not self.use_all_data:
                if self.train and object_data['name'] not in self.train_objects: continue
                if (not self.train) and object_data['name'] not in self.test_objects: continue
            else:
                if self.train and object_data['name'] in self.test_objects: continue
                if (not self.train) and object_data['name'] not in self.test_objects: continue

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
            self.window_data_dict[self.s_idx]['markers'] = np.load(os.path.join(seq_path, "markers.npy")) # T x 77 x 3
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
                self.data_dict[s_idx]['markers'] = np.load(os.path.join(seq_path, "markers.npy"))
                self.data_dict[s_idx]['motion'] = np.load(os.path.join(seq_path, "motion.npy"))
                self.data_dict[s_idx]['seq_name'] = seq
                self.data_dict[s_idx]['seq_len'] = len(self.data_dict[s_idx]['motion']) 
                self.data_dict[s_idx]['dataset'] = dataset
                s_idx += 1
            except Exception as e:
                # delete
                del self.data_dict[s_idx]
                fail_num += 1
                if isinstance(e, KeyboardInterrupt): raise e
                continue
        print(f"Failed to load {fail_num} sequences in {dataset}")
            
    
    def cal_normalize_data_input(self):
        for index in tqdm(self.data_dict, desc="Processing"):
            seq_name = self.data_dict[index]['seq_name']

            object_name = self.data_dict[index]['obj_name']

            markers = self.data_dict[index]['markers'] # T X 77 X 3
            motion = self.data_dict[index]['motion'] #T X 559
            
            obj_trans = self.data_dict[index]['obj_trans'] # T X 3
            obj_angles = self.data_dict[index]['obj_angles'] # T X 3 
            dataset = self.data_dict[index]['dataset']
            dataset_path = os.path.join(self.data_root_folder, dataset)
            seq_data_path = os.path.join(dataset_path, "sequences_canonical_window")
            os.makedirs(seq_data_path, exist_ok=True)

            num_steps = self.data_dict[index]['seq_len']
            for start_t_idx in range(0, num_steps, self.window//2):
                end_t_idx = start_t_idx + self.window 
                if end_t_idx >= num_steps:
                    end_t_idx = num_steps 

                # Skip the segment that has a length < 60 
                if end_t_idx - start_t_idx < 60:
                    continue 

                self.window_data_dict[self.s_idx] = {}
                self.window_data_dict[self.s_idx]['motion'] = motion[start_t_idx:end_t_idx,:553]
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
                np.savez(os.path.join(save_path,'object.npz'), **obj)
                np.save(os.path.join(save_path, 'markers.npy'), self.window_data_dict[self.s_idx]['markers'])
                np.save(os.path.join(save_path, 'motion.npy'), self.window_data_dict[self.s_idx]['motion'])

                self.s_idx += 1

    def get_bps_from_window_data_dict(self, start_idx, end_idx):
        # Given window_data_dict which contains canonizalized information, compute its corresponding BPS representation. 
        for k in tqdm(range(start_idx, end_idx), desc="Computing BPS"):
            try:
                window_data = self.window_data_dict[k]

                seq_name = window_data['seq_name']
                obj_name = window_data['obj_name']
                obj_trans = window_data['obj_trans']
                obj_angles = window_data['obj_angles']
                dataset = window_data['dataset']
                seq_data_path = os.path.join(self.data_root_folder, dataset, "sequences_canonical_window", seq_name)
                obj_mesh_path = os.path.join(self.data_root_folder, dataset, "objects", f"{obj_name}/{obj_name}.obj")
                dest_obj_bps_npy_path = os.path.join(seq_data_path, "object_bps.npy")
                if os.path.exists(dest_obj_bps_npy_path): continue

                mesh_obj = trimesh.load(obj_mesh_path)
                obj_verts, obj_faces = mesh_obj.vertices, mesh_obj.faces

                angle_matrix = Rotation.from_rotvec(obj_angles).as_matrix()
                obj_verts = (obj_verts)[None, ...]
                obj_verts = np.matmul(obj_verts, np.transpose(angle_matrix, (0, 2, 1))) + obj_trans[:, None, :]

                torch_obj_verts = torch.from_numpy(obj_verts).float().cuda()
                torch_obj_trans = torch.from_numpy(obj_trans).float().cuda()

                # Get object geometry 
                if not os.path.exists(dest_obj_bps_npy_path):
                # if True:
                    object_bps = self.compute_object_geo_bps(torch_obj_verts, torch_obj_trans)
                    np.save(dest_obj_bps_npy_path, object_bps.data.detach().cpu().numpy())
            
            except:
                print(f"Failed to compute BPS for {seq_name}")
                traceback.print_exc()
                continue
    
    def get_contact_from_window_data_dict(self, start_idx, end_idx):
        # Given window_data_dict which contains canonizalized information, compute its corresponding contact info. 
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
                obj_bps_data = np.load(obj_bps_npy_path) # T X N X 3
                print("obj_bps_data.shape", obj_bps_data.shape)
                print("bps.shape", bps.shape)
                print("obj_trans.shape", obj_trans.shape)
                obj_verts = bps + obj_bps_data + obj_trans[:, None, :]

                # get contact info
                contact_threshold = 0.1
                closest_indices, closest_vectors = find_closest_points_and_vectors(markers, obj_verts)
                contact_labels = (np.linalg.norm(closest_vectors, axis=-1) < contact_threshold) # T X 77

                obj_closest_indices, obj_closest_vectors = find_closest_points_and_vectors(obj_verts, markers)
                obj_contact_labels = (np.linalg.norm(obj_closest_vectors, axis=-1) < contact_threshold) # T X 1024
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
        bps_object_geo = self.bps_torch.encode(x=obj_verts, \
                    feature_type=['deltas'], \
                    custom_basis=self.obj_bps_1024.repeat(obj_trans.shape[0], \
                    1, 1)+obj_trans[:, None, :])['deltas'] # T X N X 3 

        return bps_object_geo
    
    def filter_out_object_split(self):
        # Remove some sequences from window_data_dict such that we have some unseen objects during testing. 
        new_cnt = 0
        new_window_data_dict = {}
        for k in self.window_data_dict:
            window_data = self.window_data_dict[k]
            object_name = window_data['obj_name']
            if self.train:
                if not self.use_all_data:
                    if object_name in self.train_objects:
                        new_window_data_dict[new_cnt] = self.window_data_dict[k]
                        new_cnt += 1
                else:
                    if object_name not in self.test_objects:
                        new_window_data_dict[new_cnt] = self.window_data_dict[k]
                        new_cnt += 1

            if (not self.train) and object_name in self.test_objects:
                new_window_data_dict[new_cnt] = self.window_data_dict[k]
                new_cnt += 1

        return new_window_data_dict
    
    def extract_min_max_mean_std_from_data(self):
        all_marker_data = []

        for s_idx in self.window_data_dict:
            curr_marker_data = self.window_data_dict[s_idx]['motion'] # T X D 
            all_marker_data.append(curr_marker_data[:, :77*3])

        all_marker_data = np.vstack(all_marker_data).reshape(-1, 231) # (N*T) X 77*3 
        min_markers = all_marker_data.min(axis=0)
        max_markers = all_marker_data.max(axis=0)

        stats_dict = {}
        stats_dict['global_markers_min'] = min_markers
        stats_dict['global_markers_max'] = max_markers

        return stats_dict 
    
    def normalize_markers_min_max(self, ori_makers):
        # ori_makers: T X 77 X 3 
        normalized_markers = (ori_makers - self.global_markers_min.to(ori_makers.device)) / (self.global_markers_max.to(ori_makers.device)-self.global_markers_min.to(ori_makers.device))
        normalized_markers = normalized_markers * 2 - 1 # [-1, 1] range 
        return normalized_markers # T X 77 X 3 
    
    def de_normalize_markers_min_max(self, normalize_markers):
        normalize_markers = (normalize_markers + 1) * 0.5 # [0, 1] range
        de_markers = normalize_markers * (self.global_markers_max.to(normalize_markers.device)-self.global_markers_min.to(normalize_markers.device)) + self.global_markers_min.to(normalize_markers.device)
        return de_markers # T X 77 X 3 