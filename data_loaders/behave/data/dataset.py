import torch
from torch.utils import data
import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import spacy
from torch.utils.data._utils.collate import default_collate
from data_loaders.behave.utils.word_vectorizer import WordVectorizer
from data_loaders.behave.utils.get_opt import get_opt
import trimesh
from scipy.spatial.transform import Rotation
from data_loaders.behave.scripts.motion_process import recover_from_ric, extract_features, get_human_representation
import scipy.sparse
from data_loaders.behave.utils.paramUtil import *
from utils.utils import recover_obj_points
from data_loaders.behave.utils.plot_script import plot_3d_motion


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)



class TextOnlyDataset(data.Dataset):
    def __init__(self, opt, mean, std, datasets, split_file, obj_split,debug=0):
        self.mean = mean[:opt.dim_pose]
        self.std = std[:opt.dim_pose]
        self.opt = opt
        self.data_dict = []
        self.max_length = 20
        self.pointer = 0
        self.fixed_length = 300
        self.normal_dim = opt.dim_pose

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.id_list = id_list

        min_motion_len = 40

        
        new_name_list = []
        length_list = []
        for dataset in datasets:
            print(f"Loading dataset {dataset} ...")
            dataset_path = os.path.join(opt.data_root, dataset)
            sequences_path = os.path.join(dataset_path, 'sequences_canonical')
            sequences_names = os.listdir(sequences_path)
            if debug:
                L=len(sequences_names)
                L_debug=min(L,200)
                sequences_names=sequences_names[:L_debug]
            for name in tqdm(sequences_names):
                try:
                    motion = np.load(os.path.join(sequences_path, name, 'motion.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) > 300):
                        continue
                    obj = np.load(pjoin(sequences_path, name, 'object.npz'), allow_pickle=True)
                    # load obj points----------------
                    obj_name = str(obj['name'])

                    id_name = obj_name if obj_split else name

                    if id_name not in id_list:
                        # print(id_list)
                        # print(obj_name)
                        continue

                    obj_path = pjoin(dataset_path, 'objects')
                    # obj_path = pjoin(opt.data_root, 'behave', 'object_mesh')
                    # mesh_path = os.path.join(obj_path, simplified_mesh[obj_name])

                    # temp_simp = trimesh.load(mesh_path)
                    # obj_points = np.array(temp_simp.vertices).astype(np.float32)
                    # obj_faces = np.array(temp_simp.faces).astype(np.float32)

                    # bps 
                    bps_path = pjoin(dataset_path, 'objects_bps/{}/{}_1024.npy'.format(obj_name,obj_name))
                    obj_bps = np.load(bps_path, allow_pickle=True)

                    # sample object points
                    obj_sample_path = pjoin(obj_path, '{}/sample_points.npy'.format(obj_name))
                    # print(obj_sample_path)
                    # o_choose = np.load(obj_sample_path)

                    # # contact_input = np.load(pjoin(opt.data_root, 'affordance_data/contact_'+name + '.npy'), allow_pickle=True)[None][0]
                                    
                    # # center the meshes
                    # center = np.mean(obj_points, 0)
                    # obj_points -= center


                    # obj_points = obj_points[o_choose]
                    # obj_normals = obj_faces[o_choose]
                    obj_points = np.load(obj_sample_path)
                    obj_normals = np.zeros((400, 3)) 

                    text_data = []
                    flag = False
                    with cs.open(pjoin(sequences_path, name, 'text.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            # f_tag = float(line_split[2])
                            # to_tag = float(line_split[3])
                            # f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            # to_tag = 0.0 if np.isnan(to_tag) else to_tag
                            # TODO: hardcode
                            f_tag = to_tag = 0.0

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = { 
                                                            'motion': motion,
                                                            'text':[text_dict],
                                                            'seq_name': dataset+'_'+obj_name,
                                                            'length': len(motion),
                                                            'obj_points': obj_points,
                                                            'obj_normals':obj_normals,
                                                            'obj_bps': obj_bps
                                                            }
                                    new_name_list.append(new_name)
                                    length_list.append(len(motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = { 'motion':motion,
                                            'text': text_data,
                                            'seq_name': dataset+'_'+obj_name,
                                            'length': len(motion),
                                            'obj_points': obj_points,
                                            'obj_normals':obj_normals,
                                            'obj_bps': obj_bps
                                            }
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as err:
                    print(err.__class__.__name__) # 
                    print(err) 
                    pass

        zipped_lists = zip(length_list, new_name_list)
        sorted_lists = sorted(zipped_lists, key=lambda x: x[0])
        self.length_arr, self.name_list = zip(*sorted_lists)

        self.data_dict = data_dict

    def inv_transform(self, data):
        data = data.clone()
        if data.shape[-1] == 559:
            data = data * self.std + self.mean
        else:
            data[..., :553] = data[..., :553] * self.std[:553] + self.mean[:553]
        return data

        
    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(
            data.device) + torch.from_numpy(self.mean).to(data.device)
        return data

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]] 

        text_list, seq_name, obj_points, obj_normals,obj_bps = data['text'],  data['seq_name'],  data['obj_points'], data['obj_normals'], data['obj_bps']
        length = data['length']
        motion = data['motion']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']
        return None, None, caption, None, motion, length, None, seq_name, obj_points, obj_normals, obj_bps





'''For use of training text motion matching model, and evaluations, with bps and normals as input.'''
class Text2MotionDatasetV3(data.Dataset):
    def __init__(self, opt, mean, std, datasets, split_file, obj_split, w_vectorizer,debug=0):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 30
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40
        self.normal_dim = opt.dim_pose

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())


        new_name_list = []
        length_list = []
        for dataset in datasets:
            print(f"Loading dataset {dataset} ...")
            dataset_path = os.path.join(opt.data_root, dataset)
            sequences_path = os.path.join(dataset_path, 'sequences_canonical')
            sequences_names = os.listdir(sequences_path)
            if debug or 1:
                L=len(sequences_names)
                L_debug=min(L,200)
                sequences_names=sequences_names[:L_debug]
                # sequences_names=sequences_names[:100]
            for name in tqdm(sequences_names):
                try:
                    motion = np.load(pjoin(sequences_path, name, 'motion.npy'))
                    obj = np.load(pjoin(sequences_path, name, 'object.npz'), allow_pickle=True)
                    # load obj points----------------
                    obj_name = str(obj['name'])

                    id_name = obj_name if obj_split else name

                    if id_name in id_list:
                        continue                    
                    obj_path = pjoin(dataset_path, 'objects')
                    # mesh_path = os.path.join(obj_path, obj_name, obj_name+'.obj')
                    # temp_simp = trimesh.load(mesh_path)

                    # obj_points = np.array(temp_simp.vertices)
                    # obj_faces = np.array(temp_simp.faces)

                    # center the meshes
                    # center = np.mean(obj_points, 0)
                    # obj_points -= center
                    # obj_points = obj_points.astype(np.float32)


                    # bps 
                    bps_path = pjoin(dataset_path, 'objects_bps/{}/{}_1024.npy'.format(obj_name,obj_name))
                    obj_bps = np.load(bps_path, allow_pickle=True)
                    # print(obj_bps)
                    # sample object points
                    # obj_sample_path = pjoin(dataset_path, 'object_sample/{}.npy'.format(name))
                    # o_choose = np.load(obj_sample_path)
                                    
                    obj_sample_path = pjoin(obj_path, '{}/sample_points.npy'.format(obj_name))        
                    # o_choose = np.arange(200)

                    obj_points = np.load(obj_sample_path)
                    obj_normals = np.zeros((400, 3)) 

                    if (len(motion)) < min_motion_len or (len(motion) > 400):
                        continue
    
                    # TODO: hardcode
                    motion = motion[:300].astype(np.float32)

                    # contact_input = np.load(pjoin(opt.data_root, 'affordance_data/contact_'+name + '.npy'), allow_pickle=True)[None][0]


                    text_data = []
                    flag = False
                    with cs.open(pjoin(sequences_path, name, 'text.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = 0.0
                            to_tag = 0.0
                            # f_tag = float(line_split[2])
                            # to_tag = float(line_split[3])
                            # f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            # to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
            
                                    data_dict[name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict],
                                                        'seq_name': name,
                                                        'obj_points': obj_points,
                                                        'obj_normals':obj_normals,
                                                        'obj_bps': obj_bps,
                                                        # 'gt_afford_labels': contact_input
                                                    }
                                    new_name_list.append(name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    # print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = {'motion': motion,
                                            'length': len(motion),
                                            'text': text_data,
                                            'seq_name': name,
                                            'obj_points': obj_points,
                                            'obj_normals':obj_normals,
                                            'obj_bps': obj_bps,
                                            # 'gt_afford_labels':contact_input
                                        }

                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as err:
                    print(err.__class__.__name__) 
                    print(err)
                    print(name) 
                    pass
        
        # correct_datasets = ['behave', 'intercap', 'omomo']
        # for dataset in correct_datasets:
        #     print(f"Loading dataset {dataset} ...")
        #     dataset_path = os.path.join('./data_correct', dataset)
        #     obj_dataset_path = os.path.join(opt.data_root, dataset)
        #     sequences_path = os.path.join(dataset_path, 'sequences_canonical')
        #     sequences_names = os.listdir(sequences_path)
        #     for name in tqdm(sequences_names):
        #         try:
        #             motion = np.load(pjoin(sequences_path, name, 'motion.npy'))
        #             obj = np.load(pjoin(sequences_path, name, 'object.npz'), allow_pickle=True)
        #             # load obj points----------------
        #             obj_name = str(obj['name'])

        #             id_name = obj_name if obj_split else name

        #             if id_name in id_list:
        #                 continue                    
        #             obj_path = pjoin(obj_dataset_path, 'objects')

        #             # bps 
        #             bps_path = pjoin(obj_dataset_path, 'objects_bps/{}/{}_1024.npy'.format(obj_name,obj_name))
        #             obj_bps = np.load(bps_path, allow_pickle=True)
  
        #             obj_sample_path = pjoin(obj_path, '{}/sample_points.npy'.format(obj_name))        
        #             # o_choose = np.arange(200)

        #             obj_points = np.load(obj_sample_path)
        #             obj_normals = np.zeros((400, 3)) 


        #             if (len(motion)) < min_motion_len or (len(motion) > 400):
        #                 continue

        #             # TODO: hardcode
        #             if (len(motion) > 300):
        #                 idx = random.randint(0, len(motion) - 300)
        #                 motion = motion[idx:idx+300].astype(np.float32)


        #             text_data = []
        #             flag = False
        #             with cs.open(pjoin(sequences_path, name, 'text.txt')) as f:
        #                 for line in f.readlines():
        #                     text_dict = {}
        #                     line_split = line.strip().split('#')
        #                     caption = line_split[0]
        #                     tokens = line_split[1].split(' ')
        #                     f_tag = to_tag = 0.0
        #                     # f_tag = float(line_split[2])
        #                     # to_tag = float(line_split[3])
        #                     # f_tag = 0.0 if np.isnan(f_tag) else f_tag
        #                     # to_tag = 0.0 if np.isnan(to_tag) else to_tag

        #                     text_dict['caption'] = caption
        #                     text_dict['tokens'] = tokens
        #                     if f_tag == 0.0 and to_tag == 0.0:
        #                         flag = True
        #                         text_data.append(text_dict)
        #                     else:
        #                         try:
        #                             n_motion = motion[int(f_tag*20) : int(to_tag*20)]
        #                             if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
        #                                 continue
        #                             # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
            
        #                             data_dict[name] = {'motion': n_motion,
        #                                                 'length': len(n_motion),
        #                                                 'text':[text_dict],
        #                                                 'seq_name': name,
        #                                                 'obj_points': obj_points,
        #                                                 'obj_normals':obj_normals,
        #                                                 'obj_bps': obj_bps,
        #                                                 # 'gt_afford_labels': contact_input
        #                                             }
        #                             new_name_list.append(name)
        #                             length_list.append(len(n_motion))
        #                         except:
        #                             print(line_split)
        #                             print(line_split[2], line_split[3], f_tag, to_tag, name)
        #                             # break

        #             if flag:
        #                 data_dict[name] = {'motion': motion,
        #                                     'length': len(motion),
        #                                     'text': text_data,
        #                                     'seq_name': name,
        #                                     'obj_points': obj_points,
        #                                     'obj_normals':obj_normals,
        #                                     'obj_bps': obj_bps,
        #                                     # 'gt_afford_labels':contact_input
        #                                 }

        #                 new_name_list.append(name)
        #                 length_list.append(len(motion))
        #         except Exception as err:
        #             print(err.__class__.__name__) 
        #             print(err)
        #             print(name) 
        #             pass

        # aug_datasets = ['behave', 'intercap', 'omomo', 'imhd', 'neuraldome', 'grab']
        # for dataset in aug_datasets:
        #     print(f"Loading dataset {dataset} ...")
        #     dataset_path = os.path.join('./augment_data_new_version/aug_data_3', dataset)
        #     obj_dataset_path = os.path.join(opt.data_root, dataset)
        #     sequences_path = os.path.join(dataset_path, 'sequences_canonical')
        #     sequences_names = os.listdir(sequences_path)
        #     for name in tqdm(sequences_names):
        #         try:
        #             motion = np.load(pjoin(sequences_path, name, 'motion.npy'))
        #             obj = np.load(pjoin(sequences_path, name, 'object.npz'), allow_pickle=True)
        #             # load obj points----------------
        #             obj_name = str(obj['name'])

        #             id_name = obj_name if obj_split else name

        #             if id_name in id_list:
        #                 continue                    
        #             obj_path = pjoin(obj_dataset_path, 'objects')

        #             # bps 
        #             bps_path = pjoin(obj_dataset_path, 'objects_bps/{}/{}_1024.npy'.format(obj_name,obj_name))
        #             obj_bps = np.load(bps_path, allow_pickle=True)
  
        #             obj_sample_path = pjoin(obj_path, '{}/sample_points.npy'.format(obj_name))        
        #             # o_choose = np.arange(200)

        #             obj_points = np.load(obj_sample_path)
        #             obj_normals = np.zeros((400, 3)) 


        #             if (len(motion)) < min_motion_len or (len(motion) > 400):
        #                 continue

        #             # TODO: hardcode
        #             if (len(motion) > 300):
        #                 idx = random.randint(0, len(motion) - 300)
        #                 motion = motion[idx:idx+300].astype(np.float32)


        #             text_data = []
        #             flag = False
        #             with cs.open(pjoin(sequences_path, name, 'text.txt')) as f:
        #                 for line in f.readlines():
        #                     text_dict = {}
        #                     line_split = line.strip().split('#')
        #                     caption = line_split[0]
        #                     tokens = line_split[1].split(' ')
        #                     f_tag = to_tag = 0.0
        #                     # f_tag = float(line_split[2])
        #                     # to_tag = float(line_split[3])
        #                     # f_tag = 0.0 if np.isnan(f_tag) else f_tag
        #                     # to_tag = 0.0 if np.isnan(to_tag) else to_tag

        #                     text_dict['caption'] = caption
        #                     text_dict['tokens'] = tokens
        #                     if f_tag == 0.0 and to_tag == 0.0:
        #                         flag = True
        #                         text_data.append(text_dict)
        #                     else:
        #                         try:
        #                             n_motion = motion[int(f_tag*20) : int(to_tag*20)]
        #                             if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
        #                                 continue
        #                             # new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
            
        #                             data_dict[name] = {'motion': n_motion,
        #                                                 'length': len(n_motion),
        #                                                 'text':[text_dict],
        #                                                 'seq_name': name,
        #                                                 'obj_points': obj_points,
        #                                                 'obj_normals':obj_normals,
        #                                                 'obj_bps': obj_bps,
        #                                                 # 'gt_afford_labels': contact_input
        #                                             }
        #                             new_name_list.append(name)
        #                             length_list.append(len(n_motion))
        #                         except:
        #                             print(line_split)
        #                             print(line_split[2], line_split[3], f_tag, to_tag, name)
        #                             # break

        #             if flag:
        #                 data_dict[name] = {'motion': motion,
        #                                     'length': len(motion),
        #                                     'text': text_data,
        #                                     'seq_name': name,
        #                                     'obj_points': obj_points,
        #                                     'obj_normals':obj_normals,
        #                                     'obj_bps': obj_bps,
        #                                     # 'gt_afford_labels':contact_input
        #                                 }

        #                 new_name_list.append(name)
        #                 length_list.append(len(motion))
        #         except Exception as err:
        #             print(err.__class__.__name__) 
        #             print(err)
        #             print(name) 
        #             pass

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        data = data * self.std[:data.shape[-1]] + self.mean[:data.shape[-1]]
        return data

    def inv_transform_th(self, data):
        data = data * torch.from_numpy(self.std).to(
            data.device) + torch.from_numpy(self.mean).to(data.device)
        return data


    def __len__(self):
        return len(self.data_dict) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list, seq_name, obj_points, obj_normals, obj_bps = data['motion'], data['length'], data['text'], data['seq_name'],  data['obj_points'], data['obj_normals'], data['obj_bps']

        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']


        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)

        pos_one_hots = []
        word_embeddings = []
        # for token in tokens:

        #     if len(token.split('/'))<2:
        #         print(f" {seq_name}   {tokens}")
        #         break
        #     word_emb, pos_oh = self.w_vectorizer[token]
        #     pos_one_hots.append(pos_oh[None, :])
        #     word_embeddings.append(word_emb[None, :])
        # pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        # word_embeddings = np.concatenate(word_embeddings, axis=0)

        # Crop the motions in to times of 4, and introduce small variations
        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'
        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        
        # if not self.opt.use_global:
        #     "Z Normalization"
        #     motion = np.copy(motion)
        #     if len(self.mean) == 559:
        #         motion[:,:559] = (motion[:, :559] - self.mean[:559]) / self.std[:559]
        #     else:
        #         #  for evaluation of ground truth
        #         motion[..., :553] = (motion[..., :553] - self.mean[:553]) / self.std[:553]

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)


        # Contact labels here for evaluation!
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), seq_name, obj_points, obj_normals, obj_bps



# A wrapper class for behave dataset t2m and t2afford
class Behave(data.Dataset):
    def __init__(self, mode, 
                    datapath='./assets/opt.txt', 
                    split="train",
                    dataset = 'interact',
                    obj_split=False,
                    use_global=False,
                    training_stage=1,
                    wo_obj_motion=False,
                    debug=0,
                    **kwargs):
        self.mode = mode
        self.debug=debug


        self.dataset_name = 't2m_behave'
        self.dataname = 't2m_behave'

        # Configurations of T2M dataset and KIT dataset is almost the same
        abs_base_path = f'.'
        dataset_opt_path = pjoin(abs_base_path, datapath)
        device = None  # torch.device('cuda:4') # This param is not in use in this context
        opt = get_opt(dataset_opt_path, device, use_global, wo_obj_motion)
        opt.meta_dir = pjoin(abs_base_path, opt.meta_dir)
        
        opt.model_dir = pjoin(abs_base_path, opt.model_dir)
        opt.checkpoints_dir = pjoin(abs_base_path, opt.checkpoints_dir)
        opt.save_root = pjoin(abs_base_path, opt.save_root)
        opt.meta_dir = './data'
        if dataset == 'interact':
            self.datasets = [ 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab','behave','intercap']
        elif dataset == 'interact_wobehave':
            self.datasets = [ 'neuraldome', 'chairs', 'omomo', 'imhd']
        elif dataset == 'interact_wobehave_correct':
            self.datasets = [ 'neuraldome', 'chairs', 'omomo_correct', 'imhd', 'grab']
            # self.datasets = ['behave', 'intercap', 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab']
            # self.datasets = ['neuraldome', 'chairs', 'imhd', 'grab']
        elif dataset == 'interact_correct':
            self.datasets = [ 'neuraldome', 'chairs', 'omomo_correct', 'imhd', 'grab','behave_correct','intercap_correct']
        elif dataset == 'interact_aug':
            self.datasets = [ 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab','behave','intercap','behave_aug3','grab_aug3','imhd_aug3','intercap_aug3','neuraldome_aug3']
        elif dataset == 'interact_aug_latest':
            self.datasets = [ 'neuraldome', 'chairs', 'omomo_correct', 'imhd', 'grab','behave_correct','intercap_correct','behave_aug_new','omomo_aug_new','imhd_aug_new','intercap_aug_new','neuraldome_aug_new']
        # elif dataset == 'interact_correct':
        #     self.datasets = [ 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab','behave','intercap']
        else:
            self.datasets = [dataset]
        self.opt = opt
        self.use_global = use_global
        self.training_stage = training_stage
        print('Loading dataset %s ...' % self.datasets)

        if  self.training_stage==1:
            self.split_file = pjoin(opt.meta_dir, f'{split}.txt')     #   adopt augmented data for affordance training
            if mode == 'text_only':
                self.t2m_dataset = TextOnlyAffordDataset(self.opt, self.split_file)
            else:
                self.w_vectorizer = None # WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
                self.t2m_dataset = Text2AffordDataset(self.opt,  self.split_file, self.w_vectorizer)

        elif  self.training_stage>=2:

            # used by our models
            self.mean = np.load(pjoin(opt.meta_dir, 'Mean_all_can_new.npy'))
            self.std = np.load(pjoin(opt.meta_dir, 'Std_all_can_new.npy'))



            self.split_file = pjoin(opt.meta_dir, 'test.txt') if obj_split else pjoin(opt.meta_dir, 'test_seq.txt')

            if mode == 'text_only':
                self.t2m_dataset = TextOnlyDataset(self.opt, self.mean, self.std, self.datasets, self.split_file, obj_split,self.debug)
            else:
                self.w_vectorizer = None # WordVectorizer(pjoin(abs_base_path, 'glove'), 'our_vab')
                self.t2m_dataset = Text2MotionDatasetV3(self.opt, self.mean, self.std, self.datasets, self.split_file, obj_split, self.w_vectorizer,self.debug)
                self.num_actions = 1 # dummy placeholder

        else:
            print(f"error!")
        
        assert len(self.t2m_dataset) > 1, 'You loaded an empty dataset, ' \
                                          'it is probably because your data dir has only texts and no motions.\n' \
                                          'To train and evaluate MDM you should get the FULL data as described ' \
                                          'in the README file.'


        # Load necessay variables for converting raw motion to processed data


        # # Get offsets of target skeleton
        # example_data = np.load(data_dir)
        # example_data = example_data.reshape(len(example_data), -1, 3)
        # example_data = torch.from_numpy(example_data)
        # tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, 'cpu')
        # # (joints_num, 3)
        # tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])

    def __getitem__(self, item):
        return self.t2m_dataset.__getitem__(item)

    def __len__(self):
        return self.t2m_dataset.__len__()


    def motion_to_rel_data(self, motion, model, is_norm=False):

        motion_bu = motion.detach().clone()
        # Right/Left foot
        fid_r, fid_l = [61, 52, 53, 40, 34, 49, 40], [29, 30, 18, 19, 7, 2, 15]
        # Face direction, r_hip, l_hip, sdr_r, sdr_l
        
        sample_rel_np_list = []
        for ii in range(len(motion)):
            # Data need to be [120 (timestep), 22, 3] to get feature
            sample_rel = get_human_representation(
                motion[ii].detach().cpu().clone().permute(2, 0,
                                                          1).cpu().numpy(),
                0.002, fid_r, fid_l)
            # Duplicate last motion step to match the size
            sample_rel = torch.from_numpy(sample_rel).unsqueeze(0).float()
            # sample_rel = torch.cat(
            #     [sample_rel, sample_rel[0:1, -1:, :].clone()], dim=1)
            
            # Normalize with relative normalization
            if is_norm:
                sample_rel = (sample_rel - self.mean_rel[:553]) / self.std_rel[:553]
            sample_rel = sample_rel.unsqueeze(1).permute(0, 3, 1, 2)
            sample_rel = sample_rel.to(motion.device)
            sample_rel_np_list.append(sample_rel)

        processed_data = torch.cat(sample_rel_np_list, axis=0)



        n_markers = 77
        # NOTE: check if the sequence is still that same after extract_features and converting back
        # sample = dataset.t2m_dataset.inv_transform(sample_abs.cpu().permute(0, 2, 3, 1)).float()
        # sample_after = (processed_data.permute(0, 2, 3, 1) * self.std_rel) + self.mean_rel
        
        
        # print(f"processed_data:{processed_data.shape}  {sample_after.shape}")
        # B, _, T , F = sample_after.shape
        # sample_after = sample_after[..., :66].reshape(B, T, n_joints, 3).permute(0,2,3,1)

        # sample_after = recover_from_ric(sample_after, n_joints)
        # sample_after = sample_after.view(-1, *sample_after.shape[2:]).permute(0, 2, 3, 1)

        # rot2xyz_pose_rep = 'xyz'
        # rot2xyz_mask = None
        # sample_after = model.rot2xyz(x=sample_after,
        #                     mask=rot2xyz_mask,
        #                     pose_rep=rot2xyz_pose_rep,
        #                     glob=True,
        #                     translation=True,
        #                     jointstype='smpl',
        #                     vertstrans=True,
        #                     betas=None,
        #                     beta=0,
        #                     glob_rot=None,
        #                     get_rotations_back=False)

        # from data_loaders.humanml.utils.plot_script import plot_3d_motion


        # for i in range(motion.shape[0]):
        #     # print(f"test:{ sample_after.shape}   {motion[2].permute(2,0,1).shape}")
        #     plot_3d_motion("./test_positions_{}.mp4".format(i), self.kinematic_chain, motion[i].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)
        #     plot_3d_motion("./test_positions_1_after{}.mp4".format(i), self.kinematic_chain, sample_after[i].permute(2,0,1).detach().cpu().numpy(), 'title', 'humanml', fps=20)

        # Return data already normalized with relative mean and std. shape [bs, 553, 1, 120(motion step)]
        return processed_data







