from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m_behave_collate, t2m_contact_collate, t2m_behave_bps_collate
from dataclasses import dataclass
import torch
import torch.utils.data.distributed as dist_utils



def get_dataset_class(name):
    if name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name.startswith('interact') or name in ['behave', 'intercap', 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab', 'interact','interact_wobehave','interact_wobehave_correct','interact_aug_latest']:
        from data_loaders.behave.data.dataset import Behave
        return Behave
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    num_frames: int
    obj_split: bool = False
    split: str = 'train'
    hml_mode: str = 'train'
    training_stage: int = 1
    debug: int = 0


def get_collate_fn(name, hml_mode='train', training_stage=1):
    if hml_mode == 'gt' and name in ["humanml", "kit"]:
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if hml_mode == 'gt' and name in ["behave"]:
        from data_loaders.behave.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if hml_mode == 'gt' and name in ["omomo"]:
        from data_loaders.omomo.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    elif name in ["behave"] and training_stage==1:
        return t2m_contact_collate
    elif name in ["behave"] and training_stage==2:
        return t2m_behave_collate
    elif (name.startswith('interact') or name in ['behave', 'intercap', 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab', 'interact','interact_wobehave','interact_wobehave_correct']) and training_stage==3:
        return t2m_behave_bps_collate
    else:
        return all_collate


def get_dataset(conf: DatasetConfig):
    DATA = get_dataset_class(conf.name)
    if conf.name.startswith('interact') or conf.name in ['behave', 'intercap', 'neuraldome', 'chairs', 'omomo', 'imhd', 'grab', 'interact','interact_wobehave','interact_wobehave_correct']:
        dataset = DATA(split=conf.split,
                       mode=conf.hml_mode,
                       dataset=conf.name,
                       obj_split=conf.obj_split,
                       num_frames=conf.num_frames,
                       training_stage=conf.training_stage,debug=conf.debug)
    else:
        raise NotImplementedError()
        dataset = DATA(split=split, num_frames=num_frames)
    return dataset



def get_dataset_loader(conf: DatasetConfig):
    # name, batch_size, num_frames, split='train', hml_mode='train'
    dataset = get_dataset(conf)
    collate = get_collate_fn(conf.name, conf.hml_mode, conf.training_stage)

    if conf.hml_mode == 'train':
        loader = DataLoader(
            dataset, batch_size=conf.batch_size, shuffle=True,
            num_workers=8, drop_last=True, collate_fn=collate,
        )
    else:
        loader = DataLoader(
            dataset, batch_size=conf.batch_size, shuffle=False,
            num_workers=8, drop_last=True, collate_fn=collate,
        )
    return loader