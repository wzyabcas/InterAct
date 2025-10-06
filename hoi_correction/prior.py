
"""
hand prior for SMPL-H
Author: Xianghui, 12 January 2022
"""
import numpy as np
import torch
import pickle as pkl
from os.path import join
# import pickle as pkl
# from os.path import join
# import torch
# import numpy as np


def grab_prior(root_path):
    lhand_data, rhand_data = load_grab_prior(root_path)

    prior = np.concatenate([lhand_data['mean'], rhand_data['mean']], axis=0)
    lhand_prec = lhand_data['precision']
    rhand_prec = rhand_data['precision']

    return prior, lhand_prec, rhand_prec


def load_grab_prior(root_path):
    lhand_path = join(root_path, 'priors', 'lh_prior.pkl')
    rhand_path = join(root_path, 'priors', 'rh_prior.pkl')
    lhand_data = pkl.load(open(lhand_path, 'rb'))
    rhand_data = pkl.load(open(rhand_path, 'rb'))
    return lhand_data, rhand_data


def mean_hand_pose(root_path):
    "mean hand pose computed from grab dataset"
    lhand_data, rhand_data = load_grab_prior(root_path)
    lhand_mean = np.array(lhand_data['mean'])
    rhand_mean = np.array(rhand_data['mean'])
    mean_pose = np.concatenate([lhand_mean, rhand_mean])
    return mean_pose


class HandPrior:
    HAND_POSE_NUM=45
    def __init__(self, prior_path,
                 prefix=66,
                 device='cuda:0',
                 dtype=torch.float,
                 type='grab'):
        "prefix is the index from where hand pose starts, 66 for SMPL-H"
        self.prefix = prefix
        if type == 'grab':
            prior, lhand_prec, rhand_prec = grab_prior(prior_path)
            self.mean = torch.tensor(prior, dtype=dtype).unsqueeze(axis=0).to(device)
            self.lhand_prec = torch.tensor(lhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
            self.rhand_prec = torch.tensor(rhand_prec, dtype=dtype).unsqueeze(axis=0).to(device)
            print(lhand_prec.shape,'JJJJ')
        else:
            raise NotImplemented("Only grab hand prior is supported!")

    def __call__(self, full_pose,left_or_right):
        "full_pose also include body poses, this function can be used to compute loss"
        #print(full_pose.shape,'FP')
        if left_or_right==0:
            temp = full_pose[:, :] - self.mean[:,:45]
            lhand = torch.matmul(temp[:, :], self.lhand_prec)
            return lhand
        else:
            temp = full_pose[:, :] - self.mean[:,45:]
            rhand = torch.matmul(temp[:, :], self.rhand_prec)
            return rhand
        # if self.lhand_prec is None:
        #     return (temp*temp).sum(dim=1)
        # # else:
        #     lhand = torch.matmul(temp[:, :self.HAND_POSE_NUM], self.lhand_prec)
        #     rhand = torch.matmul(temp[:, self.HAND_POSE_NUM:], self.rhand_prec)
        #     temp2 = torch.cat([lhand, rhand], axis=1)
        #     return (temp2 * temp2).sum(dim=1)
"""
If code works:
    Author: Bharat
else:
    Author: Anonymous
"""



def get_prior(model_root, gender='male', precomputed=True):
    if precomputed:
        prior = Prior(sm=None, model_root=model_root)
        return prior['Generic']
    else:
        raise NotImplemented


class ThMahalanobis(object):
    def __init__(self, mean, prec, prefix, end=66, device=torch.device("cuda:0")):
        self.mean = torch.tensor(mean.astype('float32'), requires_grad=False).unsqueeze(axis=0).to(device)
        self.prec = torch.tensor(prec.astype('float32'), requires_grad=False).to(device)
        self.prefix = prefix
        self.end = end

    def __call__(self, pose, prior_weight=1.):
        '''
        :param pose: Batch x pose_dims
        :return: weighted L2 distance of the N pose parameters, where N = 72 - prefix for SMPL model, for smplh, only compute from 3 to 66
        '''
        # return (pose[:, self.prefix:] - self.mean)*self.prec
        temp = pose[:, self.prefix:self.end] - self.mean
        temp2 = torch.matmul(temp, self.prec) * prior_weight
        return torch.sum((temp2 * temp2))#.sum(dim=1)
        

class Prior(object):
    def __init__(self, prefix=0, end=63, device=torch.device("cuda:0")):
        "end=66 for smplh, 69 for smpl"
        self.prefix = prefix
        self.device = device
        self.end = end
        
        model_root='./assets'
        file = join(model_root, 'priors', 'body_prior.pkl')
        dat = pkl.load(open(file, 'rb'))
        self.priors =  ThMahalanobis(dat['mean'],
                                                dat['precision'],
                                                self.prefix,
                                                self.end,
                                                self.device)
    def forward(self,pose,prior_weight=1.0):
        loss=self.priors(pose,prior_weight)
        return loss
        # temp = pose[:, self.prefix:self.end] - self.mean
        # temp2 = torch.matmul(temp, self.prec) * prior_weight
        # return (temp2 * temp2).sum(dim=1)


    # def __getitem__(self, pid):
    #     if pid not in self.priors:
    #         samples = [p[self.prefix:] for qsub in self.pose_subjects
    #                    for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])
    #                    if pid in name.lower()]
    #         self.priors[pid] = self.priors['Generic'] if len(samples) < 3 \
    #                            else self.create_prior_from_samples(samples)

    #     return self.priors[pid]
